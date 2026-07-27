pub mod template;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use futures_util::{SinkExt, StreamExt};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio_tungstenite::tungstenite::Message;

use ehp_core::constraints::{self, DiffVar};
use ehp_core::io;
use ehp_core::page::SATPage;
use ehp_core::pageturning;
use ehp_core::result::SATResult;
use ehp_core::seqsee::{self, DiffOverlay};
use ehp_core::solver;

// =============================================================================
// Message protocol
// =============================================================================

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    /// Differential overlay: edges + uncertain node IDs.
    DiffOverlay { overlay: DiffOverlay },
    /// Status bar text.
    SolverStatus { text: String },
    /// Tell client to reload (page structure changed).
    Reload,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    AddDifferential {
        source: [i32; 3],
        #[serde(rename = "sourceIdx")]
        source_idx: usize,
        target: [i32; 3],
        #[serde(rename = "targetIdx")]
        target_idx: usize,
        value: bool,
    },
    ToggleDifferential {
        source: [i32; 3],
        #[serde(rename = "sourceIdx")]
        source_idx: usize,
        target: [i32; 3],
        #[serde(rename = "targetIdx")]
        target_idx: usize,
    },
    Solve,
    TurnPage,
    LoadPage {
        path: String,
        r: i32,
        #[serde(rename = "maxTotal")]
        max_total: i32,
    },
}

// =============================================================================
// Server state
// =============================================================================

pub struct ServerState {
    pub page: SATPage,
    pub result: Option<SATResult>,
    pub known_diffs: hashbrown::HashMap<DiffVar, bool>,
    pub load_prefix: String,
    pub load_max_total: i32,
    pub seqsee_dir: String,
}

impl ServerState {
    pub fn new(
        prefix: &str,
        r: i32,
        max_total: i32,
        known_diffs_file: Option<&str>,
        seqsee_dir: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t0 = Instant::now();
        info!("Loading E_{} page from {} (max_t={})...", r, prefix, max_total);
        let page = io::load_page(prefix, r, max_total)?;
        info!("  load_page: {:.2}s", t0.elapsed().as_secs_f64());

        let nonzero = page.dimension.values().filter(|&&d| d > 0).count();
        info!("Loaded: {} tridegrees with elements", nonzero);

        let t2 = Instant::now();
        let known_diffs = load_known_diffs(r, known_diffs_file)?;
        let result = Self::run_solve(&page, &known_diffs);
        info!("  solve: {:.2}s", t2.elapsed().as_secs_f64());

        Ok(ServerState {
            page,
            result,
            known_diffs,
            load_prefix: prefix.to_string(),
            load_max_total: max_total,
            seqsee_dir: seqsee_dir.to_string(),
        })
    }

    pub fn run_solve(
        page: &SATPage,
        known_diffs: &hashbrown::HashMap<DiffVar, bool>,
    ) -> Option<SATResult> {
        let cutoff = page.max_t.unwrap_or(0);
        info!("Building constraint system (cutoff={})...", cutoff);

        let system = constraints::build_constraint_system(page, cutoff, known_diffs);
        info!(
            "  {} vars, {} constraints",
            system.num_vars,
            system.num_constraints()
        );

        info!("Solving...");
        let result = solver::solve(&system);

        if let Some(ref res) = result {
            let determined = system.num_vars - res.unknown.len();
            info!("  {}/{} vars determined", determined, system.num_vars);
        } else {
            warn!("No solution (inconsistent or empty system)");
        }

        result
    }

    pub fn build_overlay(&self) -> DiffOverlay {
        if let Some(ref result) = self.result {
            seqsee::build_diff_overlay(&self.page, result)
        } else {
            DiffOverlay {
                edges: vec![],
                uncertain: vec![],
            }
        }
    }

    pub fn add_known_diff(&mut self, n: i32, s: i32, f: i32, row: u16, col: u16, value: bool) {
        let n_var = n.min(s + 2);
        let dv = DiffVar::new(n_var, s, f, row, col);
        self.known_diffs.insert(dv, value);
    }

    pub fn toggle_known_diff(&mut self, n: i32, s: i32, f: i32, row: u16, col: u16) {
        let n_var = n.min(s + 2);
        let dv = DiffVar::new(n_var, s, f, row, col);
        if let Some(val) = self.known_diffs.get(&dv) {
            let new_val = !val;
            self.known_diffs.insert(dv, new_val);
        } else {
            self.known_diffs.insert(dv, true);
        }
    }

    pub fn re_solve(&mut self) {
        self.result = Self::run_solve(&self.page, &self.known_diffs);
    }

    pub fn turn_page(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let result = self.result.as_ref().ok_or("No solve result — solve first")?;

        info!(
            "Turning page E_{} -> E_{}...",
            self.page.r,
            self.page.r + 1
        );
        let (next_page, _turned) = pageturning::build_next_page(&self.page, result)?;

        let nonzero = next_page.dimension.values().filter(|&&d| d > 0).count();
        info!("E_{} has {} tridegrees with elements", next_page.r, nonzero);

        self.page = next_page;
        self.known_diffs.clear();
        self.result = None;

        Ok(())
    }

    pub fn load_page(
        &mut self,
        prefix: &str,
        r: i32,
        max_total: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        info!("Loading E_{} page from {}...", r, prefix);
        let page = io::load_page(prefix, r, max_total)?;

        self.page = page;
        self.known_diffs.clear();
        self.result = None;
        self.load_prefix = prefix.to_string();
        self.load_max_total = max_total;

        Ok(())
    }

    pub fn solver_status_text(&self) -> String {
        if let Some(ref res) = self.result {
            let total = res.vars.len();
            let determined = total - res.unknown.len();
            format!(
                "E_{} | {}/{} vars determined | {} unknown tridegrees",
                self.page.r,
                determined,
                total,
                res.unknown_tridegrees().len()
            )
        } else {
            format!("E_{} | not yet solved", self.page.r)
        }
    }
}

pub fn load_known_diffs(
    r: i32,
    file: Option<&str>,
) -> Result<hashbrown::HashMap<DiffVar, bool>, Box<dyn std::error::Error>> {
    load_known_diffs_inner(r, file, None)
}

/// Page-aware form of [`load_known_diffs`]: outside-knowledge-base rows are
/// pruned to Python parity for THIS page (see
/// `ehp_core::constraints::prune_outside_diffs`) — rows at excluded /
/// over-kept / edge degrees, whose recorded `[row, col]` coordinates cannot
/// be trusted in this pipeline's basis, are dropped with a warning instead of
/// force-enforced (which made E4 UNSAT at t=80 with correct data).
/// `EHP_OUTSIDE_PARITY=0` disables the pruning. Explicit per-page files and
/// hardcoded diffs are never pruned (they are asserted in THIS pipeline's
/// basis).
pub fn load_known_diffs_for_page(
    page: &ehp_core::page::SATPage,
    file: Option<&str>,
) -> Result<hashbrown::HashMap<DiffVar, bool>, Box<dyn std::error::Error>> {
    load_known_diffs_inner(page.r, file, Some(page))
}

fn load_known_diffs_inner(
    r: i32,
    file: Option<&str>,
    page: Option<&ehp_core::page::SATPage>,
) -> Result<hashbrown::HashMap<DiffVar, bool>, Box<dyn std::error::Error>> {
    // Externally recorded differentials (the Python pipeline's
    // `outside_diffs` knowledge base): every CSV in $EHP_OUTSIDE_DIFFS is
    // scanned for rows `[r, n, s, f, row, col, value]` matching this page.
    // These load first; a page-specific file (below) overrides on conflict.
    let mut diffs = match std::env::var("EHP_OUTSIDE_DIFFS") {
        Ok(dir) if !dir.is_empty() => {
            let mut d = load_outside_diffs(r, &dir)?;
            let parity = std::env::var("EHP_OUTSIDE_PARITY").map_or(true, |v| v != "0");
            if let (Some(page), true) = (page, parity) {
                let total = d.len();
                let dropped = ehp_core::constraints::prune_outside_diffs(page, &mut d);
                if !dropped.is_empty() {
                    warn!(
                        "EHP_OUTSIDE_DIFFS: {} of {} E_{} rows NOT enforced — their [row,col] \
                         coordinates are unreliable on this page (EHP_OUTSIDE_PARITY=0 forces them):",
                        dropped.len(),
                        total,
                        r,
                    );
                    for (dv, val, rsn) in &dropped {
                        warn!(
                            "  d_{}({},{},{})[{},{}] = {} — {}",
                            r, dv.n, dv.s, dv.f, dv.row, dv.col, *val as u8, rsn,
                        );
                    }
                }
            }
            d
        }
        _ => hashbrown::HashMap::new(),
    };

    if let Some(path) = file {
        let file_diffs = io::load_known_diffs(path, r)?;
        info!("Loaded {} known differentials from {}", file_diffs.len(), path);
        diffs.extend(file_diffs);
        return Ok(diffs);
    }

    let hardcoded: &[(DiffVar, bool)] = match r {
        2 => &[(DiffVar::new(17, 15, 1, 0, 0), true)],
        3 => &[(DiffVar::new(17, 15, 2, 0, 0), true)],
        4 => &[(DiffVar::new(40, 38, 2, 0, 0), true)],
        _ => &[],
    };
    let mut used_hardcoded = 0;
    for &(dv, v) in hardcoded {
        if let Some(&prev) = diffs.get(&dv) {
            if prev != v {
                warn!(
                    "outside diff d_{}({},{},{})[{},{}]={} conflicts with hardcoded value {} — keeping the outside value",
                    r, dv.n, dv.s, dv.f, dv.row, dv.col, prev as u8, v as u8,
                );
            }
        } else {
            diffs.insert(dv, v);
            used_hardcoded += 1;
        }
    }
    if used_hardcoded > 0 {
        info!("Using {} hardcoded known differentials", used_hardcoded);
    }
    Ok(diffs)
}

/// Load the externally recorded differentials for page `r` from a directory
/// of CSV files (the Python pipeline's `outside_diffs` format): each row is
/// `r, n, s, f, row, col, value`; every `.csv` file in the directory is
/// scanned and rows for other pages are skipped. `n` is normalized to the
/// stable representative `min(n, s+2)`, matching how differential variables
/// are keyed. Conflicting duplicate entries warn and keep the first value.
pub fn load_outside_diffs(
    r: i32,
    dir: &str,
) -> Result<hashbrown::HashMap<DiffVar, bool>, Box<dyn std::error::Error>> {
    let mut diffs = hashbrown::HashMap::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            warn!("EHP_OUTSIDE_DIFFS: cannot read {}: {}", dir, e);
            return Ok(diffs);
        }
    };
    // Files to skip, comma-separated substrings matched against the file
    // name (env `EHP_OUTSIDE_SKIP`). Defaults to "stable_Dan": the user has
    // ruled its rows out as an input source. Set EHP_OUTSIDE_SKIP="" to load
    // everything.
    let skip_patterns: Vec<String> = std::env::var("EHP_OUTSIDE_SKIP")
        .unwrap_or_else(|_| "stable_Dan".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let mut files = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("csv") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if let Some(pat) = skip_patterns.iter().find(|p| name.contains(p.as_str())) {
            info!(
                "EHP_OUTSIDE_DIFFS: skipping {} (matches EHP_OUTSIDE_SKIP pattern \"{}\")",
                name, pat,
            );
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            warn!("EHP_OUTSIDE_DIFFS: cannot read {}", path.display());
            continue;
        };
        files += 1;
        for (lineno, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<i64> = line
                .split(',')
                .map(|p| p.trim().parse::<i64>())
                .collect::<Result<_, _>>()
                .unwrap_or_default();
            if fields.len() != 7 {
                warn!(
                    "EHP_OUTSIDE_DIFFS: {}:{}: expected 7 integer fields, got {:?} — skipped",
                    path.display(),
                    lineno + 1,
                    line,
                );
                continue;
            }
            if fields[0] as i32 != r {
                continue;
            }
            let (n, s, f) = (fields[1] as i32, fields[2] as i32, fields[3] as i32);
            let dv = DiffVar::new(n.min(s + 2), s, f, fields[4] as u16, fields[5] as u16);
            let value = fields[6] != 0;
            if let Some(&prev) = diffs.get(&dv) {
                if prev != value {
                    warn!(
                        "EHP_OUTSIDE_DIFFS: conflicting values for d_{}({},{},{})[{},{}] — keeping {}",
                        r, dv.n, dv.s, dv.f, dv.row, dv.col, prev as u8,
                    );
                }
            } else {
                diffs.insert(dv, value);
            }
        }
    }
    info!(
        "Loaded {} outside differentials for E_{} from {} ({} csv files)",
        diffs.len(),
        r,
        dir,
        files,
    );
    Ok(diffs)
}

// =============================================================================
// SeqSee HTML generation
// =============================================================================

/// Generate the viewer HTML by running SeqSee's Python pipeline.
///
/// 1. Convert page data to SeqSee JSON (base chart, no differentials)
/// 2. Write JSON to temp file
/// 3. Run `python3 seqsee/main.py input.json output.html`
/// 4. Read output HTML
/// 5. Inject WebSocket overlay script
pub fn generate_seqsee_html(
    page: &SATPage,
    seqsee_dir: &str,
    ws_port: u16,
    max_total: i32,
) -> Result<String, Box<dyn std::error::Error>> {
    let title = format!("EHP E_{}", page.r);
    // Generate base chart JSON (no result → no differential edges)
    let json = seqsee::sat_page_to_seqsee_json(page, None, &title);

    let tmp = std::env::temp_dir();
    let json_path = tmp.join("ehp_seqsee_input.json");
    let html_path = tmp.join("ehp_seqsee_output.html");

    std::fs::write(&json_path, serde_json::to_string_pretty(&json)?)?;

    let main_py = Path::new(seqsee_dir).join("main.py");
    if !main_py.exists() {
        return Err(format!(
            "SeqSee main.py not found at {}. Set --seqsee-dir to the directory containing main.py.",
            main_py.display()
        )
        .into());
    }

    // Prefer Python from a virtualenv inside the SeqSee project directory.
    let seqsee_root = Path::new(seqsee_dir)
        .parent()
        .unwrap_or(Path::new(seqsee_dir));
    let python = ["venv/bin/python3", ".venv/bin/python3"]
        .iter()
        .map(|p| seqsee_root.join(p))
        .find(|p| p.exists())
        .unwrap_or_else(|| "python3".into());

    info!(
        "Running SeqSee pipeline: {} {} ...",
        python.display(),
        main_py.display()
    );

    let output = std::process::Command::new(&python)
        .arg(&main_py)
        .arg(&json_path)
        .arg(&html_path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("SeqSee pipeline failed:\n{}", stderr).into());
    }

    let base_html = std::fs::read_to_string(&html_path)?;
    info!(
        "SeqSee generated {} bytes of HTML (max_total={})",
        base_html.len(),
        max_total
    );

    Ok(template::inject_ws_script(&base_html, ws_port))
}

// =============================================================================
// WebSocket handler
// =============================================================================

pub async fn handle_ws_connection(
    stream: TcpStream,
    addr: SocketAddr,
    state: Arc<Mutex<ServerState>>,
    broadcast_tx: broadcast::Sender<String>,
    html_cache: Arc<RwLock<String>>,
    ws_port: u16,
) {
    info!("New WebSocket connection from {}", addr);

    let ws_stream = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            error!("WebSocket handshake failed for {}: {}", addr, e);
            return;
        }
    };

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Send initial overlay + status
    {
        let s = state.lock().await;
        let overlay_msg = ServerMessage::DiffOverlay {
            overlay: s.build_overlay(),
        };
        let json = serde_json::to_string(&overlay_msg).unwrap();
        if let Err(e) = ws_sender.send(Message::Text(json.into())).await {
            error!("Failed to send initial overlay to {}: {}", addr, e);
            return;
        }

        let status_msg = ServerMessage::SolverStatus {
            text: s.solver_status_text(),
        };
        let json = serde_json::to_string(&status_msg).unwrap();
        if let Err(e) = ws_sender.send(Message::Text(json.into())).await {
            error!("Failed to send status to {}: {}", addr, e);
            return;
        }
    }

    let mut broadcast_rx = broadcast_tx.subscribe();

    loop {
        tokio::select! {
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<ClientMessage>(&text) {
                            Ok(client_msg) => {
                                let response = handle_client_message(
                                    client_msg, &state, &html_cache, ws_port
                                ).await;
                                for resp in response {
                                    let json = serde_json::to_string(&resp).unwrap();
                                    let _ = broadcast_tx.send(json);
                                }
                            }
                            Err(e) => {
                                warn!("Invalid message from {}: {} (raw: {})", addr, e, text);
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        info!("Client {} disconnected", addr);
                        break;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        error!("WebSocket error from {}: {}", addr, e);
                        break;
                    }
                }
            }
            msg = broadcast_rx.recv() => {
                match msg {
                    Ok(json) => {
                        if let Err(e) = ws_sender.send(Message::Text(json.into())).await {
                            error!("Failed to send broadcast to {}: {}", addr, e);
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("Client {} lagged {} messages", addr, n);
                    }
                    Err(_) => break,
                }
            }
        }
    }
}

async fn handle_client_message(
    msg: ClientMessage,
    state: &Arc<Mutex<ServerState>>,
    html_cache: &Arc<RwLock<String>>,
    ws_port: u16,
) -> Vec<ServerMessage> {
    match msg {
        ClientMessage::AddDifferential {
            source,
            source_idx,
            target: _,
            target_idx,
            value,
        } => {
            let mut s = state.lock().await;
            s.add_known_diff(
                source[0],
                source[1],
                source[2],
                target_idx as u16,
                source_idx as u16,
                value,
            );
            info!(
                "Added known diff: ({},{},{}) [{}] -> [{}] = {}",
                source[0], source[1], source[2], source_idx, target_idx, value
            );
            s.re_solve();
            vec![
                ServerMessage::DiffOverlay {
                    overlay: s.build_overlay(),
                },
                ServerMessage::SolverStatus {
                    text: s.solver_status_text(),
                },
            ]
        }

        ClientMessage::ToggleDifferential {
            source,
            source_idx,
            target: _,
            target_idx,
        } => {
            let mut s = state.lock().await;
            s.toggle_known_diff(
                source[0],
                source[1],
                source[2],
                target_idx as u16,
                source_idx as u16,
            );
            info!(
                "Toggled diff: ({},{},{}) [{}] -> [{}]",
                source[0], source[1], source[2], source_idx, target_idx
            );
            s.re_solve();
            vec![
                ServerMessage::DiffOverlay {
                    overlay: s.build_overlay(),
                },
                ServerMessage::SolverStatus {
                    text: s.solver_status_text(),
                },
            ]
        }

        ClientMessage::Solve => {
            let mut s = state.lock().await;
            info!("Solve requested");
            s.re_solve();
            vec![
                ServerMessage::DiffOverlay {
                    overlay: s.build_overlay(),
                },
                ServerMessage::SolverStatus {
                    text: s.solver_status_text(),
                },
            ]
        }

        ClientMessage::TurnPage => {
            // Do page turn + HTML generation while holding the lock,
            // then drop lock before the html_cache await.
            let html_result: Result<String, String> = {
                let mut s = state.lock().await;
                info!("Turn page requested");
                match s.turn_page() {
                    Ok(()) => generate_seqsee_html(
                        &s.page,
                        &s.seqsee_dir,
                        ws_port,
                        s.load_max_total,
                    )
                    .map_err(|e| e.to_string()),
                    Err(e) => Err(format!("Turn page failed: {}", e)),
                }
            };
            match html_result {
                Ok(html) => {
                    *html_cache.write().await = html;
                    vec![ServerMessage::Reload]
                }
                Err(e) => {
                    error!("{}", e);
                    vec![ServerMessage::SolverStatus {
                        text: format!("Error: {}", e),
                    }]
                }
            }
        }

        ClientMessage::LoadPage { path, r, max_total } => {
            let html_result: Result<String, String> = {
                let mut s = state.lock().await;
                info!("Load page: {} r={} max_t={}", path, r, max_total);
                match s.load_page(&path, r, max_total) {
                    Ok(()) => generate_seqsee_html(
                        &s.page,
                        &s.seqsee_dir,
                        ws_port,
                        max_total,
                    )
                    .map_err(|e| e.to_string()),
                    Err(e) => Err(format!("Load page failed: {}", e)),
                }
            };
            match html_result {
                Ok(html) => {
                    *html_cache.write().await = html;
                    vec![ServerMessage::Reload]
                }
                Err(e) => {
                    error!("{}", e);
                    vec![ServerMessage::SolverStatus {
                        text: format!("Error: {}", e),
                    }]
                }
            }
        }
    }
}

// =============================================================================
// HTTP server
// =============================================================================

pub async fn serve_http(listener: TcpListener, html_cache: Arc<RwLock<String>>) {
    loop {
        let Ok((mut stream, addr)) = listener.accept().await else {
            continue;
        };

        let html_cache = html_cache.clone();
        tokio::spawn(async move {
            let mut buf = [0u8; 4096];
            match stream.read(&mut buf).await {
                Ok(0) => return,
                Ok(_) => {}
                Err(e) => {
                    warn!("HTTP read error from {}: {}", addr, e);
                    return;
                }
            }

            let html = html_cache.read().await.clone();

            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                html.len()
            );

            if let Err(e) = stream.write_all(header.as_bytes()).await {
                warn!("HTTP header write error to {}: {}", addr, e);
                return;
            }
            if let Err(e) = stream.write_all(html.as_bytes()).await {
                warn!("HTTP body write error to {}: {}", addr, e);
                return;
            }
            let _ = stream.shutdown().await;
        });
    }
}

/// Start the full server (HTTP + WebSocket), open browser, and run forever.
pub async fn run_server(
    state: ServerState,
    http_port: u16,
    ws_port: u16,
    open_browser: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Generate initial HTML via SeqSee pipeline
    info!("Generating initial chart HTML via SeqSee...");
    let html = generate_seqsee_html(&state.page, &state.seqsee_dir, ws_port, state.load_max_total)?;
    let html_cache = Arc::new(RwLock::new(html));

    let state = Arc::new(Mutex::new(state));
    let (broadcast_tx, _) = broadcast::channel::<String>(256);

    let http_addr: SocketAddr = ([127, 0, 0, 1], http_port).into();
    let http_listener = TcpListener::bind(http_addr).await?;
    info!("HTTP server listening on http://{}", http_addr);

    tokio::spawn(serve_http(http_listener, html_cache.clone()));

    let ws_addr: SocketAddr = ([127, 0, 0, 1], ws_port).into();
    let ws_listener = TcpListener::bind(ws_addr).await?;
    info!("WebSocket server listening on ws://{}", ws_addr);

    let url = format!("http://127.0.0.1:{}", http_port);
    eprintln!("Viewer ready at {}", url);

    if open_browser {
        if let Err(e) = open::that(&url) {
            warn!("Could not open browser: {}", e);
            eprintln!("Open {} in your browser", url);
        }
    }

    loop {
        let (stream, addr) = ws_listener.accept().await?;
        let state = Arc::clone(&state);
        let broadcast_tx = broadcast_tx.clone();
        let html_cache = html_cache.clone();
        tokio::spawn(handle_ws_connection(
            stream,
            addr,
            state,
            broadcast_tx,
            html_cache,
            ws_port,
        ));
    }
}
