/// Inject the WebSocket overlay script into SeqSee-generated HTML.
///
/// Takes the self-contained HTML output from SeqSee's `main.py` and inserts
/// a `<script>` block before `</body>` that adds:
/// - WebSocket client for live differential overlay updates
/// - Floating toolbar (Solve, Turn Page, Add/Toggle differential modes)
/// - Status bar showing solver state
/// - Node click handlers for interactive differential editing
pub fn inject_ws_script(seqsee_html: &str, ws_port: u16) -> String {
    let script = WS_SCRIPT.replace("{{WS_PORT}}", &ws_port.to_string());
    seqsee_html.replace("</body>", &format!("{}\n</body>", script))
}

const WS_SCRIPT: &str = r##"
<script>
(function() {
  const WS_PORT = {{WS_PORT}};
  let ws = null;
  let currentMode = 'inspect';
  let diffClickState = null;
  let currentOverlay = null;

  // =========================================================================
  // UI: toolbar + status bar
  // =========================================================================

  function createUI() {
    // Toolbar
    const toolbar = document.createElement('div');
    toolbar.id = 'ehp-toolbar';
    toolbar.innerHTML = [
      '<button class="ehp-btn active" data-mode="inspect" title="Inspect (I)">Inspect</button>',
      '<button class="ehp-btn" data-mode="add-diff" title="Add differential (D)">d_r</button>',
      '<button class="ehp-btn" data-mode="toggle-diff" title="Toggle (T)">Toggle</button>',
      '<span class="ehp-sep"></span>',
      '<button class="ehp-btn ehp-action" id="btn-solve" title="Solve (S)">Solve</button>',
      '<button class="ehp-btn ehp-action" id="btn-turn" title="Turn page (P)">Turn Page</button>',
    ].join('');
    document.body.appendChild(toolbar);

    // Status bar
    const bar = document.createElement('div');
    bar.id = 'ehp-status';
    bar.textContent = 'Connecting...';
    document.body.appendChild(bar);

    // Styles
    const style = document.createElement('style');
    style.textContent = `
      #ehp-toolbar {
        position: fixed; top: 10px; left: 10px; z-index: 100;
        display: flex; gap: 3px;
        background: rgba(255,255,255,0.94);
        border: 1px solid #bbb; border-radius: 5px;
        padding: 4px 6px; font-family: sans-serif; font-size: 11px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15);
      }
      .ehp-btn {
        background: #f0f0f0; border: 1px solid #bbb; border-radius: 3px;
        padding: 3px 8px; cursor: pointer; font-family: inherit; font-size: 11px;
      }
      .ehp-btn:hover { background: #e0e0e0; }
      .ehp-btn.active { background: #4a90d9; color: white; border-color: #357abd; }
      .ehp-action:hover { background: #c8e6c9; }
      .ehp-sep { border-left: 1px solid #ccc; margin: 0 3px; }
      #ehp-status {
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
        background: rgba(255,255,255,0.94);
        border-top: 1px solid #ccc; padding: 4px 10px;
        font-family: sans-serif; font-size: 11px; color: #333;
      }
      .ehp-selected {
        stroke: #4a90d9 !important; stroke-width: 3px !important;
      }
    `;
    document.head.appendChild(style);

    // Wire up toolbar mode buttons
    document.querySelectorAll('.ehp-btn[data-mode]').forEach(b => {
      b.addEventListener('click', () => setMode(b.dataset.mode));
    });
    document.getElementById('btn-solve').addEventListener('click', () => {
      setStatus('Solving...');
      send({ type: 'Solve' });
    });
    document.getElementById('btn-turn').addEventListener('click', () => {
      setStatus('Turning page...');
      send({ type: 'TurnPage' });
    });
  }

  function setStatus(text) {
    const el = document.getElementById('ehp-status');
    if (el) el.textContent = text;
  }

  function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.ehp-btn[data-mode]').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
    clearDiffClick();
  }

  function clearDiffClick() {
    if (diffClickState) {
      diffClickState.el.classList.remove('ehp-selected');
      diffClickState = null;
    }
  }

  // =========================================================================
  // WebSocket
  // =========================================================================

  function connectWS() {
    ws = new WebSocket('ws://localhost:' + WS_PORT);
    ws.onopen = () => setStatus('Connected');
    ws.onclose = () => {
      setStatus('Disconnected \u2014 reconnecting...');
      setTimeout(connectWS, 2000);
    };
    ws.onerror = () => {};
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'DiffOverlay') {
        applyOverlay(msg.overlay);
      } else if (msg.type === 'SolverStatus') {
        setStatus(msg.text);
      } else if (msg.type === 'Reload') {
        window.location.reload();
      }
    };
  }

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }

  // =========================================================================
  // Differential overlay rendering
  // =========================================================================

  function getNodeCenter(nodeId) {
    const el = document.getElementById(nodeId);
    if (!el) return null;
    if (el.tagName === 'circle') {
      return {
        x: parseFloat(el.getAttribute('cx')),
        y: parseFloat(el.getAttribute('cy'))
      };
    }
    // rect: use center
    const x = parseFloat(el.getAttribute('x')) + parseFloat(el.getAttribute('width')) / 2;
    const y = parseFloat(el.getAttribute('y')) + parseFloat(el.getAttribute('height')) / 2;
    return { x, y };
  }

  function applyOverlay(overlay) {
    currentOverlay = overlay;
    const group = document.getElementById('diff-overlay');
    if (!group) return;

    // Clear old overlay
    group.innerHTML = '';

    // Draw differential edges
    for (const edge of overlay.edges) {
      const src = getNodeCenter(edge.source);
      const tgt = getNodeCenter(edge.target);
      if (!src || !tgt) continue;

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', src.x);
      line.setAttribute('y1', src.y);
      line.setAttribute('x2', tgt.x);
      line.setAttribute('y2', tgt.y);
      line.setAttribute('fill', 'none');

      if (edge.style === 'determined') {
        line.setAttribute('stroke', '#d00000');
        line.setAttribute('stroke-width', '1.5');
      } else {
        line.setAttribute('stroke', '#cc8800');
        line.setAttribute('stroke-width', '1.2');
        line.setAttribute('stroke-dasharray', '5,5');
      }

      group.appendChild(line);
    }

    // Reset all node colors to default, then mark uncertain nodes
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {
      el.style.fill = '';
      el.style.stroke = '';
    });
    for (const nodeId of overlay.uncertain) {
      const el = document.getElementById(nodeId);
      if (el) {
        el.style.fill = '#999999';
        el.style.stroke = '#999999';
      }
    }
  }

  // =========================================================================
  // Node interaction
  // =========================================================================

  function setupNodeInteraction() {
    document.querySelectorAll('#nodes-group circle, #nodes-group rect').forEach(el => {
      el.style.cursor = 'pointer';
      el.addEventListener('click', onNodeClick);
    });
  }

  function onNodeClick(e) {
    const el = e.currentTarget;
    const nodeId = el.id;
    // Parse tridegree from node ID: n{n}_s{s}_f{f}_i{idx}
    const m = nodeId.match(/^n(-?\d+)_s(-?\d+)_f(-?\d+)_i(\d+)$/);
    if (!m) return;
    const n = parseInt(m[1]), s = parseInt(m[2]), f = parseInt(m[3]), idx = parseInt(m[4]);

    if (currentMode === 'add-diff') {
      if (!diffClickState) {
        // First click: select source
        diffClickState = { n, s, f, idx, el };
        el.classList.add('ehp-selected');
        setStatus('Source: (' + n + ',' + s + ',' + f + ')[' + idx + '] \u2014 click target');
      } else {
        // Second click: send AddDifferential
        const src = diffClickState;
        clearDiffClick();
        send({
          type: 'AddDifferential',
          source: [src.n, src.s, src.f], sourceIdx: src.idx,
          target: [n, s, f], targetIdx: idx,
          value: true
        });
      }
    } else if (currentMode === 'toggle-diff') {
      if (!currentOverlay) return;
      // Toggle all edges from this node
      const outEdges = currentOverlay.edges.filter(e => e.source === nodeId);
      for (const edge of outEdges) {
        send({
          type: 'ToggleDifferential',
          source: edge.source_tri, sourceIdx: edge.source_idx,
          target: edge.target_tri, targetIdx: edge.target_idx
        });
      }
    }
  }

  // =========================================================================
  // Keyboard shortcuts
  // =========================================================================

  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const k = e.key.toLowerCase();
    if (k === 'i') setMode('inspect');
    else if (k === 'd') setMode('add-diff');
    else if (k === 't') setMode('toggle-diff');
    else if (k === 's' && !e.ctrlKey && !e.metaKey) {
      setStatus('Solving...');
      send({ type: 'Solve' });
    }
    else if (k === 'p') {
      setStatus('Turning page...');
      send({ type: 'TurnPage' });
    }
    else if (k === 'escape') clearDiffClick();
  });

  // =========================================================================
  // Initialization — runs after SeqSee's DOMContentLoaded handler
  // =========================================================================

  window.addEventListener('DOMContentLoaded', () => {
    createUI();

    // Create overlay group inside content-group, between edges and nodes
    const nodesGroup = document.getElementById('nodes-group');
    if (nodesGroup && nodesGroup.parentNode) {
      const overlay = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      overlay.id = 'diff-overlay';
      nodesGroup.parentNode.insertBefore(overlay, nodesGroup);
    }

    setupNodeInteraction();
    connectWS();
  });
})();
</script>
"##;
