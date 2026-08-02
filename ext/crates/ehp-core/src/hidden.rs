//! Hidden EHP map values + Toda composition propagation (EXPERIMENTAL).
//!
//! On the terminal page, exactness sometimes forces a *hidden value* of an
//! EHP map: a map value landing at strictly higher Adams filtration than the
//! nominal target (e.g. the class at (11,13,3) supports a hidden P hitting
//! (5,17,6); the nominal P target filtration is 5). This module stores
//! user-asserted hidden values and propagates them through Toda's composition
//! relation
//!
//! ```text
//! P(a ∘ E²b) = P(a) ∘ b
//! ```
//!
//! using the page's product table — which IS the composition product:
//! factor1 at (n,s1,f1), factor2 at sphere n+s1, product at (n,s1+s2,f1+f2).
//!
//! Deductions are emitted ONLY when every product involved is nonzero
//! (E²b ≠ 0, a∘E²b ≠ 0, P(a)∘b ≠ 0): Toda's formulas hold in homotopy, and a
//! product that is algebraically zero on the page may still be nonzero in
//! homotopy via an uncomputed hidden extension — so no conclusion is drawn
//! from it.
//!
//! MODULARITY CONTRACT: this module is a pure overlay. It reads `SATPage`
//! (products, E map via `map_matrix_ref`, dimensions) and never writes any
//! solver / constraint / page-turning state; `EHP_HIDDEN` (`hidden_enabled`)
//! is deliberately NOT part of the warm-start cache `config_hash` because it
//! cannot affect the solve. Deleting this module (plus its few REPL call
//! sites) leaves the engine byte-identical.

use hashbrown::HashSet;

use crate::element::Element;
use crate::gf2::*;
use crate::map::MapKind;
use crate::page::SATPage;
use crate::tridegree::Tridegree;

/// `EHP_HIDDEN`: default ON (the subsystem is inert without assertions);
/// `EHP_HIDDEN=0` disables loading, asserting, and chart injection.
pub fn hidden_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("EHP_HIDDEN").map_or(true, |v| v != "0"))
}

/// Where a hidden value came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Asserted by the user (REPL `hidden` command or the persisted CSV).
    Asserted,
    /// Deduced from the hidden value with source `via` by composing with `b`:
    /// P(via ∘ E²b) = P(via) ∘ b.
    Deduced { via: Element, b: Element },
}

/// One hidden map value: `kind(source) = target`, with `target` at the
/// nominal target degree except for a filtration jump `delta() ≥ 1`.
/// `source`/`target` are F2 vectors (sums allowed), not just basis elements.
#[derive(Clone, Debug)]
pub struct HiddenValue {
    pub kind: MapKind,
    pub source: Element,
    pub target: Element,
    pub provenance: Provenance,
}

impl HiddenValue {
    /// Filtration jump above the nominal target degree.
    pub fn delta(&self) -> i32 {
        self.target.degree.f - self.kind.target_degree(self.source.degree).f
    }

    pub fn is_asserted(&self) -> bool {
        matches!(self.provenance, Provenance::Asserted)
    }

    fn key(&self) -> (MapKind, Element, Element) {
        (self.kind, self.source.clone(), self.target.clone())
    }
}

impl std::fmt::Display for HiddenValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({}) = {} (hidden, δ={})",
            self.kind, self.source, self.target, self.delta()
        )?;
        if let Provenance::Deduced { via, b } = &self.provenance {
            write!(f, " [via {}({}), b = {}]", self.kind, via, b)?;
        }
        Ok(())
    }
}

/// A persisted row that failed validation against the current page (basis
/// drift, page-r mismatch, …). Kept in string form so `rebuild` can re-admit
/// it when the page state changes back, and so the CSV never loses rows.
#[derive(Clone, Debug)]
pub struct QuarantinedRow {
    pub r: i32,
    pub kind: MapKind,
    pub source: String,
    pub target: String,
    pub reason: String,
}

/// Validate a candidate hidden value against a page.
///
/// Checks: map domain; target n/s equal to the nominal target (only f may
/// differ); δ ≥ 1 (δ = 0 is an ordinary map value that belongs in the map
/// tables, not here); both vectors nonzero and sized to the current page
/// dimensions (the length check doubles as the basis-drift prune).
pub fn validate(
    page: &SATPage,
    kind: MapKind,
    source: &Element,
    target: &Element,
) -> Result<(), String> {
    if kind == MapKind::Lh0 {
        return Err("hidden values are for the EHP triple (E, H, P), not lh0".into());
    }
    if !kind.domain_check(source.degree) {
        return Err(format!("{} is not defined on {}", kind, source.degree));
    }
    let nominal = kind.target_degree(source.degree);
    if target.degree.n != nominal.n || target.degree.s != nominal.s {
        return Err(format!(
            "target degree {} is not on the {}-target line of {} (nominal target {})",
            target.degree, kind, source.degree, nominal
        ));
    }
    let delta = target.degree.f - nominal.f;
    if delta < 1 {
        return Err(format!(
            "δ = {} at target {}: a hidden value must land at least one filtration \
             above the nominal target {} (δ = 0 is an ordinary map value)",
            delta, target.degree, nominal
        ));
    }
    for (name, e) in [("source", source), ("target", target)] {
        let dim = page.dim_at(e.degree);
        if dim == 0 {
            return Err(format!("{} degree {} has no classes on this page", name, e.degree));
        }
        if e.vec.len() != dim {
            return Err(format!(
                "{} {} has {} coordinates but {} currently has dimension {}",
                name, e, e.vec.len(), e.degree, dim
            ));
        }
        if e.is_zero() {
            return Err(format!("{} element is zero", name));
        }
    }
    Ok(())
}

/// Strictly parse an element string `n_s_f`, `n_s_f_i`, or a `" + "`-joined
/// sum against the page. Unlike `io.rs`'s loader (which silently drops
/// out-of-range indices), any index ≥ the current dimension is an ERROR —
/// that strictness is what makes basis-drift pruning work.
pub fn parse_element_strict(s: &str, page: &SATPage) -> Result<Element, String> {
    let s = s.trim().trim_matches('"');
    let mut degree: Option<Tridegree> = None;
    let mut vec = None;
    for part in s.split(" + ") {
        let bits: Vec<&str> = part.trim().split('_').collect();
        if bits.len() < 3 || bits.len() > 4 {
            return Err(format!("bad element term '{}' (want n_s_f or n_s_f_i)", part));
        }
        let parse =
            |x: &str| x.parse::<i32>().map_err(|_| format!("bad number '{}' in '{}'", x, part));
        let t = Tridegree::new(parse(bits[0])?, parse(bits[1])?, parse(bits[2])?);
        let i: usize = if bits.len() == 4 {
            bits[3].parse().map_err(|_| format!("bad index in '{}'", part))?
        } else {
            0
        };
        let dim = page.dim_at(t);
        if i >= dim {
            return Err(format!(
                "'{}': index {} out of range ({} has dimension {})",
                part, i, t, dim
            ));
        }
        match degree {
            None => {
                degree = Some(t);
                vec = Some(vec_zero(dim));
            }
            Some(d) if d != t => {
                return Err(format!("summands at different degrees: {} vs {}", d, t))
            }
            Some(_) => {}
        }
        vec_flip(vec.as_mut().unwrap(), i);
    }
    match (degree, vec) {
        (Some(d), Some(v)) if !v.is_zero() => Ok(Element::new(d, v)),
        (Some(_), Some(_)) => Err(format!("'{}' cancels to zero", s)),
        _ => Err("empty element string".into()),
    }
}

fn parse_kind(s: &str) -> Result<MapKind, String> {
    match s {
        "E" | "e" => Ok(MapKind::E),
        "H" | "h" => Ok(MapKind::H),
        "P" | "p" => Ok(MapKind::P),
        other => Err(format!("unknown map kind '{}' (want E, H, or P)", other)),
    }
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

/// A pluggable propagation rule. `deduce` receives one hidden value and emits
/// newly implied ones (only for products that are all nonzero — see module
/// docs); non-fatal notes (e.g. enumeration caps) go through `warn`.
pub trait HiddenRule {
    fn name(&self) -> &'static str;
    fn deduce(
        &self,
        page: &SATPage,
        hv: &HiddenValue,
        emit: &mut dyn FnMut(HiddenValue),
        warn: &mut dyn FnMut(String),
    );
}

/// Above this dimension, `b` candidates are enumerated as basis vectors only
/// instead of all 2^dim − 1 nonzero vectors (with a warning). Product-degree
/// dims in this dataset are tiny, so the cap is effectively never hit.
const MAX_ENUM_DIM: usize = 10;

/// Toda's composition relation `P(a ∘ E²b) = P(a) ∘ b`.
///
/// Given a hidden `P(a) = c` with `a` at `(n, s_a, f_a)`, candidates `b` live
/// at sphere `n + s_a − 2` — which equals `c.n + c.s`, so the same degrees
/// are the legal right composition factors for BOTH products. Enumeration
/// iterates `page.dimension` at that sphere (a product-block scan would be
/// page-wide and would miss the stable-range E identity defaults that only
/// `map_matrix_ref` supplies).
pub struct PCompositionRule;

impl HiddenRule for PCompositionRule {
    fn name(&self) -> &'static str {
        "P(a∘E²b) = P(a)∘b"
    }

    fn deduce(
        &self,
        page: &SATPage,
        hv: &HiddenValue,
        emit: &mut dyn FnMut(HiddenValue),
        warn: &mut dyn FnMut(String),
    ) {
        if hv.kind != MapKind::P {
            return;
        }
        let deg_a = hv.source.degree;
        let deg_c = hv.target.degree;
        let b_sphere = deg_a.n + deg_a.s - 2;
        debug_assert_eq!(b_sphere, deg_c.n + deg_c.s);

        // Deterministic candidate order regardless of hash-map iteration.
        let mut candidates: Vec<Tridegree> = page
            .dimension
            .iter()
            .filter(|(t, &d)| t.n == b_sphere && d > 0)
            .map(|(&t, _)| t)
            .collect();
        candidates.sort_by_key(|t| (t.s, t.f));

        for deg_b in candidates {
            // P(a)∘b lookup is independent of the suspensions — check first.
            if !page.products.has_block(deg_c, deg_b) {
                continue;
            }
            let dim_b = page.dim_at(deg_b);
            let deg_eb = MapKind::E.target_degree(deg_b);
            let deg_e2b = MapKind::E.target_degree(deg_eb);
            debug_assert_eq!(deg_e2b.n, deg_a.n + deg_a.s);
            if !page.products.has_block(deg_a, deg_e2b) {
                continue;
            }
            let x_deg = Tridegree::new(deg_a.n, deg_a.s + deg_b.s, deg_a.f + deg_b.f);
            let y_deg = Tridegree::new(deg_c.n, deg_c.s + deg_b.s, deg_c.f + deg_b.f);
            let (x_dim, y_dim) = (page.dim_at(x_deg), page.dim_at(y_deg));
            if x_dim == 0 || y_dim == 0 {
                continue;
            }

            let masks: u32 = if dim_b <= MAX_ENUM_DIM {
                (1u32 << dim_b) - 1
            } else {
                warn(format!(
                    "hidden: {} has dimension {} > {}; enumerating basis b's only",
                    deg_b, dim_b, MAX_ENUM_DIM
                ));
                0
            };
            let b_vecs: Vec<fp::vector::FpVector> = if masks > 0 {
                (1..=masks)
                    .map(|m| {
                        let mut v = vec_zero(dim_b);
                        for i in 0..dim_b {
                            if m & (1 << i) != 0 {
                                vec_flip(&mut v, i);
                            }
                        }
                        v
                    })
                    .collect()
            } else {
                (0..dim_b).map(|i| vec_basis(dim_b, i)).collect()
            };

            for b in b_vecs {
                // E²b via map_matrix_ref so stable-range identities apply.
                let eb = page.map_matrix_ref(MapKind::E, deg_b).apply_vec(&b);
                if eb.is_zero() {
                    continue;
                }
                let e2b = page.map_matrix_ref(MapKind::E, deg_eb).apply_vec(&eb);
                if e2b.is_zero() {
                    continue;
                }
                let x = page.products.multiply(deg_a, &hv.source.vec, deg_e2b, &e2b, x_dim);
                if x.is_zero() {
                    continue;
                }
                let y = page.products.multiply(deg_c, &hv.target.vec, deg_b, &b, y_dim);
                if y.is_zero() {
                    continue;
                }
                let deduced = HiddenValue {
                    kind: MapKind::P,
                    source: Element::new(x_deg, x),
                    target: Element::new(y_deg, y),
                    provenance: Provenance::Deduced {
                        via: hv.source.clone(),
                        b: Element::new(deg_b, b),
                    },
                };
                debug_assert_eq!(deduced.delta(), hv.delta());
                emit(deduced);
            }
        }
    }
}

fn default_rules() -> Vec<Box<dyn HiddenRule>> {
    vec![Box::new(PCompositionRule)]
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// The session's hidden values: user assertions plus everything the rules
/// deduce from them, deduped, with a quarantine for persisted rows that do
/// not validate against the current page state.
#[derive(Default)]
pub struct HiddenStore {
    values: Vec<HiddenValue>,
    seen: HashSet<(MapKind, Element, Element)>,
    pub quarantined: Vec<QuarantinedRow>,
}

impl HiddenStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty() && self.quarantined.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &HiddenValue> {
        self.values.iter()
    }

    pub fn asserted(&self) -> impl Iterator<Item = &HiddenValue> {
        self.values.iter().filter(|v| v.is_asserted())
    }

    pub fn deduced(&self) -> impl Iterator<Item = &HiddenValue> {
        self.values.iter().filter(|v| !v.is_asserted())
    }

    /// Validate + dedupe + record a user assertion. `Ok(true)` = newly added,
    /// `Ok(false)` = already present.
    pub fn assert_value(
        &mut self,
        page: &SATPage,
        kind: MapKind,
        source: Element,
        target: Element,
    ) -> Result<bool, String> {
        validate(page, kind, &source, &target)?;
        let hv = HiddenValue { kind, source, target, provenance: Provenance::Asserted };
        if !self.seen.insert(hv.key()) {
            return Ok(false);
        }
        self.values.push(hv);
        Ok(true)
    }

    /// Run every rule to a fixpoint over the current values (worklist with
    /// dedup; terminates because values live in the finite set of degree ×
    /// F2-vector pairs of the page). Returns the newly deduced values (in
    /// deduction order) and any rule warnings.
    pub fn propagate(&mut self, page: &SATPage) -> (Vec<HiddenValue>, Vec<String>) {
        let rules = default_rules();
        let mut warnings = Vec::new();
        let mut new_values = Vec::new();
        let mut next = 0usize;
        while next < self.values.len() {
            let hv = self.values[next].clone();
            next += 1;
            let mut emitted = Vec::new();
            for rule in &rules {
                rule.deduce(page, &hv, &mut |d| emitted.push(d), &mut |w| {
                    if !warnings.contains(&w) {
                        warnings.push(w);
                    }
                });
            }
            for d in emitted {
                debug_assert!(validate(page, d.kind, &d.source, &d.target).is_ok());
                if self.seen.insert(d.key()) {
                    new_values.push(d.clone());
                    self.values.push(d);
                }
            }
        }
        (new_values, warnings)
    }

    /// Drop all deduced values, re-validate every asserted value AND every
    /// quarantined row against the (possibly rebuilt) page — quarantining the
    /// failures and re-admitting recovered rows — then re-propagate. This is
    /// the recovery primitive after `hidden undo`/`remove` and after any
    /// differential mutation changes the page basis. Returns warnings (one
    /// per newly quarantined row, plus rule warnings).
    pub fn rebuild(&mut self, page: &SATPage, current_r: i32) -> Vec<String> {
        let mut rows: Vec<QuarantinedRow> = self
            .asserted()
            .map(|v| QuarantinedRow {
                r: current_r,
                kind: v.kind,
                source: v.source.to_string(),
                target: v.target.to_string(),
                reason: String::new(),
            })
            .collect();
        rows.append(&mut self.quarantined);
        self.values.clear();
        self.seen.clear();

        let mut warnings = Vec::new();
        for row in rows {
            if let Some(w) = self.admit_row(page, current_r, row) {
                warnings.push(w);
            }
        }
        let (_, mut rule_warnings) = self.propagate(page);
        warnings.append(&mut rule_warnings);
        warnings
    }

    /// Try to admit one persisted/re-validated row; quarantine on failure and
    /// return the warning.
    fn admit_row(
        &mut self,
        page: &SATPage,
        current_r: i32,
        mut row: QuarantinedRow,
    ) -> Option<String> {
        let attempt = (|| -> Result<(), String> {
            if row.r != current_r {
                return Err(format!(
                    "recorded on E{} but the terminal page is E{}",
                    row.r, current_r
                ));
            }
            let source = parse_element_strict(&row.source, page)?;
            let target = parse_element_strict(&row.target, page)?;
            self.assert_value(page, row.kind, source, target)?;
            Ok(())
        })();
        match attempt {
            Ok(()) => None,
            Err(reason) => {
                let warning = format!(
                    "hidden {}({}) = {} quarantined: {}",
                    row.kind, row.source, row.target, reason
                );
                row.reason = reason;
                self.quarantined.push(row);
                Some(warning)
            }
        }
    }

    /// Remove one asserted value (matched by kind + exact source/target
    /// strings). Caller should follow with `rebuild` to drop its deductions.
    pub fn remove_asserted(&mut self, kind: MapKind, source: &str, target: &str) -> bool {
        let pos = self.values.iter().position(|v| {
            v.is_asserted()
                && v.kind == kind
                && v.source.to_string() == source
                && v.target.to_string() == target
        });
        match pos {
            Some(i) => {
                let v = self.values.remove(i);
                self.seen.remove(&v.key());
                true
            }
            None => false,
        }
    }

    /// Remove the most recently asserted value (for `hidden undo`). Caller
    /// should follow with `rebuild`.
    pub fn pop_asserted(&mut self) -> Option<HiddenValue> {
        let i = self.values.iter().rposition(|v| v.is_asserted())?;
        let v = self.values.remove(i);
        self.seen.remove(&v.key());
        Some(v)
    }

    // -----------------------------------------------------------------------
    // CSV persistence
    // -----------------------------------------------------------------------

    /// Write the ASSERTED values (deduced ones are recomputed at load — that
    /// makes the file robust to basis drift) plus any quarantined rows (so a
    /// temporarily-invalid row is never silently lost). Whole-file rewrite:
    /// the file is tiny and this keeps it canonical.
    pub fn save_csv(&self, path: &std::path::Path, current_r: i32) -> std::io::Result<()> {
        use std::io::Write;
        let mut out = String::from("\"r\",\"kind\",\"source\",\"target\"\n");
        for v in self.asserted() {
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",\"{}\"\n",
                current_r, v.kind, v.source, v.target
            ));
        }
        for q in &self.quarantined {
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",\"{}\"\n",
                q.r, q.kind, q.source, q.target
            ));
        }
        let mut f = std::fs::File::create(path)?;
        f.write_all(out.as_bytes())
    }

    /// Load assertions from the CSV (if present), validating each row against
    /// the page: invalid rows are quarantined with a per-row warning, never a
    /// hard failure. Does NOT propagate — callers decide when.
    pub fn load_csv(
        &mut self,
        path: &std::path::Path,
        page: &SATPage,
        current_r: i32,
    ) -> Vec<String> {
        let mut warnings = Vec::new();
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => return warnings,
        };
        for line in text.lines().skip(1) {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<String> = split_csv_row(line);
            if fields.len() < 4 {
                warnings.push(format!("hidden CSV row skipped (want 4 fields): {}", line));
                continue;
            }
            let r: i32 = match fields[0].parse() {
                Ok(v) => v,
                Err(_) => {
                    warnings.push(format!("hidden CSV row skipped (bad r): {}", line));
                    continue;
                }
            };
            let kind = match parse_kind(&fields[1]) {
                Ok(k) => k,
                Err(e) => {
                    warnings.push(format!("hidden CSV row skipped: {}", e));
                    continue;
                }
            };
            let row = QuarantinedRow {
                r,
                kind,
                source: fields[2].clone(),
                target: fields[3].clone(),
                reason: String::new(),
            };
            if let Some(w) = self.admit_row(page, current_r, row) {
                warnings.push(w);
            }
        }
        warnings
    }
}

/// Split a quoted CSV row (`"a","b","c"` — the format we write; also accepts
/// unquoted fields). Element strings never contain commas or quotes.
fn split_csv_row(line: &str) -> Vec<String> {
    line.split(',')
        .map(|f| f.trim().trim_matches('"').to_string())
        .collect()
}

/// Path of the persisted hidden-values CSV: `<data_dir>/hidden_EHP.csv`.
/// NOTE: must never live under `EHP_OUTSIDE_DIFFS` — that directory is hashed
/// into the warm-start cache key, and this file deliberately is not (it
/// cannot affect the solve).
pub fn hidden_csv_path(data_path: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(data_path);
    let dir = if p.is_dir() {
        p.to_path_buf()
    } else {
        p.parent().map(|d| d.to_path_buf()).unwrap_or_else(|| p.to_path_buf())
    };
    dir.join("hidden_EHP.csv")
}

/// Parse a REPL `hidden` assert/remove argument list
/// `<E|H|P> <n> <s> <f> <idx> <tn> <ts> <tf> <tidx>` (idx fields accept
/// `+`-joined sums like `0+2`) into (kind, source, target). Pure parsing +
/// validation; does not mutate anything.
pub fn parse_hidden_args(
    page: &SATPage,
    args: &[&str],
) -> Result<(MapKind, Element, Element), String> {
    if args.len() != 9 {
        return Err(
            "usage: hidden <E|H|P> <n> <s> <f> <idx> <tn> <ts> <tf> <tidx> (idx may be 0+2)"
                .into(),
        );
    }
    let kind = parse_kind(args[0])?;
    let num = |x: &str| x.parse::<i32>().map_err(|_| format!("bad number '{}'", x));
    let src_deg = Tridegree::new(num(args[1])?, num(args[2])?, num(args[3])?);
    let tgt_deg = Tridegree::new(num(args[5])?, num(args[6])?, num(args[7])?);
    let elem = |deg: Tridegree, idxs: &str| -> Result<Element, String> {
        let dim = page.dim_at(deg);
        if dim == 0 {
            return Err(format!("{} has no classes on this page", deg));
        }
        let mut v = vec_zero(dim);
        for part in idxs.split('+') {
            let i: usize = part
                .trim()
                .parse()
                .map_err(|_| format!("bad index '{}' (want e.g. 0 or 0+2)", idxs))?;
            if i >= dim {
                return Err(format!("index {} out of range ({} has dimension {})", i, deg, dim));
            }
            vec_flip(&mut v, i);
        }
        if v.is_zero() {
            return Err(format!("indices '{}' cancel to zero", idxs));
        }
        Ok(Element::new(deg, v))
    };
    let source = elem(src_deg, args[4])?;
    let target = elem(tgt_deg, args[8])?;
    validate(page, kind, &source, &target)?;
    Ok((kind, source, target))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::products::ProductKey;

    fn deg(n: i32, s: i32, f: i32) -> Tridegree {
        Tridegree::new(n, s, f)
    }

    /// The motivating example's degrees: a = (11,13,3), hidden target
    /// c = (5,17,6) (nominal (5,17,5)), b's at sphere 22.
    fn example_page() -> SATPage {
        let mut page = SATPage::new(5);
        for (t, d) in [(deg(11, 13, 3), 1), (deg(5, 17, 6), 1), (deg(5, 17, 5), 1)] {
            page.dimension.insert(t, d);
        }
        page
    }

    fn basis(t: Tridegree, dim: usize, i: usize) -> Element {
        Element::basis(t, dim, i)
    }

    /// Test 1: δ/validation on the motivating example.
    #[test]
    fn validation_and_delta() {
        let page = example_page();
        let a = basis(deg(11, 13, 3), 1, 0);
        let c = basis(deg(5, 17, 6), 1, 0);

        // The motivating example is accepted with δ = 1.
        assert!(validate(&page, MapKind::P, &a, &c).is_ok());
        let hv = HiddenValue {
            kind: MapKind::P,
            source: a.clone(),
            target: c.clone(),
            provenance: Provenance::Asserted,
        };
        assert_eq!(hv.delta(), 1);

        // δ = 0 (the nominal target) is an ordinary map value: rejected.
        let nominal = basis(deg(5, 17, 5), 1, 0);
        assert!(validate(&page, MapKind::P, &a, &nominal).unwrap_err().contains("δ = 0"));

        // Even sphere: P undefined.
        let mut page2 = example_page();
        page2.dimension.insert(deg(10, 13, 3), 1);
        let a_even = basis(deg(10, 13, 3), 1, 0);
        assert!(validate(&page2, MapKind::P, &a_even, &c).is_err());

        // Wrong target stem.
        let mut page3 = example_page();
        page3.dimension.insert(deg(5, 16, 6), 1);
        let wrong_s = basis(deg(5, 16, 6), 1, 0);
        assert!(validate(&page3, MapKind::P, &a, &wrong_s).is_err());

        // Dimension mismatch (basis drift): vector length ≠ current dim.
        let stale = Element::basis(deg(5, 17, 6), 2, 0);
        assert!(validate(&page, MapKind::P, &a, &stale).is_err());

        // Zero source.
        let zero = Element::zero(deg(11, 13, 3), 1);
        assert!(validate(&page, MapKind::P, &zero, &c).is_err());
    }

    /// Build the page for the deduction tests: b at (22,1,1), E chain
    /// 22 → 23 (stable-range identity default) → 24 (stored matrix), product
    /// blocks (a, E²b) and (c, b), result degrees x = (11,14,4), y = (5,18,7).
    fn deduction_page() -> SATPage {
        let mut page = example_page();
        for (t, d) in [
            (deg(22, 1, 1), 1),
            (deg(23, 1, 1), 1),
            (deg(24, 1, 1), 1),
            (deg(11, 14, 4), 1),
            (deg(5, 18, 7), 1),
        ] {
            page.dimension.insert(t, d);
        }
        // Stored E matrix on the second step (23 → 24); first step exercises
        // the stable-range identity default (n = 22 > s + 1 = 2, equal dims).
        page.maps
            .get_mut(&MapKind::E)
            .unwrap()
            .set_matrix(deg(23, 1, 1), mat_identity(1));
        // a ∘ E²b and c ∘ b both hit the generator.
        page.products.insert(
            ProductKey::new(deg(11, 13, 3), 0, deg(24, 1, 1), 0),
            vec_basis(1, 0),
            1,
            1,
        );
        page.products.insert(
            ProductKey::new(deg(5, 17, 6), 0, deg(22, 1, 1), 0),
            vec_basis(1, 0),
            1,
            1,
        );
        page
    }

    fn assert_example(store: &mut HiddenStore, page: &SATPage) {
        let a = basis(deg(11, 13, 3), 1, 0);
        let c = basis(deg(5, 17, 6), 1, 0);
        assert!(store.assert_value(page, MapKind::P, a, c).unwrap());
    }

    /// Test 2: the full deduction chain, exercising stored + identity-default
    /// E matrices.
    #[test]
    fn p_composition_deduces() {
        let page = deduction_page();
        let mut store = HiddenStore::new();
        assert_example(&mut store, &page);
        let (new_values, warnings) = store.propagate(&page);
        assert!(warnings.is_empty());
        assert_eq!(new_values.len(), 1);
        let d = &new_values[0];
        assert_eq!(d.kind, MapKind::P);
        assert_eq!(d.source, basis(deg(11, 14, 4), 1, 0));
        assert_eq!(d.target, basis(deg(5, 18, 7), 1, 0));
        assert_eq!(d.delta(), 1);
        match &d.provenance {
            Provenance::Deduced { via, b } => {
                assert_eq!(*via, basis(deg(11, 13, 3), 1, 0));
                assert_eq!(*b, basis(deg(22, 1, 1), 1, 0));
            }
            p => panic!("wrong provenance {:?}", p),
        }
    }

    /// Test 3: zeroing any one of E²b / a∘E²b / P(a)∘b suppresses the
    /// deduction (nonzero-only policy).
    #[test]
    fn nonzero_only_gating() {
        // (a) E²b = 0: drop the stored 23→24 E matrix and give the degrees
        // different dims so the stable default is Zero, not Identity.
        let mut page = deduction_page();
        page.maps.get_mut(&MapKind::E).unwrap().remove_matrix(deg(23, 1, 1));
        page.dimension.insert(deg(24, 1, 1), 0);
        let mut store = HiddenStore::new();
        assert_example(&mut store, &page);
        assert!(store.propagate(&page).0.is_empty());

        // (b) a ∘ E²b = 0.
        let mut page = deduction_page();
        page.products.remove_block(deg(11, 13, 3), deg(24, 1, 1));
        let mut store = HiddenStore::new();
        assert_example(&mut store, &page);
        assert!(store.propagate(&page).0.is_empty());

        // (c) P(a) ∘ b = 0.
        let mut page = deduction_page();
        page.products.remove_block(deg(5, 17, 6), deg(22, 1, 1));
        let mut store = HiddenStore::new();
        assert_example(&mut store, &page);
        assert!(store.propagate(&page).0.is_empty());
    }

    /// Test 4: a deduction found only by full-vector b enumeration —
    /// b0 has x ≠ 0 but y = 0, b1 has x = 0 but y ≠ 0; only b0 + b1 passes
    /// the all-nonzero gate.
    #[test]
    fn sum_b_enumeration() {
        let mut page = example_page();
        for (t, d) in [
            (deg(22, 1, 1), 2),
            (deg(23, 1, 1), 2),
            (deg(24, 1, 1), 2),
            (deg(11, 14, 4), 1),
            (deg(5, 18, 7), 1),
        ] {
            page.dimension.insert(t, d);
        }
        // Both E steps are stable-range identity defaults (equal dims).
        // a ∘ E²b0 = generator, a ∘ E²b1 = 0.
        page.products.insert(
            ProductKey::new(deg(11, 13, 3), 0, deg(24, 1, 1), 0),
            vec_basis(1, 0),
            1,
            2,
        );
        page.products.insert(
            ProductKey::new(deg(11, 13, 3), 0, deg(24, 1, 1), 1),
            vec_zero(1),
            1,
            2,
        );
        // c ∘ b0 = 0, c ∘ b1 = generator.
        page.products.insert(
            ProductKey::new(deg(5, 17, 6), 0, deg(22, 1, 1), 0),
            vec_zero(1),
            1,
            2,
        );
        page.products.insert(
            ProductKey::new(deg(5, 17, 6), 0, deg(22, 1, 1), 1),
            vec_basis(1, 0),
            1,
            2,
        );

        let mut store = HiddenStore::new();
        assert_example(&mut store, &page);
        let (new_values, _) = store.propagate(&page);
        assert_eq!(new_values.len(), 1);
        match &new_values[0].provenance {
            Provenance::Deduced { b, .. } => {
                // b = b0 + b1 — the sum neither basis vector could supply.
                assert_eq!(b.vec, vec_from_bits(&[1, 1]));
            }
            p => panic!("wrong provenance {:?}", p),
        }
    }

    /// Test 5: fixpoint terminates and dedups — composing with the
    /// fundamental class reproduces the asserted value itself.
    #[test]
    fn fixpoint_dedup_terminates() {
        let mut page = example_page();
        // ι at (22,0,0); suspensions at 23, 24 all dim 1 (stable identities).
        for t in [deg(22, 0, 0), deg(23, 0, 0), deg(24, 0, 0)] {
            page.dimension.insert(t, 1);
        }
        // a ∘ E²ι = a, c ∘ ι = c: x lands back at deg_a, y at deg_c.
        page.products.insert(
            ProductKey::new(deg(11, 13, 3), 0, deg(24, 0, 0), 0),
            vec_basis(1, 0),
            1,
            1,
        );
        page.products.insert(
            ProductKey::new(deg(5, 17, 6), 0, deg(22, 0, 0), 0),
            vec_basis(1, 0),
            1,
            1,
        );
        let mut store = HiddenStore::new();
        assert_example(&mut store, &page);
        let (new_values, _) = store.propagate(&page);
        assert!(new_values.is_empty(), "self-reproduction must dedup");
        // Idempotent.
        assert!(store.propagate(&page).0.is_empty());
        assert_eq!(store.iter().count(), 1);
    }

    /// Test 6: CSV roundtrip, and pruning (quarantine) on basis drift.
    #[test]
    fn csv_roundtrip_and_prune() {
        let mut page = deduction_page();
        // A dim-2 degree so a sum element and an indexed element appear.
        page.dimension.insert(deg(11, 13, 3), 2);
        page.dimension.insert(deg(5, 17, 6), 2);
        let a_sum = Element::new(deg(11, 13, 3), vec_from_bits(&[1, 1]));
        let c0 = basis(deg(5, 17, 6), 2, 0);
        let mut store = HiddenStore::new();
        assert!(store.assert_value(&page, MapKind::P, a_sum.clone(), c0.clone()).unwrap());

        let path = std::env::temp_dir().join(format!("hidden_test_{}.csv", std::process::id()));
        store.save_csv(&path, 5).unwrap();

        // Reload against the same page: identical asserted set, no warnings.
        let mut reloaded = HiddenStore::new();
        let warnings = reloaded.load_csv(&path, &page, 5);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(reloaded.asserted().count(), 1);
        let v = reloaded.asserted().next().unwrap();
        assert_eq!(v.source, a_sum);
        assert_eq!(v.target, c0);

        // Reload against a page where the source degree shrank to dim 1: the
        // sum's index 1 is out of range → quarantined with a warning.
        let mut small = deduction_page();
        small.dimension.insert(deg(5, 17, 6), 2);
        let mut pruned = HiddenStore::new();
        let warnings = pruned.load_csv(&path, &small, 5);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("quarantined"));
        assert_eq!(pruned.asserted().count(), 0);
        assert_eq!(pruned.quarantined.len(), 1);

        // Quarantined rows survive a save (never silently lost) and are
        // re-admitted by rebuild once the page recovers.
        pruned.save_csv(&path, 5).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.lines().count() == 2, "{text}");
        let warnings = pruned.rebuild(&page, 5);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(pruned.asserted().count(), 1);
        assert!(pruned.quarantined.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    /// Test 7: E/H hidden values are stored and displayed but propagate
    /// nothing (only the P rule exists).
    #[test]
    fn e_h_values_are_inert() {
        let mut page = example_page();
        page.dimension.insert(deg(5, 10, 2), 1);
        page.dimension.insert(deg(6, 10, 3), 1);
        let src = basis(deg(5, 10, 2), 1, 0);
        let tgt = basis(deg(6, 10, 3), 1, 0);
        let mut store = HiddenStore::new();
        assert!(store.assert_value(&page, MapKind::E, src, tgt).unwrap());
        let (new_values, warnings) = store.propagate(&page);
        assert!(new_values.is_empty());
        assert!(warnings.is_empty());
        assert_eq!(store.iter().count(), 1);
    }

    /// `hidden` REPL argument parsing.
    #[test]
    fn parse_repl_args() {
        let page = example_page();
        let (kind, source, target) = parse_hidden_args(
            &page,
            &["P", "11", "13", "3", "0", "5", "17", "6", "0"],
        )
        .unwrap();
        assert_eq!(kind, MapKind::P);
        assert_eq!(source, basis(deg(11, 13, 3), 1, 0));
        assert_eq!(target, basis(deg(5, 17, 6), 1, 0));

        // δ = 0 rejected end-to-end.
        assert!(parse_hidden_args(&page, &["P", "11", "13", "3", "0", "5", "17", "5", "0"])
            .is_err());

        // Sum indices.
        let mut page2 = example_page();
        page2.dimension.insert(deg(5, 17, 6), 3);
        let (_, _, target) = parse_hidden_args(
            &page2,
            &["P", "11", "13", "3", "0", "5", "17", "6", "0+2"],
        )
        .unwrap();
        assert_eq!(target.vec, vec_from_bits(&[1, 0, 1]));
    }

    /// Strict element parsing: out-of-range indices are errors, not dropped.
    #[test]
    fn strict_parse() {
        let page = example_page();
        assert!(parse_element_strict("11_13_3", &page).is_ok());
        assert!(parse_element_strict("11_13_3_0", &page).is_ok());
        assert!(parse_element_strict("11_13_3_1", &page).is_err()); // dim 1
        assert!(parse_element_strict("11_13_3_0 + 11_13_3_0", &page).is_err()); // cancels
        assert!(parse_element_strict("11_13_3_0 + 5_17_6_0", &page).is_err()); // mixed degree
    }
}
