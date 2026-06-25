use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use serde::Deserialize;
use std::path::Path;

// --- Internal representation ---

#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub prime: u32,
    pub generators: Vec<Generator>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone)]
pub struct Generator {
    pub name: String,
    pub degree: i32,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub op_name: String,
    pub op_degree: u32,
    pub from_idx: usize,
    pub to_idx: usize,
}

// --- sseq format (primary: what the actual module files use) ---

#[derive(Deserialize)]
struct SseqModuleDef {
    #[allow(dead_code)]
    p: u32,
    #[allow(dead_code)]
    r#type: Option<String>,
    name: Option<String>,
    gens: IndexMap<String, i32>,
    #[serde(default)]
    actions: Vec<SseqAction>,
}

/// Actions in sseq format are strings like "Sq1 x0 = x1"
#[derive(Deserialize)]
#[serde(untagged)]
enum SseqAction {
    StringForm(String),
    ObjectForm(ActionDef),
}

// --- Native spec format (fallback) ---

#[derive(Deserialize)]
struct NativeModuleDef {
    name: String,
    prime: u32,
    gens: IndexMap<String, i32>,
    actions: Vec<ActionDef>,
}

#[derive(Deserialize)]
struct ActionDef {
    op: String,
    from: String,
    to: String,
}

/// Parse the degree from an operation name like "Sq4" -> 4
pub fn op_degree(op: &str) -> Result<u32> {
    let digits = op
        .strip_prefix("Sq")
        .with_context(|| format!("operation '{op}' does not start with 'Sq'"))?;
    digits
        .parse::<u32>()
        .with_context(|| format!("cannot parse degree from '{op}'"))
}

/// Parsed action: one operation applied to a source yielding one or more targets.
struct ParsedAction {
    op: String,
    from: String,
    targets: Vec<String>, // one per summand on RHS; empty if "= 0"
}

/// Parse an sseq-format action string like "Sq1 x0 = x1" or "Sq2 x0 = x2 + x3"
/// or "Sq1 x0 = 0".
fn parse_action_string(s: &str) -> Result<ParsedAction> {
    // Split on '=' first
    let eq_parts: Vec<&str> = s.splitn(2, '=').collect();
    if eq_parts.len() != 2 {
        bail!("invalid action format: '{s}' (no '=' found)");
    }

    let lhs = eq_parts[0].trim();
    let rhs = eq_parts[1].trim();

    // LHS: "Sq1 x0" — split by whitespace, first token is op, second is source
    let lhs_tokens: Vec<&str> = lhs.split_whitespace().collect();
    if lhs_tokens.len() != 2 {
        bail!(
            "invalid action LHS: '{lhs}' (expected 'SqN source', got {} tokens)",
            lhs_tokens.len()
        );
    }
    let op = lhs_tokens[0].to_string();
    let from = lhs_tokens[1].to_string();

    // RHS: "0", "x2", or "x2 + x3"
    if rhs == "0" {
        return Ok(ParsedAction {
            op,
            from,
            targets: vec![],
        });
    }

    // Split on '+' for sums
    let targets: Vec<String> = rhs
        .split('+')
        .map(|t| {
            // Handle possible scalar coefficients (odd-prime): take last token
            let tokens: Vec<&str> = t.split_whitespace().collect();
            tokens.last().unwrap_or(&"").to_string()
        })
        .filter(|t| !t.is_empty())
        .collect();

    if targets.is_empty() {
        bail!("invalid action RHS: '{rhs}' (no targets found)");
    }

    Ok(ParsedAction { op, from, targets })
}

/// Load a module from a JSON file path.
pub fn load_module(path: &Path) -> Result<Module> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unnamed".to_string());
    load_module_from_str(&content, &name)
}

/// Load a module from a JSON string (tries sseq format first, then native).
pub fn load_module_from_str(json: &str, name: &str) -> Result<Module> {
    // Try sseq format first
    if let Ok(sseq) = serde_json::from_str::<SseqModuleDef>(json) {
        return build_from_sseq(sseq, name);
    }
    // Try native spec format
    if let Ok(native) = serde_json::from_str::<NativeModuleDef>(json) {
        return build_from_native(native);
    }
    bail!("could not parse module JSON as either sseq or native format")
}

fn build_from_sseq(def: SseqModuleDef, name: &str) -> Result<Module> {
    let generators: Vec<Generator> = def
        .gens
        .iter()
        .enumerate()
        .map(|(i, (name, &deg))| Generator {
            name: name.clone(),
            degree: deg,
            index: i,
        })
        .collect();

    let gen_index: IndexMap<&str, usize> = generators
        .iter()
        .map(|g| (g.name.as_str(), g.index))
        .collect();

    let mut edges = Vec::new();
    for action in &def.actions {
        match action {
            SseqAction::StringForm(s) => {
                let parsed = match parse_action_string(s) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("Warning: skipping action '{s}': {e}");
                        continue;
                    }
                };

                let deg = match op_degree(&parsed.op) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Warning: skipping action '{s}': {e}");
                        continue;
                    }
                };

                let from_idx = match gen_index.get(parsed.from.as_str()) {
                    Some(&idx) => idx,
                    None => {
                        eprintln!(
                            "Warning: unknown source generator '{}' in action '{s}', skipping",
                            parsed.from
                        );
                        continue;
                    }
                };

                // Each target produces a separate edge
                for target in &parsed.targets {
                    let to_idx = match gen_index.get(target.as_str()) {
                        Some(&idx) => idx,
                        None => {
                            eprintln!(
                                "Warning: unknown target generator '{target}' in action '{s}', skipping"
                            );
                            continue;
                        }
                    };

                    let expected = generators[to_idx].degree - generators[from_idx].degree;
                    if deg as i32 != expected {
                        eprintln!(
                            "Warning: degree mismatch for {} {} = {}: op degree {} != {} (= {} - {}), skipping",
                            parsed.op, parsed.from, target, deg, expected,
                            generators[to_idx].degree, generators[from_idx].degree
                        );
                        continue;
                    }

                    edges.push(Edge {
                        op_name: parsed.op.clone(),
                        op_degree: deg,
                        from_idx,
                        to_idx,
                    });
                }
            }
            SseqAction::ObjectForm(a) => {
                let from_idx = match gen_index.get(a.from.as_str()) {
                    Some(&idx) => idx,
                    None => {
                        eprintln!(
                            "Warning: unknown generator '{}' in action, skipping",
                            a.from
                        );
                        continue;
                    }
                };
                let to_idx = match gen_index.get(a.to.as_str()) {
                    Some(&idx) => idx,
                    None => {
                        eprintln!(
                            "Warning: unknown generator '{}' in action, skipping",
                            a.to
                        );
                        continue;
                    }
                };
                let deg = match op_degree(&a.op) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Warning: skipping action: {e}");
                        continue;
                    }
                };

                let expected = generators[to_idx].degree - generators[from_idx].degree;
                if deg as i32 != expected {
                    eprintln!(
                        "Warning: degree mismatch for {} {} = {}: op degree {} != {}",
                        a.op, a.from, a.to, deg, expected
                    );
                    continue;
                }

                edges.push(Edge {
                    op_name: a.op.clone(),
                    op_degree: deg,
                    from_idx,
                    to_idx,
                });
            }
        }
    }

    Ok(Module {
        name: def.name.unwrap_or_else(|| name.to_string()),
        prime: def.p,
        generators,
        edges,
    })
}

fn build_from_native(def: NativeModuleDef) -> Result<Module> {
    let generators: Vec<Generator> = def
        .gens
        .iter()
        .enumerate()
        .map(|(i, (name, &deg))| Generator {
            name: name.clone(),
            degree: deg,
            index: i,
        })
        .collect();

    let gen_index: IndexMap<&str, usize> = generators
        .iter()
        .map(|g| (g.name.as_str(), g.index))
        .collect();

    let mut edges = Vec::new();
    for a in &def.actions {
        let from_idx = *gen_index
            .get(a.from.as_str())
            .with_context(|| format!("unknown generator '{}'", a.from))?;
        let to_idx = *gen_index
            .get(a.to.as_str())
            .with_context(|| format!("unknown generator '{}'", a.to))?;
        let deg = op_degree(&a.op)?;

        let expected = generators[to_idx].degree - generators[from_idx].degree;
        if deg as i32 != expected {
            bail!(
                "degree mismatch for {} {} = {}: op degree {deg} != {expected}",
                a.op,
                a.from,
                a.to
            );
        }

        edges.push(Edge {
            op_name: a.op.clone(),
            op_degree: deg,
            from_idx,
            to_idx,
        });
    }

    Ok(Module {
        name: def.name,
        prime: def.prime,
        generators,
        edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_degree() {
        assert_eq!(op_degree("Sq1").unwrap(), 1);
        assert_eq!(op_degree("Sq2").unwrap(), 2);
        assert_eq!(op_degree("Sq4").unwrap(), 4);
        assert_eq!(op_degree("Sq8").unwrap(), 8);
        assert_eq!(op_degree("Sq16").unwrap(), 16);
        assert!(op_degree("P1").is_err());
    }

    #[test]
    fn test_parse_action_string() {
        let parsed = parse_action_string("Sq1 x0 = x1").unwrap();
        assert_eq!(parsed.op, "Sq1");
        assert_eq!(parsed.from, "x0");
        assert_eq!(parsed.targets, vec!["x1"]);
    }

    #[test]
    fn test_parse_action_zero_rhs() {
        let parsed = parse_action_string("Sq1 x0 = 0").unwrap();
        assert_eq!(parsed.op, "Sq1");
        assert_eq!(parsed.from, "x0");
        assert!(parsed.targets.is_empty());
    }

    #[test]
    fn test_parse_action_sum_rhs() {
        let parsed = parse_action_string("Sq2 x0 = x2 + x3").unwrap();
        assert_eq!(parsed.op, "Sq2");
        assert_eq!(parsed.from, "x0");
        assert_eq!(parsed.targets, vec!["x2", "x3"]);
    }

    #[test]
    fn test_parse_action_whitespace() {
        let parsed = parse_action_string("  Sq2   x0  =  x2  +  x3  ").unwrap();
        assert_eq!(parsed.op, "Sq2");
        assert_eq!(parsed.from, "x0");
        assert_eq!(parsed.targets, vec!["x2", "x3"]);
    }

    #[test]
    fn test_sseq_format() {
        let json = r#"{
            "p": 2,
            "type": "finite dimensional module",
            "gens": { "x0": 0, "x1": 1 },
            "actions": ["Sq1 x0 = x1"]
        }"#;
        let m = load_module_from_str(json, "C2").unwrap();
        assert_eq!(m.name, "C2");
        assert_eq!(m.prime, 2);
        assert_eq!(m.generators.len(), 2);
        assert_eq!(m.edges.len(), 1);
        assert_eq!(m.edges[0].op_degree, 1);
    }

    #[test]
    fn test_native_format() {
        let json = r#"{
            "name": "C2",
            "prime": 2,
            "gens": { "x0": 0, "x1": 1 },
            "actions": [{ "op": "Sq1", "from": "x0", "to": "x1" }]
        }"#;
        let m = load_module_from_str(json, "fallback").unwrap();
        assert_eq!(m.name, "C2");
        assert_eq!(m.edges.len(), 1);
    }

    #[test]
    fn test_degree_mismatch() {
        let json = r#"{
            "p": 2,
            "type": "finite dimensional module",
            "gens": { "x0": 0, "x1": 1 },
            "actions": ["Sq2 x0 = x1"]
        }"#;
        // Degree mismatch now warns and skips rather than failing
        let m = load_module_from_str(json, "bad").unwrap();
        assert_eq!(m.edges.len(), 0);
    }

    #[test]
    fn test_joker() {
        let json = r#"{
            "p": 2,
            "type": "finite dimensional module",
            "gens": { "x0": 0, "x1": 1, "x2": 2, "x3": 3, "x4": 4 },
            "actions": [
                "Sq1 x0 = x1",
                "Sq2 x0 = x2",
                "Sq2 x1 = x3",
                "Sq2 x2 = x4",
                "Sq1 x3 = x4"
            ]
        }"#;
        let m = load_module_from_str(json, "Joker").unwrap();
        assert_eq!(m.generators.len(), 5);
        assert_eq!(m.edges.len(), 5);
    }

    #[test]
    fn test_zero_action() {
        let json = r#"{
            "p": 2,
            "type": "finite dimensional module",
            "gens": { "x0": 0, "x1": 1 },
            "actions": ["Sq1 x0 = 0"]
        }"#;
        let m = load_module_from_str(json, "test").unwrap();
        assert_eq!(m.edges.len(), 0);
    }

    #[test]
    fn test_sum_action() {
        let json = r#"{
            "p": 2,
            "type": "finite dimensional module",
            "gens": { "a": 0, "b": 1, "c": 2, "d": 2 },
            "actions": ["Sq2 a = c + d"]
        }"#;
        let m = load_module_from_str(json, "sum").unwrap();
        assert_eq!(m.edges.len(), 2);
        assert_eq!(m.edges[0].to_idx, 2); // c
        assert_eq!(m.edges[1].to_idx, 3); // d
    }

    #[test]
    fn test_unknown_generator_skipped() {
        let json = r#"{
            "p": 2,
            "type": "finite dimensional module",
            "gens": { "x0": 0, "x1": 1 },
            "actions": ["Sq1 x0 = x1", "Sq1 x0 = unknown"]
        }"#;
        let m = load_module_from_str(json, "test").unwrap();
        assert_eq!(m.edges.len(), 1); // only the valid one
    }
}
