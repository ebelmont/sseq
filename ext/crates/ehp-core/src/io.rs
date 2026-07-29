use std::collections::HashMap as StdHashMap;
use std::io::{self, BufRead, BufReader, Read as IoRead, Write};
use std::path::Path;

use fp::vector::FpVector;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::constraints::DiffVar;
use crate::element::Element;
use crate::gf2::*;
use crate::map::MapKind;
use crate::page::{MapTable, SATPage};
use crate::products::{ProductKey, ProductMatrix};
use crate::tridegree::Tridegree;

/// JSON manifest for a page.
#[derive(Serialize, Deserialize)]
pub struct PageManifest {
    pub r: i32,
    pub max_n: Option<i32>,
    pub max_s: Option<i32>,
    pub max_f: Option<i32>,
    pub max_t: Option<i32>,
    pub dimensions: Vec<DimensionEntry>,
}

#[derive(Serialize, Deserialize)]
pub struct DimensionEntry {
    pub n: i32,
    pub s: i32,
    pub f: i32,
    pub dim: usize,
}

/// Load a spectral sequence page from the Python CSV format.
///
/// Expects files:
/// - `{prefix}_rank.csv` or `{prefix}/E{r}_rank.csv` — dimensions
/// - `{prefix}_relations.csv` or `{prefix}/E{r}_relations.csv` — products
/// - `{prefix}_E.csv`, `_H.csv`, `_P.csv` — map tables
/// - `{prefix}_names.json` — element names
pub fn load_from_csv(prefix: &str, r: i32, max_t: i32) -> io::Result<SATPage> {
    let is_dir = Path::new(prefix).is_dir();
    let (rank_file, relations_file, e_file, h_file, p_file, lh0_file, names_file) = if is_dir {
        (
            format!("{}/E{}_rank.csv", prefix, r),
            format!("{}/E{}_relations.csv", prefix, r),
            format!("{}/E{}_E.csv", prefix, r),
            format!("{}/E{}_H.csv", prefix, r),
            format!("{}/E{}_P.csv", prefix, r),
            format!("{}/E{}_lh0.csv", prefix, r),
            format!("{}/E{}_names.json", prefix, r),
        )
    } else {
        (
            format!("{}_rank.csv", prefix),
            format!("{}_relations.csv", prefix),
            format!("{}_E.csv", prefix),
            format!("{}_H.csv", prefix),
            format!("{}_P.csv", prefix),
            format!("{}_lh0.csv", prefix),
            format!("{}_names.json", prefix),
        )
    };

    let mut page = SATPage::new(r);

    // Load dimensions
    eprintln!("Loading dimensions from {}...", rank_file);
    load_dimensions(&rank_file, &mut page, max_t)?;
    page.compute_max_values();
    let tridegree_count = page.dimension.values().filter(|&&d| d > 0).count();
    eprintln!("  Loaded {} tridegrees with non-zero dimension", tridegree_count);

    // Build page (basis elements at each tridegree)
    for (&t, &dim) in &page.dimension {
        if dim > 0 {
            let basis: Vec<Element> = (0..dim)
                .map(|i| Element::basis(t, dim, i))
                .collect();
            page.page.insert(t, basis);
        }
    }

    // Load products
    eprintln!("Loading multiplication table from {}...", relations_file);
    let product_count = load_products(&relations_file, &mut page)?;
    let shared = page.products.dedup_shared_blocks();
    eprintln!("  Loaded {} products ({} duplicate blocks share storage)", product_count, shared);

    // Load maps (E/H/P are the EHP triple; lh0 is the extra display/induced
    // map — same element→image CSV shape, loaded here so page turns can
    // induce it and the stem view can render it).
    for (kind, file) in [
        (MapKind::E, &e_file),
        (MapKind::H, &h_file),
        (MapKind::P, &p_file),
        (MapKind::Lh0, &lh0_file),
    ] {
        match load_map_csv(file, kind, &page) {
            Ok((mut map_table, count)) => {
                let shared = map_table.dedup_shared();
                page.maps.insert(kind, map_table);
                eprintln!("  {} map: {} entries ({} share storage)", kind.name(), count, shared);
            }
            Err(_) => {
                eprintln!("  {} map: file not found, using empty table", kind.name());
            }
        }
    }

    // Load names
    match load_names(&names_file) {
        Ok(names) => {
            eprintln!("  Loaded {} element names", names.len());
            page.names = names;
        }
        Err(_) => {
            eprintln!("  Names file not found, using empty dictionary");
        }
    }

    eprintln!("Successfully loaded E_{} page", r);
    Ok(page)
}

/// Parse an element string like "n_s_f" or "n_s_f_i" and look up in the page.
fn parse_element_ref(s: &str, _page: &SATPage) -> Option<(Tridegree, usize)> {
    let s = s.trim().trim_matches('"');
    if s == "0" {
        return None;
    }

    let parts: Vec<&str> = s.split('_').collect();
    if parts.len() < 3 {
        return None;
    }

    let n: i32 = parts[0].parse().ok()?;
    let sv: i32 = parts[1].parse().ok()?;
    let f: i32 = parts[2].parse().ok()?;
    let i: usize = if parts.len() >= 4 {
        parts[3].parse().ok()?
    } else {
        0
    };

    let t = Tridegree::new(n, sv, f);
    Some((t, i))
}

/// Parse a full element string (may contain " + " for sums).
fn parse_element_full(s: &str, page: &SATPage) -> Option<Element> {
    let s = s.trim().trim_matches('"');
    if s == "0" {
        return None;
    }

    let monomials: Vec<&str> = s.split(" + ").collect();
    if monomials.is_empty() {
        return None;
    }

    let (first_t, _first_i) = parse_element_ref(monomials[0], page)?;
    let dim = page.dim_at(first_t);
    if dim == 0 {
        return None;
    }

    let mut vec = vec_zero(dim);
    for m in &monomials {
        let (t, i) = parse_element_ref(m, page)?;
        if t != first_t {
            return None; // Summands must be in same tridegree
        }
        if i < dim {
            vec_flip(&mut vec, i);
        }
    }

    Some(Element::new(first_t, vec))
}

fn load_dimensions(path: &str, page: &mut SATPage, max_t: i32) -> io::Result<()> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    let header = lines.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty file"))??;
    let headers: Vec<&str> = header.split(',').map(|s| s.trim().trim_matches('"')).collect();

    let n_idx = headers.iter().position(|&h| h == "n").unwrap_or(0);
    let s_idx = headers.iter().position(|&h| h == "s").unwrap_or(1);
    let f_idx = headers.iter().position(|&h| h == "f").unwrap_or(2);
    let dim_idx = headers.iter().position(|&h| h == "dimension").unwrap_or(3);

    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split(',').map(|s| s.trim().trim_matches('"')).collect();
        if fields.len() <= dim_idx {
            continue;
        }
        let n: i32 = match fields[n_idx].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let s: i32 = match fields[s_idx].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let f: i32 = match fields[f_idx].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dim: usize = match fields[dim_idx].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        if s + f > max_t {
            continue;
        }

        let t = Tridegree::new(n, s, f);
        page.dimension.insert(t, dim);
    }

    Ok(())
}

fn load_products(path: &str, page: &mut SATPage) -> io::Result<usize> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(0),
    };
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut count = 0;

    // Skip header
    lines.next();

    for line in lines {
        let line = line?;
        // CSV with quoted fields: "factor1","factor2","result"
        let fields: Vec<&str> = line.split(',').map(|s| s.trim().trim_matches('"')).collect();
        if fields.len() < 3 {
            continue;
        }

        let factor1_str = fields[0];
        let factor2_str = fields[1];

        let (t1, i1) = match parse_element_ref(factor1_str, page) {
            Some(v) => v,
            None => continue,
        };
        let (t2, i2) = match parse_element_ref(factor2_str, page) {
            Some(v) => v,
            None => continue,
        };

        let dim1 = page.dim_at(t1);
        let dim2 = page.dim_at(t2);
        // A factor outside the loaded s+f window has dim 0 here: no product
        // query can ever reference this row (multiply on a 0-dim factor is
        // zero, and ProductTable::multiply treats absent blocks as zero).
        // Skipping saves the result-string allocation and ~2/3 of the table
        // inserts at typical max_t — the load was several minutes of startup.
        if dim1 == 0 || dim2 == 0 {
            continue;
        }

        let result_str = fields[2..].join(","); // Handle commas in result

        let result = match parse_element_full(result_str.trim_matches('"'), page) {
            Some(e) => e.vec,
            None => {
                // Zero product
                let prod_t = Tridegree::new(t1.n, t1.s + t2.s, t1.f + t2.f);
                let dim = page.dim_at(prod_t);
                vec_zero(dim)
            }
        };

        let key = ProductKey::new(t1, i1 as u16, t2, i2 as u16);
        page.products.insert(key, result, dim1, dim2);
        count += 1;
    }

    Ok(count)
}

fn load_map_csv(
    path: &str,
    kind: MapKind,
    page: &SATPage,
) -> io::Result<(MapTable, usize)> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut map_table = MapTable::new(kind);
    let mut count = 0;

    // Skip header
    lines.next();

    // Collect per-tridegree entries, then build matrices
    let mut entries_by_src: HashMap<Tridegree, Vec<(usize, FpVector)>> = HashMap::new();

    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split(',').map(|s| s.trim().trim_matches('"')).collect();
        if fields.len() < 2 {
            continue;
        }

        let elem_str = fields[0];

        let (src_t, src_i) = match parse_element_ref(elem_str, page) {
            Some(v) => v,
            None => continue,
        };

        // Sources outside the loaded window never produce a matrix (the
        // build loop below skips src_dim == 0) — skip before the string work.
        if page.dim_at(src_t) == 0 {
            continue;
        }

        let tgt_t = kind.target_degree(src_t);
        let tgt_dim = page.dim_at(tgt_t);

        let image_str = fields[1..].join(",");
        let image_str = image_str.trim_matches('"');

        let image_vec = if image_str.trim() == "0" || tgt_dim == 0 {
            vec_zero(tgt_dim)
        } else {
            match parse_element_full(image_str, page) {
                Some(e) => e.vec,
                None => vec_zero(tgt_dim),
            }
        };

        entries_by_src
            .entry(src_t)
            .or_default()
            .push((src_i, image_vec));
        count += 1;
    }

    // Build matrices for each source tridegree
    for (src_t, entries) in entries_by_src {
        let src_dim = page.dim_at(src_t);
        let tgt_t = kind.target_degree(src_t);
        let tgt_dim = page.dim_at(tgt_t);
        if src_dim == 0 || tgt_dim == 0 {
            continue;
        }

        // Matrix: src_dim rows × tgt_dim cols
        // Row i = image of basis element i
        let mut mat = mat_zero(src_dim, tgt_dim);
        for (i, vec) in &entries {
            if *i < src_dim && vec.len() == tgt_dim {
                mat_set_row(&mut mat, *i, vec);
            }
        }
        map_table.set_matrix(src_t, mat);
    }

    Ok((map_table, count))
}

fn load_names(path: &str) -> io::Result<HashMap<String, String>> {
    let contents = std::fs::read_to_string(path)?;
    let parsed: StdHashMap<String, String> = serde_json::from_str(&contents)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(parsed.into_iter().collect())
}

/// Save a page to the new JSON+binary format.
pub fn save_page_json(page: &SATPage, directory: &str) -> io::Result<()> {
    std::fs::create_dir_all(directory)?;

    // Write manifest
    let manifest = PageManifest {
        r: page.r,
        max_n: page.max_n,
        max_s: page.max_s,
        max_f: page.max_f,
        max_t: page.max_t,
        dimensions: page
            .dimension
            .iter()
            .filter(|(_, &d)| d > 0)
            .map(|(&t, &d)| DimensionEntry {
                n: t.n,
                s: t.s,
                f: t.f,
                dim: d,
            })
            .collect(),
    };

    let manifest_path = format!("{}/manifest.json", directory);
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    std::fs::write(&manifest_path, manifest_json)?;

    // Write dimensions CSV (Python-compatible)
    let rank_path = format!("{}/E{}_rank.csv", directory, page.r);
    let mut rank_file = std::fs::File::create(&rank_path)?;
    writeln!(rank_file, "n,s,f,dimension")?;
    let mut dims: Vec<(&Tridegree, &usize)> = page.dimension.iter().collect();
    dims.sort_by_key(|(t, _)| (t.n, t.s, t.f));
    for (&t, &d) in &dims {
        if d > 0 {
            writeln!(rank_file, "{},{},{},{}", t.n, t.s, t.f, d)?;
        }
    }

    // Write products CSV (Python-compatible)
    let relations_path = format!("{}/E{}_relations.csv", directory, page.r);
    let mut rel_file = std::fs::File::create(&relations_path)?;
    writeln!(rel_file, "\"factor1\",\"factor2\",\"result\"")?;
    for (key, result) in page.products.iter() {
        if result.is_zero() {
            continue;
        }
        let f1 = format!(
            "{}_{}_{}_{}", key.deg1.n, key.deg1.s, key.deg1.f, key.idx1
        );
        let f2 = format!(
            "{}_{}_{}_{}", key.deg2.n, key.deg2.s, key.deg2.f, key.idx2
        );
        let result_elem = Element::new(
            Tridegree::new(key.deg1.n, key.deg1.s + key.deg2.s, key.deg1.f + key.deg2.f),
            result.clone(),
        );
        writeln!(rel_file, "\"{}\",\"{}\",\"{}\"", f1, f2, result_elem)?;
    }

    // Write map CSVs (include lh0 so it round-trips through CSV save/load)
    for kind in MapKind::all_with_lh0() {
        if let Some(map_table) = page.maps.get(&kind) {
            let map_path = format!("{}/E{}_{}.csv", directory, page.r, kind.name());
            let mut map_file = std::fs::File::create(&map_path)?;
            writeln!(map_file, "\"element\",\"image\"")?;
            for (&src_t, mat) in &map_table.matrices {
                let src_dim = page.dim_at(src_t);
                let tgt_t = kind.target_degree(src_t);
                for i in 0..src_dim {
                    let row = mat.row_vec(i);
                    let elem_str = if src_dim == 1 {
                        format!("{}_{}_{}", src_t.n, src_t.s, src_t.f)
                    } else {
                        format!("{}_{}_{}_{}", src_t.n, src_t.s, src_t.f, i)
                    };
                    let image_elem = Element::new(tgt_t, row);
                    writeln!(map_file, "\"{}\",\"{}\"", elem_str, image_elem)?;
                }
            }
        }
    }

    // Write names
    if !page.names.is_empty() {
        let names_path = format!("{}/E{}_names.json", directory, page.r);
        let names_std: StdHashMap<&String, &String> = page.names.iter().collect();
        let names_json = serde_json::to_string_pretty(&names_std)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(&names_path, names_json)?;
    }

    Ok(())
}

// =============================================================================
// Binary format (.ehp)
// =============================================================================

const BINARY_MAGIC: &[u8; 4] = b"EHPB";
const BINARY_VERSION: u32 = 1;

// Section type constants
const SECTION_DIMENSIONS: u32 = 1;
const SECTION_PRODUCTS: u32 = 2;
const SECTION_MAP_E: u32 = 3;
const SECTION_MAP_H: u32 = 4;
const SECTION_MAP_P: u32 = 5;
const SECTION_NAMES: u32 = 6;
const SECTION_PRODUCTS_V2: u32 = 7;
const SECTION_MAP_E_V2: u32 = 8;
const SECTION_MAP_H_V2: u32 = 9;
const SECTION_MAP_P_V2: u32 = 10;
const SECTION_MAP_LH0_V2: u32 = 11;

fn section_type_for_map(kind: MapKind) -> u32 {
    match kind {
        MapKind::E => SECTION_MAP_E_V2,
        MapKind::H => SECTION_MAP_H_V2,
        MapKind::P => SECTION_MAP_P_V2,
        MapKind::Lh0 => SECTION_MAP_LH0_V2,
    }
}

/// Legacy fp-limb map sections (old .ehp files).
fn map_kind_for_section(section_type: u32) -> Option<MapKind> {
    match section_type {
        SECTION_MAP_E => Some(MapKind::E),
        SECTION_MAP_H => Some(MapKind::H),
        SECTION_MAP_P => Some(MapKind::P),
        _ => None,
    }
}

/// Compact V2 map sections (raw block words).
fn map_kind_for_section_v2(section_type: u32) -> Option<MapKind> {
    match section_type {
        SECTION_MAP_E_V2 => Some(MapKind::E),
        SECTION_MAP_H_V2 => Some(MapKind::H),
        SECTION_MAP_P_V2 => Some(MapKind::P),
        SECTION_MAP_LH0_V2 => Some(MapKind::Lh0),
        _ => None,
    }
}

// --- Binary write helpers ---

pub(crate) fn write_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn write_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn write_i32(w: &mut Vec<u8>, v: i32) {
    w.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn write_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}

// --- Binary read helpers ---

pub(crate) fn read_u16(data: &[u8], pos: &mut usize) -> io::Result<u16> {
    if *pos + 2 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_u16"));
    }
    let v = u16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap());
    *pos += 2;
    Ok(v)
}

pub(crate) fn read_u32(data: &[u8], pos: &mut usize) -> io::Result<u32> {
    if *pos + 4 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_u32"));
    }
    let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

pub(crate) fn read_i32(data: &[u8], pos: &mut usize) -> io::Result<i32> {
    if *pos + 4 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_i32"));
    }
    let v = i32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

pub(crate) fn read_u64(data: &[u8], pos: &mut usize) -> io::Result<u64> {
    if *pos + 8 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_u64"));
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

/// Save a page to the binary `.ehp` format.
pub fn save_to_binary(page: &SATPage, path: &str) -> io::Result<()> {
    // Build all sections first, then write header + directory + sections
    let mut sections: Vec<(u32, Vec<u8>)> = Vec::new();

    // Section 1: DIMENSIONS
    {
        let mut buf = Vec::new();
        let dims: Vec<(&Tridegree, &usize)> = page.dimension.iter()
            .filter(|(_, &d)| d > 0)
            .collect();
        write_u32(&mut buf, dims.len() as u32);
        for (&t, &d) in &dims {
            write_i32(&mut buf, t.n);
            write_i32(&mut buf, t.s);
            write_i32(&mut buf, t.f);
            write_u32(&mut buf, d as u32);
        }
        sections.push((SECTION_DIMENSIONS, buf));
    }

    // Section 2: PRODUCTS
    {
        let mut buf = Vec::new();
        let blocks: Vec<_> = page.products.iter_blocks().collect();
        write_u32(&mut buf, blocks.len() as u32);
        for (&(deg1, deg2), pm) in &blocks {
            write_i32(&mut buf, deg1.n);
            write_i32(&mut buf, deg1.s);
            write_i32(&mut buf, deg1.f);
            write_i32(&mut buf, deg2.n);
            write_i32(&mut buf, deg2.s);
            write_i32(&mut buf, deg2.f);
            write_u16(&mut buf, pm.dim1);
            write_u16(&mut buf, pm.dim2);
            write_u16(&mut buf, pm.tgt_dim);
            // Compact layout: the block's bit-packed words verbatim (no fp
            // Matrix round trip — that cost more than the CSV parse).
            let words = pm.raw_words();
            write_u32(&mut buf, words.len() as u32);
            for &w in words {
                write_u64(&mut buf, w);
            }
        }
        sections.push((SECTION_PRODUCTS_V2, buf));
    }

    // Sections 8/9/10/11: MAP_E/H/P/LH0 (compact V2 — raw block words verbatim)
    for kind in MapKind::all_with_lh0() {
        if let Some(map_table) = page.maps.get(&kind) {
            if map_table.matrices.is_empty() {
                continue;
            }
            let mut buf = Vec::new();
            let entries: Vec<_> = map_table.matrices.iter().collect();
            write_u32(&mut buf, entries.len() as u32);
            for (&t, mat) in &entries {
                write_i32(&mut buf, t.n);
                write_i32(&mut buf, t.s);
                write_i32(&mut buf, t.f);
                write_u16(&mut buf, mat.dim1);
                write_u16(&mut buf, mat.tgt_dim);
                let words = mat.raw_words();
                write_u32(&mut buf, words.len() as u32);
                for &w in words {
                    write_u64(&mut buf, w);
                }
            }
            sections.push((section_type_for_map(kind), buf));
        }
    }

    // Section 6: NAMES
    if !page.names.is_empty() {
        let names_std: StdHashMap<&String, &String> = page.names.iter().collect();
        let json = serde_json::to_string(&names_std)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let mut buf = Vec::new();
        let json_bytes = json.as_bytes();
        write_u32(&mut buf, json_bytes.len() as u32);
        buf.extend_from_slice(json_bytes);
        sections.push((SECTION_NAMES, buf));
    }

    // Now build the file
    let num_sections = sections.len() as u32;

    // Header: 32 bytes
    let mut header = Vec::with_capacity(32);
    header.extend_from_slice(BINARY_MAGIC);     // 4
    write_u32(&mut header, BINARY_VERSION);      // 4
    write_i32(&mut header, page.r);              // 4
    write_i32(&mut header, page.max_n.unwrap_or(0)); // 4
    write_i32(&mut header, page.max_s.unwrap_or(0)); // 4
    write_i32(&mut header, page.max_f.unwrap_or(0)); // 4
    write_i32(&mut header, page.max_t.unwrap_or(0)); // 4
    write_u32(&mut header, num_sections);         // 4
    assert_eq!(header.len(), 32);

    // Directory: 20 bytes per section
    let dir_size = num_sections as usize * 20;
    let data_start = 32 + dir_size;

    let mut directory = Vec::with_capacity(dir_size);
    let mut offset = data_start as u64;
    for (section_type, section_data) in &sections {
        write_u32(&mut directory, *section_type);
        write_u64(&mut directory, offset);
        write_u64(&mut directory, section_data.len() as u64);
        offset += section_data.len() as u64;
    }

    // Write everything
    let mut file = std::fs::File::create(path)?;
    file.write_all(&header)?;
    file.write_all(&directory)?;
    for (_, section_data) in &sections {
        file.write_all(section_data)?;
    }

    Ok(())
}

/// Load a page from the binary `.ehp` format.
pub fn load_from_binary(path: &str, filter_max_t: Option<i32>) -> io::Result<SATPage> {
    let mut file = std::fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    if data.len() < 32 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "file too short for header"));
    }

    // Parse header
    if &data[0..4] != BINARY_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic bytes"));
    }
    let mut pos = 4usize;
    let version = read_u32(&data, &mut pos)?;
    if version != BINARY_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported version {}", version),
        ));
    }
    let r = read_i32(&data, &mut pos)?;
    let max_n = read_i32(&data, &mut pos)?;
    let max_s = read_i32(&data, &mut pos)?;
    let max_f = read_i32(&data, &mut pos)?;
    let max_t = read_i32(&data, &mut pos)?;
    let num_sections = read_u32(&data, &mut pos)?;

    let mut page = SATPage::new(r);
    page.max_n = Some(max_n);
    page.max_s = Some(max_s);
    page.max_f = Some(max_f);
    page.max_t = Some(max_t);

    // Parse section directory
    struct SectionEntry {
        section_type: u32,
        offset: u64,
        length: u64,
    }

    let mut directory = Vec::with_capacity(num_sections as usize);
    for _ in 0..num_sections {
        let section_type = read_u32(&data, &mut pos)?;
        let offset = read_u64(&data, &mut pos)?;
        let length = read_u64(&data, &mut pos)?;
        directory.push(SectionEntry { section_type, offset, length });
    }

    // Process sections
    for entry in &directory {
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        if end > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "section extends past EOF"));
        }
        let section = &data[start..end];
        let mut sp = 0usize;

        match entry.section_type {
            SECTION_DIMENSIONS => {
                let count = read_u32(section, &mut sp)? as usize;
                for _ in 0..count {
                    let n = read_i32(section, &mut sp)?;
                    let s = read_i32(section, &mut sp)?;
                    let f = read_i32(section, &mut sp)?;
                    let dim = read_u32(section, &mut sp)? as usize;
                    // Filter by max_t if provided
                    if let Some(mt) = filter_max_t {
                        if s + f > mt {
                            continue;
                        }
                    }
                    let t = Tridegree::new(n, s, f);
                    page.dimension.insert(t, dim);
                    if dim > 0 {
                        let basis: Vec<Element> = (0..dim)
                            .map(|i| Element::basis(t, dim, i))
                            .collect();
                        page.page.insert(t, basis);
                    }
                }
            }

            SECTION_PRODUCTS_V2 => {
                let block_count = read_u32(section, &mut sp)? as usize;
                for _ in 0..block_count {
                    let n1 = read_i32(section, &mut sp)?;
                    let s1 = read_i32(section, &mut sp)?;
                    let f1 = read_i32(section, &mut sp)?;
                    let n2 = read_i32(section, &mut sp)?;
                    let s2 = read_i32(section, &mut sp)?;
                    let f2 = read_i32(section, &mut sp)?;
                    let dim1 = read_u16(section, &mut sp)?;
                    let dim2 = read_u16(section, &mut sp)?;
                    let tgt_dim = read_u16(section, &mut sp)?;
                    let num_words = read_u32(section, &mut sp)? as usize;
                    if let Some(mt) = filter_max_t {
                        if s1 + f1 > mt || s2 + f2 > mt {
                            sp += num_words * 8;
                            continue;
                        }
                    }
                    let mut words = Vec::with_capacity(num_words);
                    for _ in 0..num_words {
                        words.push(read_u64(section, &mut sp)?);
                    }
                    let pm = ProductMatrix::from_raw_parts(dim1, dim2, tgt_dim, words)
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "bad product block")
                        })?;
                    page.products.insert_block(
                        Tridegree::new(n1, s1, f1),
                        Tridegree::new(n2, s2, f2),
                        pm,
                    );
                }
            }

            SECTION_PRODUCTS => {
                let pair_count = read_u32(section, &mut sp)? as usize;
                for _ in 0..pair_count {
                    let n1 = read_i32(section, &mut sp)?;
                    let s1 = read_i32(section, &mut sp)?;
                    let f1 = read_i32(section, &mut sp)?;
                    let n2 = read_i32(section, &mut sp)?;
                    let s2 = read_i32(section, &mut sp)?;
                    let f2 = read_i32(section, &mut sp)?;
                    let dim1 = read_u16(section, &mut sp)?;
                    let dim2 = read_u16(section, &mut sp)?;
                    let tgt_dim = read_u16(section, &mut sp)?;
                    let num_words = read_u32(section, &mut sp)? as usize;
                    // Filter by max_t: skip blocks where either source is out of range
                    if let Some(mt) = filter_max_t {
                        if s1 + f1 > mt || s2 + f2 > mt {
                            sp += num_words * 8;
                            continue;
                        }
                    }
                    let mut words = Vec::with_capacity(num_words);
                    for _ in 0..num_words {
                        words.push(read_u64(section, &mut sp)?);
                    }
                    let nrows = dim1 as usize * dim2 as usize;
                    let matrix = mat_from_raw_words(nrows, tgt_dim as usize, words);
                    let deg1 = Tridegree::new(n1, s1, f1);
                    let deg2 = Tridegree::new(n2, s2, f2);
                    page.products.insert_block(
                        deg1,
                        deg2,
                        ProductMatrix::from_matrix(dim1, dim2, tgt_dim, &matrix),
                    );
                }
            }

            st if map_kind_for_section_v2(st).is_some() => {
                let kind = map_kind_for_section_v2(st).unwrap();
                let entry_count = read_u32(section, &mut sp)? as usize;
                let map_table = page.maps.entry(kind).or_insert_with(|| MapTable::new(kind));
                for _ in 0..entry_count {
                    let n = read_i32(section, &mut sp)?;
                    let s = read_i32(section, &mut sp)?;
                    let f = read_i32(section, &mut sp)?;
                    let rows = read_u16(section, &mut sp)?;
                    let tgt = read_u16(section, &mut sp)?;
                    let num_words = read_u32(section, &mut sp)? as usize;
                    if let Some(mt) = filter_max_t {
                        if s + f > mt {
                            sp += num_words * 8;
                            continue;
                        }
                    }
                    let mut words = Vec::with_capacity(num_words);
                    for _ in 0..num_words {
                        words.push(read_u64(section, &mut sp)?);
                    }
                    let block = ProductMatrix::from_raw_parts(rows, 1, tgt, words)
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "bad map block")
                        })?;
                    map_table.set_block(Tridegree::new(n, s, f), block);
                }
            }

            st if map_kind_for_section(st).is_some() => {
                let kind = map_kind_for_section(st).unwrap();
                let entry_count = read_u32(section, &mut sp)? as usize;
                let map_table = page.maps.entry(kind).or_insert_with(|| MapTable::new(kind));
                for _ in 0..entry_count {
                    let n = read_i32(section, &mut sp)?;
                    let s = read_i32(section, &mut sp)?;
                    let f = read_i32(section, &mut sp)?;
                    let nrows = read_u16(section, &mut sp)? as usize;
                    let ncols = read_u16(section, &mut sp)? as usize;
                    let num_words = read_u32(section, &mut sp)? as usize;
                    // Filter by max_t
                    if let Some(mt) = filter_max_t {
                        if s + f > mt {
                            sp += num_words * 8;
                            continue;
                        }
                    }
                    let mut words = Vec::with_capacity(num_words);
                    for _ in 0..num_words {
                        words.push(read_u64(section, &mut sp)?);
                    }
                    let mat = mat_from_raw_words(nrows, ncols, words);
                    let t = Tridegree::new(n, s, f);
                    map_table.set_matrix(t, mat);
                }
            }

            SECTION_NAMES => {
                let json_len = read_u32(section, &mut sp)? as usize;
                if sp + json_len > section.len() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "names section truncated"));
                }
                let json_str = std::str::from_utf8(&section[sp..sp + json_len])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let parsed: StdHashMap<String, String> = serde_json::from_str(json_str)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                page.names = parsed.into_iter().collect();
            }

            _ => {
                // Unknown section type, skip
            }
        }
    }

    // Update max values to reflect filtered data
    if filter_max_t.is_some() {
        page.compute_max_values();
    }

    let filtered_note = if let Some(mt) = filter_max_t {
        format!(", filtered to s+f <= {}", mt)
    } else {
        String::new()
    };
    eprintln!("Loaded binary E_{} page ({} tridegrees, {} product blocks{})",
        r,
        page.dimension.values().filter(|&&d| d > 0).count(),
        page.products.num_blocks(),
        filtered_note,
    );

    Ok(page)
}

/// Detected data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    Binary,
    Csv,
}

/// Detect whether a path points to binary (.ehp) or CSV data.
///
/// Checks the file extension first; if ambiguous, peeks at magic bytes.
pub fn detect_format(path: &str) -> DataFormat {
    if path.ends_with(".ehp") {
        return DataFormat::Binary;
    }

    // Try reading magic bytes
    if let Ok(mut f) = std::fs::File::open(path) {
        let mut magic = [0u8; 4];
        if f.read_exact(&mut magic).is_ok() && &magic == BINARY_MAGIC {
            return DataFormat::Binary;
        }
    }

    DataFormat::Csv
}

/// Load a page, auto-detecting binary vs CSV format.
///
/// `max_t` filters to tridegrees with `s + f <= max_t`.
pub fn load_page(path: &str, r: i32, max_t: i32) -> io::Result<SATPage> {
    match detect_format(path) {
        DataFormat::Binary => load_from_binary(path, Some(max_t)),
        DataFormat::Csv => load_from_csv(path, r, max_t),
    }
}

/// Load known differentials from a CSV file.
///
/// Format: `r, n, s, f, row, col, value` (one per line, no header).
/// Only entries whose `r` field matches `page_r` are loaded.
pub fn load_known_diffs(path: &str, page_r: i32) -> io::Result<HashMap<DiffVar, bool>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut diffs = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 7 {
            continue;
        }
        let r: i32 = match fields[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if r != page_r {
            continue;
        }
        let n: i32 = match fields[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let s: i32 = match fields[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let f: i32 = match fields[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let row: u16 = match fields[4].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let col: u16 = match fields[5].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let val: i32 = match fields[6].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        diffs.insert(DiffVar::new(n, s, f, row, col), val != 0);
    }

    Ok(diffs)
}
