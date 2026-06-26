use std::sync::Arc;

use algebra::{Algebra, module::Module};
use ext::chain_complex::{ChainComplex, HomCochainComplex};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    eprintln!("This script computes Ext(M, N)");
    let res = ext::utils::query_module_only("Module M", None, false)?;
    let module_spec = query::raw("Module N", ext::utils::parse_module_name);
    #[cfg(not(feature = "nassau"))]
    let module = algebra::module::steenrod_module::from_json(res.algebra(), &module_spec)?;

    #[cfg(feature = "nassau")]
    let module = algebra::module::FDModule::from_json(res.algebra(), &module_spec)?;

    let max = Bidegree::n_s(
        query::raw("Max n", str::parse),
        query::raw("Max s", str::parse),
    );

    res.compute_through_stem(max + Bidegree::n_s(module.max_degree().unwrap(), 1));
    res.algebra()
        .compute_basis(max.t() + module.max_degree().unwrap() + 2);

    let hom_cc = HomCochainComplex::new(Arc::new(res), Arc::new(module));
    hom_cc.compute_through_stem(max);

    // FreeChainComplex::graded_dimension_string
    let mut result = String::new();
    for s in (0..=max.s()).rev() {
        for n in hom_cc.min_degree()..=max.n() {
            let b = Bidegree::n_s(n, s);
            result.push(ext::utils::unicode_num(hom_cc.homology_dimension(b)));
            result.push(' ');
        }
        result.push('\n');
        // If it is empty so far, don't print anything
        if result.trim_start().is_empty() {
            result.clear()
        }
    }
    print!("{result}");

    Ok(())
}
