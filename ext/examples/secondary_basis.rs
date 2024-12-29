use std::sync::Arc;

use ext::{chain_complex::FreeChainComplex, secondary::*, utils::query_module};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging();

    let resolution = Arc::new(query_module(Some(algebra::AlgebraType::Milnor), true)?);

    let lift = SecondaryResolution::new(Arc::clone(&resolution));
    lift.extend_all();

    let sseq = lift.e3_page();
    let get_r = |b: Bidegree| {
        let d = sseq.page_data(b.n(), b.s() as i32);
        std::cmp::min(3, d.len() - 1)
    };

    for b in resolution.iter_nonzero_stem() {
        let r = get_r(b);

        for element in sseq.bze_basis(b.n(), b.s() as i32, r) {
            println!("{}", element);
        }
    }

    Ok(())
}
