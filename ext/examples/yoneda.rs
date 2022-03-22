use algebra::module::homomorphism::{FullModuleHomomorphism, IdentityHomomorphism};
use algebra::module::Module;
use ext::chain_complex::{AugmentedChainComplex, ChainComplex, FreeChainComplex};
use ext::resolution_homomorphism::ResolutionHomomorphism;
use ext::utils;
use ext::yoneda::yoneda_representative_element;
use serde_json::value::Value;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let resolution = Arc::new(utils::query_module_only("Module", None, false)?);

    let module = resolution.target().module(0);
    let min_degree = resolution.min_degree();

    let n: i32 = query::with_default("n", "20", str::parse);
    let s: u32 = query::with_default("s", "4", str::parse);
    let i: usize = query::with_default("idx", "0", str::parse);

    let start = Instant::now();
    let t = n + s as i32;

    resolution.compute_through_stem(s, n);

    eprintln!("Resolving time: {:?}", start.elapsed());

    let start = Instant::now();
    let yoneda = Arc::new(yoneda_representative_element(
        Arc::clone(&resolution),
        s,
        t,
        i,
    ));

    eprintln!("Finding representative time: {:?}", start.elapsed());

    let f = ResolutionHomomorphism::from_module_homomorphism(
        "".to_string(),
        Arc::clone(&resolution),
        Arc::clone(&yoneda),
        &FullModuleHomomorphism::identity_homomorphism(Arc::clone(&module)),
    );

    f.extend_through_stem(s, n);
    let final_map = f.get_map(s);
    let num_gens = resolution.number_of_gens_in_bidegree(s, t);
    for i_ in 0..num_gens {
        assert_eq!(final_map.output(t, i_).len(), 1);
        if i_ == i {
            assert_eq!(final_map.output(t, i_).entry(0), 1);
        } else {
            assert_eq!(final_map.output(t, i_).entry(0), 0);
        }
    }

    let mut check =
        bivec::BiVec::from_vec(min_degree, vec![0; t as usize + 1 - min_degree as usize]);
    for s in 0..=s {
        let module = yoneda.module(s);

        println!(
            "Dimension of {}th module is {}",
            s,
            module.total_dimension()
        );

        for t in min_degree..=t {
            check[t] += (if s % 2 == 0 { 1 } else { -1 }) * module.dimension(t) as i32;
        }
    }
    for t in min_degree..=t {
        assert_eq!(
            check[t],
            module.dimension(t) as i32,
            "Incorrect Euler characteristic at t = {}",
            t
        );
    }

    let filename: String = match query::optional("Output file name", str::parse) {
        None => return Ok(()),
        Some(x) => x,
    };

    let mut module_strings = Vec::with_capacity(s as usize + 2);

    #[cfg(feature = "nassau")]
    module_strings.push(module.to_minimal_json());

    #[cfg(not(feature = "nassau"))]
    module_strings.push(
        algebra::module::steenrod_module::as_fd_module(&module)
            .unwrap()
            .to_minimal_json(),
    );

    for s in 0..=s {
        module_strings.push(yoneda.module(s).to_minimal_json());
    }

    let mut output_path_buf = PathBuf::from(filename);
    output_path_buf.set_extension("json");
    std::fs::write(&output_path_buf, Value::from(module_strings).to_string())?;
    Ok(())
}
