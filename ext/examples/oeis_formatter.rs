use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let file = File::open("rp_d2_sweep.csv").expect("Failed to open CSV file");
    let reader = BufReader::new(file);

    let mut ranks = Vec::new();
    let mut max_degrees = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("Failed to read line");
        if i == 0 {
            continue; // Skip header
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 2 {
            continue;
        }

        let n: i32 = parts[0].parse().expect("Failed to parse n");
        let num_patterns: u128 = parts[1].parse().expect("Failed to parse num_patterns");

        // Compute rank as log2
        let rank = if num_patterns == 0 {
            0
        } else {
            (num_patterns as f64).log2() as i32
        };

        ranks.push(rank);
        max_degrees.push(n);
    }

    println!("=== OEIS Submission Data ===\n");

    println!("SEQUENCE NAME:");
    println!("Rank of the space of secondary differentials d_2 for the real projective space RP^n.\n");

    println!("OFFSET:");
    println!("2,11\n");

    println!("DATA (a(n) for n=2..{}):", max_degrees.last().unwrap());
    print_sequence(&ranks, 80);
    println!();

    println!("FORMULA:");
    println!("a(n) = log_2(number of distinct secondary Steenrod differentials d_2 for RP^n)\n");

    println!("EXAMPLE:");
    if ranks.len() > 9 {
        println!("a(11) = {} because there are {} = 2^{} distinct d_2 patterns for RP^11.",
                 ranks[9], 1u128 << ranks[9], ranks[9]);
    }
    println!();

    println!("COMMENTS:");
    println!("The sequence measures ambiguity in lifting the classical Adams spectral sequence");
    println!("differential d_2 to secondary operations in the Steenrod algebra.");
    println!("Values are powers of 2 by construction (dimension of an F_2 vector space).\n");

    println!("=== Analysis ===\n");

    // Find positions where rank > 0
    println!("Non-zero ranks occur at n =");
    let nonzero: Vec<i32> = max_degrees.iter().zip(&ranks)
        .filter(|(_, &r)| r > 0)
        .map(|(&n, _)| n)
        .collect();
    print_sequence(&nonzero, 80);
    println!();

    // Find positions for each rank value
    for rank_val in 1..=4 {
        let positions: Vec<i32> = max_degrees.iter().zip(&ranks)
            .filter(|(_, &r)| r == rank_val)
            .map(|(&n, _)| n)
            .collect();
        if !positions.is_empty() {
            println!("\nRank {} occurs at n =", rank_val);
            print_sequence(&positions, 80);
        }
    }

    println!("\n\n=== Modular Patterns ===\n");

    // Check patterns modulo various numbers
    for modulus in [4, 8, 16] {
        println!("n mod {} for non-zero rank:", modulus);
        let residues: Vec<i32> = nonzero.iter().map(|&n| n % modulus).collect();
        print_sequence(&residues, 80);
        println!();
    }
}

fn print_sequence<T: std::fmt::Display>(seq: &[T], line_width: usize) {
    let mut line = String::new();
    for (i, val) in seq.iter().enumerate() {
        let term = if i == seq.len() - 1 {
            format!("{}", val)
        } else {
            format!("{},", val)
        };

        if line.len() + term.len() > line_width {
            println!("{}", line);
            line = term;
        } else {
            if !line.is_empty() {
                line.push_str(&term);
            } else {
                line = term;
            }
        }
    }
    if !line.is_empty() {
        println!("{}", line);
    }
}
