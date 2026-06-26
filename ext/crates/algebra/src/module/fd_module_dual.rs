use std::sync::Arc;

use bivec::BiVec;

use crate::algebra::Algebra;
use super::finite_dimensional_module::FiniteDimensionalModule;
use super::Module;

impl<A: Algebra> FiniteDimensionalModule<A> {
    /// F2-linear dual D(M).
    ///
    /// D(M) in degree `-d` has dimension equal to `M.dimension(d)`.
    /// The action of `(op_deg, op_idx)` on D(M) from degree `-(d+op_deg)` to `-d`
    /// is the transpose of M's action from degree `d` to `d+op_deg`.
    ///
    /// At p=2, the conjugation chi(Sq^n) = Sq^n, so this equals the module-theoretic dual.
    ///
    /// Panics if `p != 2` (odd-prime antipode support is not yet implemented).
    pub fn linear_dual(&self) -> Self {
        self.dual_with_shift(0)
    }

    /// Spanier-Whitehead dual: linear dual with degree shift.
    ///
    /// The resulting module has degrees shifted by `shift` relative to the linear dual,
    /// i.e., degrees in `[-max_deg + shift, -min_deg + shift]`.
    ///
    /// Panics if `p != 2`.
    pub fn spanier_whitehead_dual(&self, shift: i32) -> Self {
        self.dual_with_shift(shift)
    }

    fn dual_with_shift(&self, shift: i32) -> Self {
        let p = self.prime();
        assert!(
            p == 2,
            "Dual construction currently only supports p=2 (chi(Sq^n) = Sq^n)"
        );

        let min_deg = self.min_degree();
        let max_deg = match self.max_degree() {
            Some(d) => d,
            None => {
                // Empty module: return empty module with shifted min degree
                return Self::new(
                    self.algebra(),
                    format!("D({})", self.name),
                    BiVec::new(-min_deg + shift),
                );
            }
        };

        // D(M) has degrees [-max_deg + shift, -min_deg + shift]
        let dual_min = -max_deg + shift;
        let dual_max = -min_deg + shift;

        let mut graded_dim = BiVec::with_capacity(dual_min, dual_max + 1);
        for dual_d in dual_min..=dual_max {
            // dual_d corresponds to original degree (shift - dual_d)
            let orig_d = shift - dual_d;
            graded_dim.push(self.dimension(orig_d));
        }

        let algebra = self.algebra();
        let mut dual = Self::new(
            Arc::clone(&algebra),
            format!("D({})", self.name),
            graded_dim,
        );

        // Set ALL operations (not just generators), so no extend_actions needed.
        // Action of (op_deg, op_idx) on D(M) from degree dual_input to dual_output
        // where dual_output = dual_input + op_deg.
        //
        // This corresponds to the transpose of M's action from degree
        // (shift - dual_output) to (shift - dual_input), i.e., from degree
        // (shift - dual_input - op_deg) to (shift - dual_input).
        for dual_input in dual_min..=dual_max {
            for dual_output in (dual_input + 1)..=dual_max {
                let op_deg = dual_output - dual_input;

                let dual_input_dim = dual.dimension(dual_input);
                let dual_output_dim = dual.dimension(dual_output);
                if dual_input_dim == 0 || dual_output_dim == 0 {
                    continue;
                }

                // Original degrees: the transpose goes from
                // orig_out_deg = shift - dual_input (original output)
                // orig_in_deg = shift - dual_output (original input)
                // with orig_out_deg - orig_in_deg = op_deg
                let orig_in_deg = shift - dual_output;
                let orig_out_deg = shift - dual_input;

                // Check original degrees are valid
                if orig_in_deg < min_deg || orig_out_deg > max_deg {
                    continue;
                }

                let orig_in_dim = self.dimension(orig_in_deg);
                let orig_out_dim = self.dimension(orig_out_deg);
                if orig_in_dim == 0 || orig_out_dim == 0 {
                    continue;
                }

                // dual_input_dim = orig_out_dim (dimension at orig_out_deg)
                // dual_output_dim = orig_in_dim (dimension at orig_in_deg)

                for op_idx in 0..algebra.dimension(op_deg) {
                    // Transpose of M's action matrix:
                    // M's action: orig_in_dim inputs -> orig_out_dim outputs
                    // Transpose: orig_out_dim inputs -> orig_in_dim outputs
                    // = dual_input_dim inputs -> dual_output_dim outputs
                    for dual_input_idx in 0..dual_input_dim {
                        let mut output = vec![0u32; dual_output_dim];
                        // dual_input_idx corresponds to row dual_input_idx of the transpose
                        // = column dual_input_idx of the original action
                        for dual_output_idx in 0..dual_output_dim {
                            // dual_output_idx corresponds to column dual_output_idx of transpose
                            // = row dual_output_idx of original
                            // Original: action(op_deg, op_idx, orig_in_deg, dual_output_idx)
                            //   entry at dual_input_idx
                            let orig_action =
                                self.action(op_deg, op_idx, orig_in_deg, dual_output_idx);
                            output[dual_output_idx] = orig_action.entry(dual_input_idx);
                        }
                        dual.set_action(
                            op_deg,
                            op_idx,
                            dual_input,
                            dual_input_idx,
                            &output,
                        );
                    }
                }
            }
        }

        dual
    }
}
