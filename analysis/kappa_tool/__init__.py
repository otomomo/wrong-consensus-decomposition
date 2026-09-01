"""kappa_tool — reusable κ-plurality decomposition package for agreement-ceiling.

Externally packaged form of the validated analysis behind the paper. It accepts
a per-sample long table (case_id, run_id, answer, is_correct|ground_truth) and
re-covers the full decomposition that kappa_decompose / kappa_rival_preference
produce from the Ding 2026 pre-aggregated tables. The math is identical and
import-shared with those scripts; this package adds the aggregation + a stable
library interface + a parity test against the committed results/*.json.

Parity contract (decompose.py docstring is normative):
 * rival family bit-exact vs results/kappa_rival_preference.json (mirrors
 kappa_rival_preference.main's rng sequence; requires the Ding stored A/C
 floats -- cli parity mode)
 * decompose-only MC family asserted within a documented tolerance vs
 results/kappa_decompose.json (independent re-draws)

Public API (see each module):
 load.load_samples(path, ...) -> canonical per-sample DataFrame
 load.aggregate_to_runs(df, ...) -> per-(case_id, run_id) rows (Ding-like)
 load.derive_ground_truth(df, ...) -> per-case gt from correct samples
 decompose.decompose(df, ...) -> cells with all κ fields + CIs
 decompose.decompose_runs(runs, ...) -> same, over aggregated per-run rows
 cli.main( -> CLI (per-sample mode / parity DB mode)

Conventions enforced here:
 * p = single-sample accuracy, NOT consensus accuracy
 * C = answer-option count; GPQA = 4, AIME via --aime-c-mode
 * i.i.d.-MC tie-break = np.argmax(counts) (smallest label)
 * headline plurality_share = κ_iid / κ_empirical
 * Bootstrap CIs B=10^4; case-clustered + coupled for ratios
"""

__version__ = "0.1.0"

__all__ = ["load", "decompose", "cli"]