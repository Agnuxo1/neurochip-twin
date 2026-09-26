"""DoseCompass: next-concentration selection by expected information about potency (R6)."""
from .eig import (DEFAULT_ELL, DEFAULT_SAMPLES, DEFAULT_TAU, PlanStep, Posterior, bmc_batch,  # noqa: F401
                  eig_of_candidates, full_information_potency, load_fold_models, log_score_bits,
                  mask_divs, member_predictions, mutual_info_bits, plan_step, posterior_from_predictions,
                  potency_codes, recommend_next, reference_grid, sequential_design)
