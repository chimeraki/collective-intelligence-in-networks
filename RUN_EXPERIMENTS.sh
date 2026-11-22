#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p rerun
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python src/check_theorem.py
python src/placement_causal_validation.py
python src/reviewer_validation.py
python src/reviewer_placement_budget.py
python src/reviewer_rewiring_dgd.py
python src/test_full.py
python src/rank_bound.py
python src/design_test.py
python src/higher_order_experiment.py
python src/higher_order_dyadic_control.py
python src/targeted_controls.py
python src/summarize_controls.py
python src/plot_theorem.py
python src/plot_higher_order.py
python src/new_figures.py
python verify_archives.py
