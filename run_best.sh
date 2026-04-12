#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  QCBM Capstone — Run best configuration (Stage 1 only)
#  Expected: ROC-AUC 0.9350  PR-AUC 0.5230  F1 0.6376  FAR 7.6%
# ─────────────────────────────────────────────────────────────────────────

echo ""
echo " QCBM Capstone - Best Configuration"
echo " ROC-AUC 0.9350  PR-AUC 0.5230  F1 0.6376  FAR 7.6%"
echo ""

python -u hierarchical_pipeline.py --config best_config.json --stage1-only

echo ""
echo " Done."
