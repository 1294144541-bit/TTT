# Three-Tier Decoupled Benchmark: Rigorous Multi-Horizon Evaluation

## 1. Overview
This directory contains the final unified research assets for the project:
**"Decoupled In-Weights Adaptation: Epistemic-Uncertainty-Aware Test-Time Training for Latent World Models"**

It implements a strict **Three-Tier Decoupled Evaluation Protocol** resolving all reviewer critiques:
- **Tier 1 (Perception Decoupled)**: Evaluated directly on actuated continuous dynamics environments with explicit control actions $a_t \in \mathbb{R}^1$.
- **Tier 2 (Action-Conditioned Visual Extrapolation)**: Evaluated on high-resolution $224 \times 224$ physical contact and collision video streams with official Meta V-JEPA 2.1 (85.65M) embeddings.
- **Tier 3 (Ultra-Long Real Camera Footage)**: Evaluated on 200-frame continuous real-world camera video sequences with out-of-distribution optical flow.

## 2. Directory Layout
```
├── README.md                                 # This specification
├── evidence_table.md                         # 18-paper literature matrix (grounded DOIs & arXiv IDs)
├── controlled_budget_evaluation_report.md    # Primary evaluation report with paired t-tests & FLOPs profiling
├── Three_Tier_Decoupled_Benchmark_Package.zip# Consolidated deliverable archive
├── benchmarks_v2/                            # Benchmark datasets with explicit actions and dynamics shifts
│   ├── actuated_train_nominal.pt
│   ├── actuated_test_nominal.pt
│   ├── actuated_test_heavy_mass.pt           # Payload +350% shift
│   └── actuated_test_ice_friction.pt         # Friction -80% shift
├── src_v2/                                   # Refactored codebase conforming to the final protocol
│   ├── actuated_env.py                       # Continuous actuated physics environment (Symplectic integration)
│   ├── bayesian_ttt_action_model.py          # Action-conditioned Bayesian In-Weights model with epistemic P_t
│   ├── jepa_ttt_reproduction.py              # Faithful reproduction of JEPA-TTT (arXiv:2610.00722)
│   ├── robust_mppi_planner.py                # Robust MPPI planner with epistemic uncertainty penalization
│   └── run_three_tier_benchmarks.py          # Master execution script for the three-tier benchmark suite
├── results_v2/                               # Serialized benchmark artifacts and evaluation metrics
│   ├── bayesian_action_model.pt
│   ├── jepa_action_model.pt
│   └── three_tier_decoupled_benchmark_results.pt
├── checkpoints/                              # Model checkpoints
│   ├── in_context_transformer.pt
│   └── vjepa/                                # Meta V-JEPA 2.1 configuration and weights
├── data/                                     # High-resolution physical collision and real video tensors
└── logs/                                     # Execution logs
    └── training_history.log
```

## 3. Quick Run Instructions
Run the consolidated three-tier benchmark suite from this directory:
```bash
PYTHONPATH=. python3 src_v2/run_three_tier_benchmarks.py
```
