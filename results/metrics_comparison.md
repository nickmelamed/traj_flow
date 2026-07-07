# TrajFlow — Metrics Comparison

Every model / eval-split / difficulty-filter combination run so far, logged honestly
(including underperforming results — nothing is rounded or omitted to look better).
See `data/SCHEMA.md` for column definitions and `CLAUDE.md` for the phase plan.

| Phase | Model | Eval Split | Difficulty | N | minADE (m) | minFDE (m) | Miss Rate @2m | Notes |
|---|---|---|---|---|---|---|---|---|
| 2 | Constant Velocity | test | all | 1626 | 0.5109 | 1.1012 | 0.0769 |  |
| 2 | Constant Velocity | test | easy | 747 | 0.4231 | 0.9312 | 0.0535 |  |
| 2 | Constant Velocity | test | hard | 879 | 0.5855 | 1.2457 | 0.0967 |  |
| 2 | XGBoost | test | all | 1626 | 1.6559 | 3.6880 | 0.6328 | underperforms CV: trees underestimate displacement for higher-speed agents (can't extrapolate past training-range leaf values); see README limitations |
| 2 | XGBoost | test | easy | 747 | 1.6367 | 3.5358 | 0.5355 | underperforms CV: trees underestimate displacement for higher-speed agents (can't extrapolate past training-range leaf values); see README limitations |
| 2 | XGBoost | test | hard | 879 | 1.6722 | 3.8173 | 0.7156 | underperforms CV: trees underestimate displacement for higher-speed agents (can't extrapolate past training-range leaf values); see README limitations |
| 3 | Transformer (pretrained, easy-only) | val | all | 701 | 2.2892 | 5.1209 | 0.3937 | model selection metric (best checkpoint by val minADE) |
| 3 | Transformer (pretrained, easy-only) | test | all | 1626 | 0.6016 | 1.0856 | 0.0726 | trained on easy scenes only; test/hard measures out-of-distribution generalization pre-fine-tune |
| 3 | Transformer (pretrained, easy-only) | test | easy | 747 | 0.5078 | 0.9379 | 0.0482 | trained on easy scenes only; test/hard measures out-of-distribution generalization pre-fine-tune |
| 3 | Transformer (pretrained, easy-only) | test | hard | 879 | 0.6813 | 1.2110 | 0.0933 | trained on easy scenes only; test/hard measures out-of-distribution generalization pre-fine-tune |
| 4 | Transformer (fine-tuned-v1, hard) | val | all | 701 | 2.1751 | 4.8478 | 0.4080 | model selection metric (best checkpoint by val minADE, starting from Phase 3 pretrained weights) |
| 4 | Transformer (fine-tuned-v1, hard) | test | all | 1626 | 0.6814 | 1.2796 | 0.0689 |  |
| 4 | Transformer (fine-tuned-v1, hard) | test | easy | 747 | 0.6613 | 1.2552 | 0.0616 |  |
| 4 | Transformer (fine-tuned-v1, hard) | test | hard | 879 | 0.6985 | 1.3004 | 0.0751 | primary Phase 4 comparison metric (pretrained vs fine-tuned) -- with only 6 train scenes total, fine-tuning can improve val minADE while still regressing on test generalization (scene-specific overfitting); compare against the Phase 3 pretrained row above and see README limitations for the actual direction observed |
