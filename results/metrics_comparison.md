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
