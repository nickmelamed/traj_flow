# TrajFlow — Metrics Comparison

Every model / eval-split / difficulty-filter combination run so far, logged honestly
(including underperforming results — nothing is rounded or omitted to look better).
See `data/SCHEMA.md` for column definitions and `CLAUDE.md` for the phase plan.

| Phase | Model | Eval Split | Difficulty | N | minADE (m) | minFDE (m) | Miss Rate @2m | Notes |
|---|---|---|---|---|---|---|---|---|
| 2 | Constant Velocity | test | all | 1626 | 0.5109 | 1.1012 | 0.0769 |  |
| 2 | Constant Velocity | test | easy | 747 | 0.4231 | 0.9312 | 0.0535 |  |
| 2 | Constant Velocity | test | hard | 879 | 0.5855 | 1.2457 | 0.0967 |  |
| 2 | XGBoost | test | all | 1626 | 1.0038 | 2.0428 | 0.1771 | still underperforms CV, but the gap narrowed substantially (was minADE 1.656 on test/all) after removing a leaked absolute-heading feature that doesn't generalize across scenes; remaining gap likely still trees underestimating displacement for higher-speed agents; see README limitations |
| 2 | XGBoost | test | easy | 747 | 1.1245 | 2.2400 | 0.1968 | still underperforms CV, but the gap narrowed substantially (was minADE 1.656 on test/all) after removing a leaked absolute-heading feature that doesn't generalize across scenes; remaining gap likely still trees underestimating displacement for higher-speed agents; see README limitations |
| 2 | XGBoost | test | hard | 879 | 0.9013 | 1.8753 | 0.1604 | still underperforms CV, but the gap narrowed substantially (was minADE 1.656 on test/all) after removing a leaked absolute-heading feature that doesn't generalize across scenes; remaining gap likely still trees underestimating displacement for higher-speed agents; see README limitations |
| 3 | Transformer (pretrained, easy-only) | val | all | 701 | 2.2968 | 5.0240 | 0.4066 | model selection metric (best checkpoint by val minADE) |
| 3 | Transformer (pretrained, easy-only) | test | all | 1626 | 0.7584 | 1.3749 | 0.1052 | trained on easy scenes only; test/hard measures out-of-distribution generalization pre-fine-tune. Note: removing the leaked absolute-heading feature (see XGBoost row) made this model's test metrics slightly worse, not better (test/all minADE was 0.602 with heading included) -- unlike XGBoost, the transformer apparently extracted some real (if likely non-generalizable) signal from it; kept removed for methodological consistency/frame-invariance, see README limitations |
| 3 | Transformer (pretrained, easy-only) | test | easy | 747 | 0.7123 | 1.2818 | 0.0977 | trained on easy scenes only; test/hard measures out-of-distribution generalization pre-fine-tune. Note: removing the leaked absolute-heading feature (see XGBoost row) made this model's test metrics slightly worse, not better (test/all minADE was 0.602 with heading included) -- unlike XGBoost, the transformer apparently extracted some real (if likely non-generalizable) signal from it; kept removed for methodological consistency/frame-invariance, see README limitations |
| 3 | Transformer (pretrained, easy-only) | test | hard | 879 | 0.7976 | 1.4540 | 0.1115 | trained on easy scenes only; test/hard measures out-of-distribution generalization pre-fine-tune. Note: removing the leaked absolute-heading feature (see XGBoost row) made this model's test metrics slightly worse, not better (test/all minADE was 0.602 with heading included) -- unlike XGBoost, the transformer apparently extracted some real (if likely non-generalizable) signal from it; kept removed for methodological consistency/frame-invariance, see README limitations |
| 4 | Transformer (fine-tuned-v1, hard) | val | all | 701 | 2.1638 | 4.9005 | 0.4137 | model selection metric (best checkpoint by val minADE, starting from Phase 3 pretrained weights) |
| 4 | Transformer (fine-tuned-v1, hard) | test | all | 1626 | 0.9249 | 1.7757 | 0.0824 |  |
| 4 | Transformer (fine-tuned-v1, hard) | test | easy | 747 | 0.9587 | 1.8533 | 0.0776 |  |
| 4 | Transformer (fine-tuned-v1, hard) | test | hard | 879 | 0.8961 | 1.7097 | 0.0865 | primary Phase 4 comparison metric (pretrained vs fine-tuned) -- fine-tuning improved val minADE but regressed test/hard minADE (0.798 -> 0.896) while improving Miss Rate@2m (0.112 -> 0.086, i.e. fewer complete misses but higher average error); with only 6 train scenes total this reads as scene-specific overfitting rather than a clean win; see README limitations |
