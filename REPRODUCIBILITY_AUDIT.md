# TNSE final computational and bibliographic audit

**Audit date:** 22 November 2026. **Audited manuscript:** 11-page *Where Information Sits*, Sanjukta Krishnagopal, UCSB.

## Computational findings

- Reran the 96-graph, two-noise, four-placement, three-horizon primary experiment. **2,304/2,304 output rows identical to archived results**, including the primary 120-round means: random 0.1006774087, clustered 0.1580129833, interleaved 0.0939898659, graph-aware 0.0937563230.
- Reran the exact finite-time DGD transfer computation: **768/768 model-risk evaluations** and **576/576 paired contrasts** identical to archived tables. Direct DGD and transfer calculations agree at floating-point tolerance. A separate 60,000-draw small-system Monte Carlo test estimated risk 0.628247 against the analytic 0.627810.
- Reran the rank-observability bounds (1,920 rows) and 24-instance exact-risk placement design (72 rows); outputs match archived tables exactly. Risk-based search improved all 24 graph instances.
- Reran the higher-order group experiment (1,000 rows) and dyadic-control experiment (2,000 rows); both archived tables match exactly. Dyadic emulation matches the authorized group estimator.
- Reran 16-graph design control (320 rows) and reviewer-family/step-size/graph-order and rewiring experiments. **Fixed-proposal objective and MSE columns reproduce exactly**, while timing columns differ. The machine-dependent 0.16-second budget changes the number of proposals and some gains.
- Independently recomputed the leave-one-family-out scores: cross-specialty 0.399900, task-information 0.708059, combined 0.705426.
- Reran the earlier broad decentralized-collaboration benchmark (5,040 rows). This is supporting exploratory work, not the principal matched intervention.

## Gaps and scope

1. The 48-graph fresh-seed replication archive contains 1,152 rows and its reported 120-round means recompute correctly (random 0.105448, clustered 0.170587, interleaved 0.099105, graph-aware 0.095915), but **its original seed-generation script was not in the GitHub-ready archive**. That study is archive-verified only.
2. The historical main figures are provided as PDF artwork. Some original plotting scripts were not included in the release. Main tables and model outcomes are reproducible; the release is not yet a perfect figure-by-figure reproduction archive.
3. Python 3.13.5 and the available installed scientific dependencies were used. A new installation on an independent machine has not been performed. Wall-clock results will vary with CPU speed and scheduling.
4. These are computational reruns by the same assistant against the same model code, not independent confirmation of modeling assumptions or third-party replication.

## Math and inference

- Exact finite-time DGD decomposition follows from the stated affine recursion and first two moments; confirmed by direct recurrence and Monte Carlo testing.
- Finite-time rank lower bound requires rotational symmetry of the state distribution, beyond identity covariance. The manuscript already states this assumption.
- The 56.95% clustering penalty is the **ratio of pooled MSE means**, not the average of instance-level percentage penalties. The manuscript labels this distinction.
- 96 graph-data clusters (not 192 noise conditions or 2,304 rows) are the primary resampling units. The held-out-family predictor measures within-graph placement deviations.

## References

The audited source originally contained 39 bibliography entries but only 37 distinct cited keys. The two uncited entries (`shamaiah2010`, `nemhauser1978`) have been removed in this audited copy. All 37 remaining keys resolve, and title/author/venue/year data were checked against indexed publisher, proceedings, journal, or original-article records listed in `BIBLIOGRAPHY_AUDIT.csv`. This verifies bibliographic identity, not every contextual interpretation of a cited paper.

## Submission decision

**Main numerical findings are reproducible. Full release is not yet complete** until the fresh-seed generator is recovered or a new documented replication is run, a clean install is tested, and the figure-generation dependency chain is confirmed. The manuscript's use of the wall-clock control should remain explicitly exploratory.
