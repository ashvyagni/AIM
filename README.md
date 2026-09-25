# AIM — Research Intelligence Model

The architectural baseline is the [complete research handbook](documents/AIM_Complete_Research_Architecture_and_Project_Handbook.pdf). The current engineering work is **Model-1**, in [model-1/](model-1/README.md).

## Start here

- [Model-1 setup and runnable commands](model-1/README.md)
- [Phase 1 implementation evidence](model-1/reports/phase-1-implementation.md)
- [Implementation directive and next-agent handoff](documents/AIM_Model_1_Implementation_Directive.md)
- [Current engineering decisions](model-1/docs/DECISIONS.md)
- [Phase 2A learned Researcher results](model-1/reports/phase-2a-implementation.md)
- [Phase 2A.1 arithmetic-supervision comparison](model-1/reports/phase-2a1-implementation.md)
- [Exact Researcher checkpoint continuation audit](model-1/reports/research-resume-audit.md)
- [Phase 2A.2 curriculum implementation and execution record](model-1/reports/phase-2a2-implementation.md)
- [Phase 2B Judge calibration and decision results](model-1/reports/phase-2b-implementation.md)

Latest completed experiment: Phase 2B trained twelve separate feature Judges and evaluated raw/scaled forecasts on 768 held-out candidate records. All 102 tests passed. Richer inputs improved mean familiar Brier score from .14612 to .11588, but one seed had negative decision utility under shift; the all-seed evidence gate failed. The default Judge remains verification-first. Next: shared measurement costs and explicit evidence-acquisition decisions.

Phase 2A.2 compared worked supervision with a staged arithmetic curriculum across six native Researcher models: mean verified test success 6.77% versus 6.25%, with both capability gates failed. Its 1,664 research-loop cases and observation audit remain available. The deterministic Researcher remains the default.

The earlier Phase 2A.1 plain/worked comparison remains available with its original results. Different world sets and coefficient ranges make its rates unsuitable as a direct comparison with Phase 2A.2.

Model-1 implements the modular Researcher → Judge → deterministic Controller → independent Verifiers → provenance Memory architecture. It contains a working numerical research-loop reference, a small transformer trained from random weights, separate preference/RLVR/Judge training mechanisms, evaluation fixtures, and local/distributed benchmark tools.

It is an engineering foundation. It is not a trained general research model, a validated VIT cluster deployment, or evidence of an industry breakthrough. Those remain experimental goals with explicit gates.

Existing research documents remain the historical baseline. Later owner requirements and prototype-specific decisions are documented in the new decision record; they are not silently written into the historical PDF. The existing empty `AIM Model 1/` folder is preserved; executable development uses `model-1/`.
