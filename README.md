# AIM — Research Intelligence Model

The architectural baseline is the [complete research handbook](documents/AIM_Complete_Research_Architecture_and_Project_Handbook.pdf). The current engineering work is **Model-1**, in [model-1/](model-1/README.md).

## Start here

- [Model-1 setup and runnable commands](model-1/README.md)
- [Phase 1 implementation evidence](model-1/reports/phase-1-implementation.md)
- [Implementation directive and next-agent handoff](documents/AIM_Model_1_Implementation_Directive.md)
- [Current engineering decisions](model-1/docs/DECISIONS.md)

Model-1 implements the modular Researcher → Judge → deterministic Controller → independent Verifiers → provenance Memory architecture. It contains a working numerical research-loop reference, a small transformer trained from random weights, separate preference/RLVR/Judge training mechanisms, evaluation fixtures, and local/distributed benchmark tools.

It is an engineering foundation. It is not a trained general research model, a validated VIT cluster deployment, or evidence of an industry breakthrough. Those remain experimental goals with explicit gates.

Existing research documents remain the historical baseline. Later owner requirements and prototype-specific decisions are documented in the new decision record; they are not silently written into the historical PDF. The existing empty `AIM Model 1/` folder is preserved; executable development uses `model-1/`.
