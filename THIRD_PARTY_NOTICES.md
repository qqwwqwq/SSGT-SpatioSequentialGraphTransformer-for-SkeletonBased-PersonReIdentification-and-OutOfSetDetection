# Source provenance

The original working tree contained the TranSG implementation and its MIT license:

> Copyright (c) 2023 Kali-Hac

That license is reproduced without modification in `LICENSE`. The inherited model,
training routines, graph construction, and dataset utilities remain subject to that
notice. Source: https://github.com/Kali-Hac/TranSG

Reference publication: Haocong Rao and Chunyan Miao, “TranSG: Transformer-Based
Skeleton Graph Prototype Contrastive Learning With Structure-Trajectory Prompted
Reconstruction for Person Re-Identification,” CVPR 2023.

The SSGT changes in the supplied working tree include temporal attention,
configurable position encoding, adaptive loss weighting, and OOS evaluation.
The release refactor separates those routines and adds portable entry points,
documentation, and validation. It does not claim that the original TranSG authors
authored these later changes.

The directory organization and documentation conventions were informed by
https://github.com/qqwwqwq/DualSOM--SOFTWAREX- . No DualSOM model implementation,
dataset, or pretrained weights are included here.

Third-party libraries are installed separately and retain their respective
licenses. Dataset distribution terms are independent of the source-code license.
