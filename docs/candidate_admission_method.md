# Candidate Admission Method

The research spine is candidate admission failure diagnosis and repair.

Implemented mechanisms include:

- route-aware channel gating
- side-index/postings fast paths
- conservative early exit
- dense-preserving safe fusion diagnostics
- parent-to-segment anchor selection diagnostics
- reranker-drop diagnostics

Optimization must not hardcode gold ids, reduce top-k globally, disable required
KnowMe/CloneMem channels, or remove dense preserve and safe fusion.
