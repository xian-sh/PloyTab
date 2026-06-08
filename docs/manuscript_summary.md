# Manuscript Summary

Manuscript title: **PolyTab: Structure-Informed Reconstruction of Sparse Polymer Property Profiles**

PolyTab addresses profile-level incompleteness in polymer property records. Instead of predicting a single property in isolation, it reconstructs a multi-property table from sparse per-polymer observations.

Core workflow:

1. Encode polymer PSMILES with a pretrained polymer language model.
2. Build a sparse property table using controlled retention settings.
3. Use quartile-interval classification to initialize missing entries with distribution-aware estimates.
4. Refine missing entries with stacked adaptive regression layers that combine structural embeddings, observed values, missing masks, and statistical priors.
5. Preserve observed experimental or simulated values while updating only missing positions.

Benchmarks compare PolyTab with Bayesian linear regression, extremely randomized trees, and TabPFN across heterogeneous polymer-property datasets. The manuscript evaluates reconstruction accuracy, correlation recovery, noise robustness, downstream modeling utility, and application cases in 6FDA-based gas-separation membranes and PP-based functional films.
