# RDFAnalysisFramework

This repository now has a modular backbone under `src/rdf_analysis/`:

- `core/`: shared `RDataFrame` processor utilities
- `physics/`: recoil and kinematic helpers
- `stats/`: fit and histogram helpers
- `analyses/`: analysis-specific workflows and orchestration

`src/` now contains only package code. The analysis modules are directly runnable:

```bash
python3 src/rdf_analysis/analyses/hadrecoil_threshold_scan.py --input input.root --tree myTree --output threshold_scan_results.npz --nEvents 10000
python3 src/rdf_analysis/analyses/hadrecoil_response_study.py --input input.root --tree myTree --output hadrecoil_study_2.root --nEvents 10000
python3 src/rdf_analysis/analyses/hadrecoil_study_eta_region_validation.py --input input.root --tree myTree --output hadrecoil_study_3.root --nEvents 10000
python3 src/rdf_analysis/analyses/jet.py --input input.root --tree myTree --output jet.root --nEvents 10000
```
