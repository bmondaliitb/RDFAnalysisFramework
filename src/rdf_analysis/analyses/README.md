# MET input validation and resolution study

The MET workflow is split into two independent executables:

- `validate_met_inputs.py` only validates the MET calculation against
  `METMaker`.
- `met_resolution_study.py` only performs the forward-jet resolution study.

Shared physics objects and formulas are kept in `met_inputs_common.py`.
Neither executable creates canvases, PDFs, or PNGs. All numerical
distributions and binned results are saved as histograms in ROOT files.

The MET performance definitions follow
[arXiv:2402.05858v2](./2402.05858v2.pdf).

## MET calculation

For each hard-object input term,

\[
\mathbf{p}^{\,\mathrm{vis}}_{\mathrm{T},t}
 = \sum_{i \in t}
   w_i p_{\mathrm{T},i}
   \left(\cos\phi_i,\sin\phi_i\right),
\qquad
t \in \{e,\mu,\mathrm{jet}\}.
\]

The `met_softtrk_mpx` and `met_softtrk_mpy` branches already carry the
missing-momentum sign. Therefore,

\[
\boxed{
\mathbf{p}^{\,\mathrm{miss}}_\mathrm{T}
 = -\left(
   \mathbf{p}^{\,\mathrm{vis}}_{\mathrm{T},e}
 + \mathbf{p}^{\,\mathrm{vis}}_{\mathrm{T},\mu}
 + \mathbf{p}^{\,\mathrm{vis}}_{\mathrm{T},\mathrm{jet}}
 \right)
 + \mathbf{p}^{\,\mathrm{miss,soft}}_\mathrm{T}
 }.
\]

Momentum and energy inputs are converted from MeV to GeV before summing.

## 1. METMaker validation

Run:

```bash
python validate_met_inputs.py \
  --input input.root \
  --tree treeAnaWZ \
  --output validate_met_inputs.root
```

The script compares the independently calculated vector with
`met_maker_mpx` and `met_maker_mpy`. The output ROOT file contains:

- `h_delta_mpx_maker`;
- `h_delta_mpy_maker`;
- `h_delta_et_maker`;
- `h_delta_phi_maker`.

Each histogram is calculated minus METMaker. Means, RMS values, maximum
absolute differences, and event counts are also printed to the terminal.

Required branches:

- `met_input_el_pt`, `met_input_el_phi`, `met_input_el_weight`;
- `met_input_mu_pt`, `met_input_mu_phi`, `met_input_mu_weight`;
- `met_input_jet_pt`, `met_input_jet_phi`, `met_input_jet_weight`;
- `met_softtrk_mpx`, `met_softtrk_mpy`;
- `met_maker_mpx`, `met_maker_mpy`.

## 2. Forward-jet MET resolution study

Run:

```bash
python met_resolution_study.py \
  --input input.root \
  --tree treeAnaWZ \
  --output met_resolution_study.root
```

The selected channel is \(Z\rightarrow\mu\mu\). By default,

\[
|m_{\mu\mu}-91.1876~\mathrm{GeV}| < 20~\mathrm{GeV},
\]

and a forward MET-input jet satisfies

\[
2.5 \leq |\eta_\mathrm{jet}| < 4.9.
\]

The following definitions from the reference paper are evaluated in
\(p_\mathrm{T}^{Z}\) bins:

\[
\hat{\mathbf A}^{Z}
 = \frac{\mathbf p_\mathrm{T}^{Z}}{p_\mathrm{T}^{Z}},
\qquad
P^{Z}
 = \mathbf p_\mathrm{T}^{\mathrm{miss}}\cdot\hat{\mathbf A}^{Z},
\]

\[
C^{Z}
 = 1 + \frac{\langle P^{Z}\rangle}
              {\langle p_\mathrm{T}^{Z}\rangle}.
\]

The truth-MET magnitude and azimuth are read from the `second` values of the
`tu_pt` and `tu_phi` map branches, respectively, using entry zero. Flattened
`tu_pt/tu_pt.second` and `tu_phi/tu_phi.second` branches are also accepted for
compatibility. If `tu_phi` is absent, the direction is derived from
`fMETTruthPt` and `fMETTruthPhi` when both truth-particle branches are
available. No reconstructed direction is substituted for a missing truth
direction.

The response-corrected \(Z\)-balance resolution is

\[
\frac{\mathrm{RMS}(P^Z)}{C^Z}
\quad\text{versus}\quad p_\mathrm{T}^Z.
\]

`numpy.std` is used for the RMS width, matching the ROOT histogram RMS
convention.

### Resolution scenarios

All scenarios use the same selected events containing at least one forward
MET-input jet.

1. `current`

   The validated MET calculation without modifications.

2. `keep_jets_forward_clusters_SCALE`

   The forward jets remain in the MET hard term. Clusters are added only
   when they are in the configured forward region and do not overlap any
   retained forward jet:

   \[
   \eta_\mathrm{min} \leq |\eta_c| < \eta_\mathrm{max},
   \qquad
   \Delta R(c,j) \geq R
   \quad\text{for every forward jet}.
   \]

   The resulting MET is

   \[
   \mathbf p_{\mathrm{T,keep\ jets}}^{\mathrm{miss},s}
   = \mathbf p_\mathrm{T}^\mathrm{miss}
   - \sum_{\substack{c\ \mathrm{forward}\\c\ \mathrm{nonoverlap}}}
     \frac{E_c^s}{\cosh\eta_c}
     (\cos\phi_c,\sin\phi_c).
   \]

   Each cluster is counted once. The EM, LCW, ML, ML-forward, and truth
   energy-scale versions are stored separately.

3. `clusters_away_SCALE`

   As an intermediate step, removing the forward jets adds their weighted
   visible momentum back to MET. Clusters satisfying
   \(\Delta R(\mathrm{cluster},j)\geq R\) for every removed jet are added to
   the visible sum:

   \[
   \mathbf p_{\mathrm{T,away}}^{\mathrm{miss},s}
   = \mathbf p_{\mathrm{T,no\ fwd\ jets}}^\mathrm{miss}
   - \sum_{c\ \mathrm{away}}
     \frac{E_c^s}{\cosh\eta_c}
     (\cos\phi_c,\sin\phi_c).
   \]

4. `all_clusters_SCALE`

   Forward jets are removed and every global cluster is added:

   \[
   \mathbf p_{\mathrm{T,all}}^{\mathrm{miss},s}
   = \mathbf p_{\mathrm{T,no\ fwd\ jets}}^\mathrm{miss}
   - \sum_{c\ \mathrm{all}}
     \frac{E_c^s}{\cosh\eta_c}
     (\cos\phi_c,\sin\phi_c).
   \]

`SCALE` is one of:

- `em`: `fCluster_rawE`;
- `lcw`: `fCluster_calE`;
- `ml`: `fCluster_MLE`;
- `ml_forward`: `cluster_e_ML_forward`;
- `truth`: `fCluster_truthE`.

The all-cluster case is an inclusive diagnostic. It does not reproduce
METMaker signal-ambiguity resolution and can double count signals already
represented by retained objects or the track soft term.

### Study ROOT output

For every scenario, the ROOT file contains:

- `h_met_SCENARIO`;
- `h_met_error_SCENARIO`;
- `h_pz_SCENARIO`;
- `h_pz_residual_SCENARIO`;
- `h2_pz_vs_pTZ_SCENARIO`;
- `h2_pz_residual_vs_pTZ_SCENARIO`;
- `h_entries_vs_pTZ_SCENARIO`;
- `h_mean_pz_vs_pTZ_SCENARIO`;
- `h_response_vs_pTZ_SCENARIO`;
- `h_raw_resolution_vs_pTZ_SCENARIO`;
- `h_resolution_vs_pTZ_SCENARIO`;
- `h_resolution_improvement_SCENARIO`.

The default \(p_\mathrm{T}^{Z}\) binning is

\[
[0,10,20,30,40,50,60,70,80,90,100,120,160,200,260]\ \mathrm{GeV}.
\]

The improvement is

\[
\sigma_\mathrm{current}-\sigma_\mathrm{scenario},
\]

so a positive value means improved resolution.

Additional options:

- `--input-dir`: recursively find ROOT inputs;
- `--nEvents`: process only the requested number of events;
- `--eta-min`, `--eta-max`: configure the forward region;
- `--cluster-radius`: configure the clusters-away separation;
- `--ptz-bins`: override the performance \(p_\mathrm{T}^{Z}\) bin edges with
  a comma-separated list in GeV;
- `--no-z-window`: disable the dilepton mass selection.

The resolution study additionally requires the `tu_pt` map. Momentum values
are expected in MeV and azimuthal values in radians. Truth-vector residuals
use the `second` value of `tu_phi`, with `fMETTruthPt` plus `fMETTruthPhi` as
a fallback direction source.
