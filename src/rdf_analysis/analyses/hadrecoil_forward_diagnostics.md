# Forward recoil diagnostics

Run in the existing ROOT/Python analysis environment:

```bash
python -m rdf_analysis.analyses.hadrecoil_study_highmu_forward --input /path/to/input.root --output hadronic_recoil_highmu_forward.root --min-avg-mu 40
python -m unittest rdf_analysis.analyses.test_hadrecoil_forward_diagnostics -v
```

The existing CLI is unchanged. The truth mass diagnostics additionally require
`tlep_bare_{pt,phi,pdgId}` and `tneutrino_{pt,phi,pdgId}`. Existing selections and user changes
are preserved. `met_inputs_common.py` was not changed by this extension.

## Exact new histogram naming scheme

For each scenario `{s}`, the following names are created (brace alternatives
expand literally):

- `h_mean_{balance,u_perp}_vs_{qT,avgMu,nPrimVtx,deltaPhi_W_mu}_{s}`
- `h_mean_balance_vs_{forward_sum_pt_em,n_forward_clusters_em,n_forward_jets}_{s}`
- `h_mean_abs_balance_vs_forward_sum_pt_em_{s}`
- `h_mean_paired_delta_abs_balance_vs_{nPrimVtx,qT}_{s}`
- `h_mean_paired_delta_abs_u_perp_vs_nPrimVtx_{s}`

Each of these 15 mean histograms has two companion histograms, obtained by
replacing `h_mean_` with `h_mean_entries_` and `h_mean_valid_`.
The former records finite paired x/y entries; the latter is 1 for a valid mean.

Two one-bin summaries per scenario:

- `h_angular_flatness_u_perp_{s}`
- `h_angular_offset_u_perp_{s}`

Three shared histograms:

- `h_deltaPhi_W_mu`
- `h_angular_reference_entries`
- `h_angular_flatness_valid`

Scenarios are `standard_met`, `tree_{truth,em,lcw,ml,pfoEM}` and each of
`central_clusters_{c}`, `all_clusters_{c}`, `keep_jets_forward_clusters_{c}`,
`replace_forward_jets_with_all_forward_clusters_{c}`, where
`{c}` is `em`, `lcw`, `ml`, `ml_forward`, or `truth`. This adds 47 histograms
per scenario plus 3 shared histograms, with no new MET-resolution histograms.

## Definitions and interpretation

Means require at least 10 finite events; errors are sample standard deviation
(ddof=1) divided by sqrt(N). Invalid estimates and errors remain NaN rather than
being converted to zeros by the shared histogram utility. Plot with validity
masks. The final upper edge belongs to the final bin. Profiles exclude values
outside the listed bin edges; entries reveal the supported range.

Mean balance and mean perpendicular recoil closer to zero indicate smaller
systematic biases. Mean absolute balance closer to zero indicates smaller
absolute event errors; it is not a resolution width. Negative paired changes
indicate smaller absolute errors than `standard_met` on the same events. The
baseline's paired curves are identically zero where valid.

The signed angle is atan2(sin(phi_W - phi_leading_mu), cos(...)), in radians,
with 24 bins from -pi to pi. Flatness uses common finite-event angular counts
and common valid bins across all scenarios. At least two angular bins are
required. Smaller flatness indicates less angular variation, while the offset
must separately be near zero. Flatness is not resolution and is not corrected
for statistical fluctuations; no uncertainty is assigned to this scalar.

Forward activity uses EM energy in GeV, pT=E/cosh(eta), nonnegative finite
energies/directions, the configured inclusive energy or pT threshold and
eta_min <= |eta| < eta_max. It is measured before lepton-cone removal and
underlying-event correction, independently of reconstruction scenario.
Selected forward-jet multiplicity reuses the existing unweighted jet-pT cut.

Stored-recoil projections now use sin/cos of stored W phi directly to avoid
0/0 at qT=0. The W direction at exactly zero pT is conventional. Existing ML
scale choices are preserved: `all_clusters_ml` combines central fCluster_MLE
(MeV converted to GeV) with cluster_e_ML_forward (already GeV). Other `ml`
subsets use fCluster_MLE; `ml_forward` subsets use cluster_e_ML_forward.

## Truth transverse mass

The reference is computed directly from `tlep_bare_pt`, `tlep_bare_phi`,
`tlep_bare_pdgId`, `tneutrino_pt`, `tneutrino_phi`, and `tneutrino_pdgId`.
Both pT values are assumed to be in MeV, like `tboson_pt`, and converted to GeV.
The massless transverse-mass formula uses no reconstructed quantities or `tu`
MET component. Bare muons are chosen to match the muon-only reconstructed
observable; born and dressed lepton definitions are not mixed into this reference.

Scalar and singleton entries are accepted. Each must contain a finite,
nonnegative pT and finite phi, with muon/neutrino PDG IDs of absolute values
13/14 and opposite signs. Empty, multiple, malformed or wrong-flavour event
records produce an invalid reference (NaN). They remain in the existing recoil
analysis but are omitted from mass comparisons. The cutflow records
`valid_truth_mT` and `invalid_truth_mT` among selected events. Missing required
branches fail explicitly instead of falling back to a hybrid reference.

The named branches are assumed to describe the same W decay. Their parent
association and MeV units cannot be established from the screenshot alone and
should be checked against the ntuple producer. No extra reconstructed-to-truth
matching cut is imposed: the existing leading reconstructed muon is used for
all scenario masses, including any misidentification effects. The existing
`h_mT_tree_truth` remains a scenario using reconstructed leptons; the new
`h_mT_truth` is the independent full truth reference.

### Exact additional mass histogram names

Shared: `h_mT_truth`.
For every scenario `{s}`:

- `h_mT_residual_{s}`
- `h_mean_mT_residual_vs_{qT,nPrimVtx}_{s}`
- `h_sigma_mT_residual_vs_{qT,nPrimVtx}_{s}`
- `h_mean_paired_delta_abs_mT_residual_vs_{qT,nPrimVtx}_{s}`
- `h_mT_truth_vs_reco_counts_{s}`
- `h_mT_truth_vs_reco_normalized_{s}`
- `h_mT_migration_truth_entries_{s}`
- `h_mT_migration_truth_valid_{s}`

Every mass mean has the same `h_mean_entries_` and `h_mean_valid_` companions
as the recoil means. This adds 19 histograms per scenario plus one shared
truth distribution. Widths use the existing utility's (q84-q16)/2 with at least
10 finite events; the residual mean's validity/entries companions apply to
these widths as well. Mean residual near zero indicates small bias; smaller
IQR width means a narrower residual core; negative paired changes mean lower
absolute mass error than the baseline on the same event.

Migration X is truth, Y is reconstruction, with 5 GeV bins from 0 to 200 GeV.
Each truth row is normalized over all reconstructed cells including ROOT
underflow/overflow; the final edge is overflow, following ROOT conventions.
Raw counts retain all finite pairs. Empty normalized rows remain NaN and are
flagged in the row-validity histogram, including flow bins. Nonempty rows sum
to one including flows; visible cells alone need not. Raw count errors are
sqrt(N); normalized cell errors are marginal binomial errors. Cell correlations
are not stored. Migration normalization has no minimum-count threshold beyond
a nonempty row; inspect the row counts before interpreting sparse rows.

## Validation limits

Nine synthetic tests cover recoil diagnostics plus truth-mass units, scalar and
singleton branches, both W charge assignments, invalid/ambiguous truth records,
missing branches, periodic azimuths, zero pT, migration flows and empty rows,
IQR widths, mean biases, paired signs, and histogram production. Histogram
production uses doubles: real ROOT serialization and event processing on
actual ntuples still require validation in the configured analysis environment.
