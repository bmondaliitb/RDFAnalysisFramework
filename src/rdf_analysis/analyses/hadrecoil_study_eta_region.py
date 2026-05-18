#!/usr/bin/env python3
"""
Analysis flow: eta-region hadrecoil study
=========================================

This script runs a region-wise hadronic recoil study where regions are defined by
cluster |eta| (not lepton eta), and produces both spectrum histograms and
resolution-vs-qT summaries for each region.

Workflow in `study_3`:

1) Materialize event content from the RDataFrame wrapper
   - Lepton kinematics: `lep_pt`, `lep_eta`, `lep_phi`
   - Cluster inputs: `clus_e_truth`, `clus_e_em`, `clus_e_ml`, `clus_eta`, `clus_phi`
   - Truth-particle inputs: `met_truth_particle_pt/eta/phi/e`

2) Define eta regions (all in absolute eta)
   - `abs_eta_lt2p5`: [0.0, 2.5)
   - `abs_eta_2p5_3p2`: [2.5, 3.2)
   - `abs_eta_3p2_4p9`: [3.2, 4.9)
   - `abs_eta_0_4p9`:  [0.0, 4.9)

3) Loop over events and split objects by region
   - Build cluster-region masks from `abs(clus_eta)`
   - Build truth-particle-region masks from `abs(truth_eta)`
   - For each region, recompute hadrecoil from region-selected clusters for
     truth/EM/ML scales using `hadronic_recoil_from_clusters_python`
   - Recompute region-restricted truth-particle recoil proxy with
     `hadrecoil_from_truth_particles`

4) Accumulate per-region observables
   - Recoil pT spectra (`hr_pt`) for cluster truth/EM/ML and truth-particle proxy
   - SumET from energies (`compute_sum_et_from_energy`) for cluster truth/EM/ML
     and truth particles
   - Raw energy spectra for clusters and truth particles
   - Resolution ingredients:
     * `qT` from dilepton vector sum (z_x, z_y)
     * `u_perp` computed from region recoil components and qT axis
     * `u_pt` for normalization in qT bins

5) Build and write per-region histograms
   - Log-binned spectra for recoil pT, SumET, and energies
   - Resolution curves vs qT:
     * Fit sigma(u_perp) in each qT bin via Gaussian fits
     * Compute mean(uT) in same qT bins
     * Save `sigma(u_perp)/mean(uT)` for truth/EM/ML/truth-particle

6) Output layout
   - One ROOT directory per eta region
   - Each directory contains spectra (including `u_perp` distributions) plus:
     * `h_sigma_u_perp_truth`
     * `h_sigma_u_perp_em`
     * `h_sigma_u_perp_ml`
     * `h_sigma_u_perp_truth_particle`
"""

import argparse
import numpy as np
from pathlib import Path
import ROOT
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.hadrecoil_common import NtupleProcessorRDF_hadrecoil
from rdf_analysis.physics import compute_sum_et_from_energy, hadrecoil_from_truth_particles, hadronic_recoil_from_clusters_python
from rdf_analysis.stats import get_bins_log, make_numpy_hist, make_hist_from_bin_contents
from rdf_analysis.stats.fits import calculate_gaussian_fit_params_from_arrays, calculate_mean_sigma_in_x_bins


def study_3(proc, outfile):
    # Keep track of the current ROOT directory so per-region writes are scoped safely.
    prev_dir = ROOT.gDirectory

    # Materialize event-level inputs needed to recompute recoil observables in eta slices.
    df_lep_pt = proc.to_pandas(["lep_pt"])
    df_lep_eta = proc.to_pandas(["lep_eta"])
    df_lep_phi = proc.to_pandas(["lep_phi"])

    df_clus_e_truth = proc.to_pandas(["clus_e_truth"])
    df_clus_e_em = proc.to_pandas(["clus_e_em"])
    df_clus_e_ml = proc.to_pandas(["clus_e_ml"])
    df_clus_eta = proc.to_pandas(["clus_eta"])
    df_clus_phi = proc.to_pandas(["clus_phi"])

    df_truth_particle_pt = proc.to_pandas(["met_truth_particle_pt"])
    df_truth_particle_eta = proc.to_pandas(["met_truth_particle_eta"])
    df_truth_particle_phi = proc.to_pandas(["met_truth_particle_phi"])
    df_truth_particle_e = proc.to_pandas(["met_truth_particle_e"])

    eta_regions = [
        ("abs_eta_lt2p5", 0.0, 2.5),
        ("abs_eta_2p5_3p2", 2.5, 3.2),
        ("abs_eta_3p2_4p9", 3.2, 4.9),
        ("abs_eta_0_4p9", 0, 4.9),
    ]

    # Allocate per-region buffers for hadrecoil, sumET, and constituent-energy distributions.
    hadrecoil_pt_truth_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    hadrecoil_pt_em_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    hadrecoil_pt_ml_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    hadrecoil_pt_truth_particle_by_region = {region_name: [] for region_name, _, _ in eta_regions}

    sumet_truth_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    sumet_em_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    sumet_ml_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    sumet_truth_particle_by_region = {region_name: [] for region_name, _, _ in eta_regions}

    clus_energy_truth_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    clus_energy_em_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    clus_energy_ml_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    truth_particle_energy_by_region = {region_name: [] for region_name, _, _ in eta_regions}

    qT_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    u_perp_truth_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    u_perp_em_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    u_perp_ml_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    u_perp_truth_particle_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    u_pt_truth_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    u_pt_em_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    u_pt_ml_by_region = {region_name: [] for region_name, _, _ in eta_regions}
    u_pt_truth_particle_by_region = {region_name: [] for region_name, _, _ in eta_regions}

    total_events = df_lep_pt.shape[0]
    event_counter = 0

    # Event loop: split clusters/truth particles by |eta| region and recompute observables region-by-region.
    for event in range(total_events):
        if event_counter % 10000 == 0:
            print("[Info]:: Processing event {}/{}".format(event, total_events))

        event_lep_pt = np.array(df_lep_pt.iloc[event]["lep_pt"])
        event_lep_eta = np.array(df_lep_eta.iloc[event]["lep_eta"])
        event_lep_phi = np.array(df_lep_phi.iloc[event]["lep_phi"])

        event_clus_e_truth = np.array(df_clus_e_truth.iloc[event]["clus_e_truth"])
        event_clus_e_em = np.array(df_clus_e_em.iloc[event]["clus_e_em"])
        event_clus_e_ml = np.array(df_clus_e_ml.iloc[event]["clus_e_ml"])
        event_clus_eta = np.array(df_clus_eta.iloc[event]["clus_eta"])
        event_clus_phi = np.array(df_clus_phi.iloc[event]["clus_phi"])

        event_truth_particle_pt = np.array(df_truth_particle_pt.iloc[event]["met_truth_particle_pt"])
        event_truth_particle_eta = np.array(df_truth_particle_eta.iloc[event]["met_truth_particle_eta"])
        event_truth_particle_phi = np.array(df_truth_particle_phi.iloc[event]["met_truth_particle_phi"])
        event_truth_particle_e = np.array(df_truth_particle_e.iloc[event]["met_truth_particle_e"])

        z_x = np.sum(event_lep_pt * np.cos(event_lep_phi))
        z_y = np.sum(event_lep_pt * np.sin(event_lep_phi))
        qT = float(np.hypot(z_x, z_y))
        qT_denom = max(qT, 1e-9)

        abs_clus_eta = np.abs(event_clus_eta)
        abs_truth_eta = np.abs(event_truth_particle_eta)

        for region_name, eta_min, eta_max in eta_regions:
            # Region masks are defined on cluster eta (and truth-particle eta for truth MET proxy).
            region_clus_mask = (abs_clus_eta >= eta_min) & (abs_clus_eta < eta_max)
            region_truth_mask = (abs_truth_eta >= eta_min) & (abs_truth_eta < eta_max)

            region_clus_e_truth = event_clus_e_truth[region_clus_mask]
            region_clus_e_em = event_clus_e_em[region_clus_mask]
            region_clus_e_ml = event_clus_e_ml[region_clus_mask]
            region_clus_eta = event_clus_eta[region_clus_mask]
            region_clus_phi = event_clus_phi[region_clus_mask]

            region_truth_particle_pt = event_truth_particle_pt[region_truth_mask]
            region_truth_particle_eta = event_truth_particle_eta[region_truth_mask]
            region_truth_particle_phi = event_truth_particle_phi[region_truth_mask]
            region_truth_particle_e = event_truth_particle_e[region_truth_mask]

            hadrecoil_em_region = hadronic_recoil_from_clusters_python(
                event_lep_pt, event_lep_eta, event_lep_phi, region_clus_e_em, region_clus_eta, region_clus_phi
            )
            hadrecoil_ml_region = hadronic_recoil_from_clusters_python(
                event_lep_pt, event_lep_eta, event_lep_phi, region_clus_e_ml, region_clus_eta, region_clus_phi
            )
            hadrecoil_truth_region = hadronic_recoil_from_clusters_python(
                event_lep_pt, event_lep_eta, event_lep_phi, region_clus_e_truth, region_clus_eta, region_clus_phi
            )
            hadrecoil_truth_particle_region = hadrecoil_from_truth_particles(
                region_truth_particle_pt, region_truth_particle_eta, region_truth_particle_phi
            )

            # Store recoil-scale and response inputs used later for histogramming/fits.
            hadrecoil_pt_em_by_region[region_name].append(hadrecoil_em_region["hr_pt"])
            hadrecoil_pt_ml_by_region[region_name].append(hadrecoil_ml_region["hr_pt"])
            hadrecoil_pt_truth_by_region[region_name].append(hadrecoil_truth_region["hr_pt"])
            hadrecoil_pt_truth_particle_by_region[region_name].append(hadrecoil_truth_particle_region["met_pt"])

            qT_by_region[region_name].append(qT)
            u_perp_truth_by_region[region_name].append(
                (z_x * hadrecoil_truth_region["u_y"] - z_y * hadrecoil_truth_region["u_x"]) / qT_denom
            )
            u_perp_em_by_region[region_name].append(
                (z_x * hadrecoil_em_region["u_y"] - z_y * hadrecoil_em_region["u_x"]) / qT_denom
            )
            u_perp_ml_by_region[region_name].append(
                (z_x * hadrecoil_ml_region["u_y"] - z_y * hadrecoil_ml_region["u_x"]) / qT_denom
            )
            u_perp_truth_particle_by_region[region_name].append(
                (z_x * hadrecoil_truth_particle_region["u_y"] - z_y * hadrecoil_truth_particle_region["u_x"]) / qT_denom
            )
            u_pt_truth_by_region[region_name].append(hadrecoil_truth_region["u_pt"])
            u_pt_em_by_region[region_name].append(hadrecoil_em_region["u_pt"])
            u_pt_ml_by_region[region_name].append(hadrecoil_ml_region["u_pt"])
            u_pt_truth_particle_by_region[region_name].append(hadrecoil_truth_particle_region["u_pt"])

            # Store scalar activity observables and ingredient energy spectra for shape comparisons.
            sumet_truth_by_region[region_name].append(compute_sum_et_from_energy(region_clus_e_truth, region_clus_eta))
            sumet_em_by_region[region_name].append(compute_sum_et_from_energy(region_clus_e_em, region_clus_eta))
            sumet_ml_by_region[region_name].append(compute_sum_et_from_energy(region_clus_e_ml, region_clus_eta))
            sumet_truth_particle_by_region[region_name].append(
                compute_sum_et_from_energy(region_truth_particle_e, region_truth_particle_eta)
            )

            clus_energy_truth_by_region[region_name].extend(region_clus_e_truth)
            clus_energy_em_by_region[region_name].extend(region_clus_e_em)
            clus_energy_ml_by_region[region_name].extend(region_clus_e_ml)
            truth_particle_energy_by_region[region_name].extend(region_truth_particle_e)

        event_counter += 1

    bin_edges_recoil = get_bins_log(0.5, 150, 40)
    bin_edges_sumet = get_bins_log(0.5, 1000, 40)
    bin_edges_energy = get_bins_log(0.05, 150, 50)
    bin_edges_qt = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 250]
    n_bins_u_perp = 30
    u_perp_min = -30.0
    u_perp_max = 30.0

    # Write one output directory per eta region: spectra first, then resolution-vs-qT summaries.
    for region_name, eta_min, eta_max in eta_regions:
        prev_dir.cd()
        out_dir = outfile.Get(region_name)
        if not out_dir:
            out_dir = outfile.mkdir(region_name)
        out_dir.cd()

        hists = [
            make_numpy_hist("h_recoil_pt_cluster_truth", "", bin_edges_recoil, hadrecoil_pt_truth_by_region[region_name]),
            make_numpy_hist("h_recoil_pt_cluster_em", "", bin_edges_recoil, hadrecoil_pt_em_by_region[region_name]),
            make_numpy_hist("h_recoil_pt_cluster_ml", "", bin_edges_recoil, hadrecoil_pt_ml_by_region[region_name]),
            make_numpy_hist("h_recoil_pt_truth_particle", "", bin_edges_recoil, hadrecoil_pt_truth_particle_by_region[region_name]),
            make_numpy_hist("h_u_perp_truth", "", n_bins_u_perp, u_perp_min, u_perp_max, u_perp_truth_by_region[region_name]),
            make_numpy_hist("h_u_perp_em", "", n_bins_u_perp, u_perp_min, u_perp_max, u_perp_em_by_region[region_name]),
            make_numpy_hist("h_u_perp_ml", "", n_bins_u_perp, u_perp_min, u_perp_max, u_perp_ml_by_region[region_name]),
            make_numpy_hist("h_u_perp_truth_particle", "", n_bins_u_perp, u_perp_min, u_perp_max, u_perp_truth_particle_by_region[region_name]),
            make_numpy_hist("h_sumet_cluster_truth", "", bin_edges_sumet, sumet_truth_by_region[region_name]),
            make_numpy_hist("h_sumet_cluster_em", "", bin_edges_sumet, sumet_em_by_region[region_name]),
            make_numpy_hist("h_sumet_cluster_ml", "", bin_edges_sumet, sumet_ml_by_region[region_name]),
            make_numpy_hist("h_sumet_truth_particle", "", bin_edges_sumet, sumet_truth_particle_by_region[region_name]),
            make_numpy_hist("h_clus_energy_em", "", bin_edges_energy, clus_energy_em_by_region[region_name]),
            make_numpy_hist("h_clus_energy_ml", "", bin_edges_energy, clus_energy_ml_by_region[region_name]),
            make_numpy_hist("h_clus_energy_truth", "", bin_edges_energy, clus_energy_truth_by_region[region_name]),
            make_numpy_hist("h_clus_energy_truth_particle", "", bin_edges_energy, truth_particle_energy_by_region[region_name]),
        ]

        for hist in hists:
            hist.Write()
            hist.Delete()

        # Build arrays for this region and extract u_perp resolution normalized by mean uT.
        region_qT = np.array(qT_by_region[region_name], dtype=float)
        region_u_perp_truth = np.array(u_perp_truth_by_region[region_name], dtype=float)
        region_u_perp_em = np.array(u_perp_em_by_region[region_name], dtype=float)
        region_u_perp_ml = np.array(u_perp_ml_by_region[region_name], dtype=float)
        region_u_perp_truth_particle = np.array(u_perp_truth_particle_by_region[region_name], dtype=float)
        region_u_pt_truth = np.array(u_pt_truth_by_region[region_name], dtype=float)
        region_u_pt_em = np.array(u_pt_em_by_region[region_name], dtype=float)
        region_u_pt_ml = np.array(u_pt_ml_by_region[region_name], dtype=float)
        region_u_pt_truth_particle = np.array(u_pt_truth_particle_by_region[region_name], dtype=float)

        if len(region_qT) > 0:
            # Sigma(u_perp) from Gaussian fits in qT bins.
            _, sigma_truth_uperp = calculate_gaussian_fit_params_from_arrays(
                region_qT, region_u_perp_truth, bin_edges_qt, y_name=f"u_perp_truth_{region_name}"
            )
            _, sigma_em_uperp = calculate_gaussian_fit_params_from_arrays(
                region_qT, region_u_perp_em, bin_edges_qt, y_name=f"u_perp_em_{region_name}"
            )
            _, sigma_ml_uperp = calculate_gaussian_fit_params_from_arrays(
                region_qT, region_u_perp_ml, bin_edges_qt, y_name=f"u_perp_ml_{region_name}"
            )
            _, sigma_truth_particle_uperp = calculate_gaussian_fit_params_from_arrays(
                region_qT, region_u_perp_truth_particle, bin_edges_qt, y_name=f"u_perp_truth_particle_{region_name}"
            )

            # Mean(uT) in the same qT bins for normalization.
            mean_truth_uT, _ = calculate_mean_sigma_in_x_bins(region_qT, region_u_pt_truth, bin_edges_qt)
            mean_em_uT, _ = calculate_mean_sigma_in_x_bins(region_qT, region_u_pt_em, bin_edges_qt)
            mean_ml_uT, _ = calculate_mean_sigma_in_x_bins(region_qT, region_u_pt_ml, bin_edges_qt)
            mean_truth_particle_uT, _ = calculate_mean_sigma_in_x_bins(region_qT, region_u_pt_truth_particle, bin_edges_qt)

            # Persist the normalized resolution curves sigma(u_perp)/<uT> for truth, EM and ML.
            make_hist_from_bin_contents(
                "h_sigma_u_perp_truth", "Sigma u_perp_truth;qT;Sigma u_perp_truth/mean(uT)",
                bin_edges_qt, sigma_truth_uperp / mean_truth_uT
            ).Write()
            make_hist_from_bin_contents(
                "h_sigma_u_perp_em", "Sigma u_perp_em;qT;Sigma u_perp_em/mean(uT)",
                bin_edges_qt, sigma_em_uperp / mean_em_uT
            ).Write()
            make_hist_from_bin_contents(
                "h_sigma_u_perp_ml", "Sigma u_perp_ml;qT;Sigma u_perp_ml/mean(uT)",
                bin_edges_qt, sigma_ml_uperp / mean_ml_uT
            ).Write()
            make_hist_from_bin_contents(
                "h_sigma_u_perp_truth_particle", "Sigma u_perp_truth_particle;qT;Sigma u_perp_truth_particle/mean(uT)",
                bin_edges_qt, sigma_truth_particle_uperp / mean_truth_particle_uT
            ).Write()


def parse_args():
    parser = argparse.ArgumentParser(description="Run hadrecoil study 3.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="hadrecoil_study_eta_region.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to process.")
    return parser.parse_args()


def main(args):
    proc = NtupleProcessorRDF_hadrecoil(args.input, args.tree, args.nEvents)
    proc.build_dataframe()
    out_file = ROOT.TFile(args.output, "RECREATE")
    study_3(proc, out_file)
    out_file.Close()
    return proc


if __name__ == "__main__":
    main(parse_args())

