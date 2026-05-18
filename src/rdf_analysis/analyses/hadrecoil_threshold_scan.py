#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import sys

import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.stats import make_hist_from_bin_contents, make_numpy_hist
from rdf_analysis.stats.fits import calculate_gaussian_fit_params_from_arrays, calculate_mean_sigma_in_x_bins
from rdf_analysis.analyses.hadrecoil_common import (
    NtupleProcessorRDF_hadrecoil,
    build_threshold_scan_observables,
    build_truth_particle_threshold_scan_observables,
)


def study_threshold_scans(proc, threshold_values):
    df_qT = proc.to_pandas(["qT"])["qT"].to_numpy()

    df_lep_pt = proc.to_pandas(["lep_pt"])
    df_lep_eta = proc.to_pandas(["lep_eta"])
    df_lep_phi = proc.to_pandas(["lep_phi"])

    df_truth_particle_pt = proc.to_pandas(["met_truth_particle_pt"])
    df_truth_particle_eta = proc.to_pandas(["met_truth_particle_eta"])
    df_truth_particle_phi = proc.to_pandas(["met_truth_particle_phi"])
    df_truth_particle_e = proc.to_pandas(["met_truth_particle_e"])

    df_clus_e_truth = proc.to_pandas(["clus_e_truth"])
    df_clus_e_em = proc.to_pandas(["clus_e_em"])
    df_clus_e_lcw = proc.to_pandas(["clus_e_lcw"])
    df_clus_e_ml = proc.to_pandas(["clus_e_ml"])
    df_clus_eta = proc.to_pandas(["clus_eta"])
    df_clus_phi = proc.to_pandas(["clus_phi"])

    df_qx = proc.to_pandas(["q_x"])["q_x"].to_numpy()
    df_qy = proc.to_pandas(["q_y"])["q_y"].to_numpy()
    df_qT_safe = np.maximum(df_qT, 1e-9)

    threshold_results = {}

    for calib_name in ["truth_particle", "cluster_e_truth", "em", "lcw", "ml"]:
        for thr in threshold_values:
            threshold_results[(calib_name, thr)] = {"qT": [], "uT": [], "u_perp": [], "u_par_plus_qT": []}

    total_events = len(df_qT)

    for event in range(total_events):
        if event % 10000 == 0:
            print(f"[Info]:: Threshold scan event {event}/{total_events}")

        lep_pt = np.array(df_lep_pt.iloc[event]["lep_pt"])
        lep_eta = np.array(df_lep_eta.iloc[event]["lep_eta"])
        lep_phi = np.array(df_lep_phi.iloc[event]["lep_phi"])

        truth_particle_pt = np.array(df_truth_particle_pt.iloc[event]["met_truth_particle_pt"])
        truth_particle_eta = np.array(df_truth_particle_eta.iloc[event]["met_truth_particle_eta"])
        truth_particle_phi = np.array(df_truth_particle_phi.iloc[event]["met_truth_particle_phi"])
        truth_particle_e = np.array(df_truth_particle_e.iloc[event]["met_truth_particle_e"])

        qx = df_qx[event]
        qy = df_qy[event]
        qT = df_qT_safe[event]

        truth_particle_scan_out = build_truth_particle_threshold_scan_observables(
            truth_particle_pt,
            truth_particle_eta,
            truth_particle_phi,
            truth_particle_e,
            threshold_values,
        )

        for thr in threshold_values:
            ux = truth_particle_scan_out[thr]["u_x"]
            uy = truth_particle_scan_out[thr]["u_y"]

            u_perp = (qx * uy - qy * ux) / qT
            u_par = (ux * qx + uy * qy) / qT
            u_par_plus_qT = u_par + df_qT[event]

            threshold_results[("truth_particle", thr)]["qT"].append(df_qT[event])
            threshold_results[("truth_particle", thr)]["uT"].append(truth_particle_scan_out[thr]["u_pt"])
            threshold_results[("truth_particle", thr)]["u_perp"].append(u_perp)
            threshold_results[("truth_particle", thr)]["u_par_plus_qT"].append(u_par_plus_qT)

        clus_eta = np.array(df_clus_eta.iloc[event]["clus_eta"])
        clus_phi = np.array(df_clus_phi.iloc[event]["clus_phi"])

        for calib_name, df_clus_e, clus_e_name in [
            ("cluster_e_truth", df_clus_e_truth, "clus_e_truth"),
            ("em", df_clus_e_em, "clus_e_em"),
            ("lcw", df_clus_e_lcw, "clus_e_lcw"),
            ("ml", df_clus_e_ml, "clus_e_ml"),
        ]:
            clus_e = np.array(df_clus_e.iloc[event][clus_e_name])
            scan_out = build_threshold_scan_observables(
                lep_pt,
                lep_eta,
                lep_phi,
                clus_e,
                clus_eta,
                clus_phi,
                threshold_values,
            )

            for thr in threshold_values:
                ux = scan_out[thr]["u_x"]
                uy = scan_out[thr]["u_y"]

                u_perp = (qx * uy - qy * ux) / qT
                u_par = (ux * qx + uy * qy) / qT
                u_par_plus_qT = u_par + df_qT[event]

                threshold_results[(calib_name, thr)]["qT"].append(df_qT[event])
                threshold_results[(calib_name, thr)]["uT"].append(scan_out[thr]["u_pt"])
                threshold_results[(calib_name, thr)]["u_perp"].append(u_perp)
                threshold_results[(calib_name, thr)]["u_par_plus_qT"].append(u_par_plus_qT)

    return threshold_results


def save_threshold_results(output_path, threshold_results):
    payload = {}
    for (calib_name, thr), observables in threshold_results.items():
        thr_label = str(thr).replace(".", "p")
        prefix = f"{calib_name}_thr_{thr_label}"
        for observable_name, values in observables.items():
            payload[f"{prefix}_{observable_name}"] = np.asarray(values, dtype=np.float64)

    np.savez_compressed(output_path, **payload)

def save_threshold_results_as_hists(results):
    # make root histograms

    qt_bin_edges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 250]

    # qT = vector pT sum of two leptons
    # this is not scale dependent. We only need one hist
    first_result = next(iter(results.values()))
    make_numpy_hist(f"h_qT",
                                f"qT distribution", 30, 20, 200, first_result["qT"]).Write()
    for (calib_name, thr), observables in results.items():
        thr_label = str(thr).replace(".", "p")
        make_numpy_hist(f"h_u_perp_{calib_name}_{thr_label}",
                        f"u_perp distribution for {calib_name} threshold {thr}",30, -30,30, observables["u_perp"]).Write()
        make_numpy_hist(f"h_uT_{calib_name}_{thr_label}",
                        f"uT distribution for {calib_name} threshold {thr}",30, 0, 100, observables["uT"]).Write()
        make_numpy_hist(f"h_u_par_plus_qT_{calib_name}_{thr_label}",
                        f"u_par_plus_qT distribution for {calib_name} threshold {thr}",30, -30, 50, observables["u_par_plus_qT"]).Write()

        qT = np.asarray(observables["qT"], dtype=np.float64)
        u_perp = np.asarray(observables["u_perp"], dtype=np.float64)
        uT = np.asarray(observables["uT"], dtype=np.float64)

        # Build the same resolution observable as response study: sigma(u_perp)/mean(uT) vs qT.
        _, sigma_u_perp = calculate_gaussian_fit_params_from_arrays(
            qT,
            u_perp,
            qt_bin_edges,
            y_name=f"u_perp_{calib_name}_{thr_label}",
        )
        mean_uT, _ = calculate_mean_sigma_in_x_bins(qT, uT, qt_bin_edges)

        resolution = np.divide(
            sigma_u_perp,
            mean_uT,
            out=np.full_like(sigma_u_perp, np.nan, dtype=np.float64),
            where=np.isfinite(mean_uT) & (mean_uT != 0.0),
        )

        make_hist_from_bin_contents(
            f"h_sigma_u_perp_{calib_name}_{thr_label}",
            f"Sigma u_perp/mean(uT) for {calib_name} threshold {thr};qT;Sigma u_perp/mean(uT)",
            qt_bin_edges,
            resolution,
        ).Write()

        # Keep an explicit MET-truth-particle naming alias for easier downstream lookup.
        if calib_name == "truth_particle":
            make_hist_from_bin_contents(
                f"h_sigma_u_perp_met_truth_particle_{thr_label}",
                f"Sigma u_perp/mean(uT) for MET truth particles threshold {thr};qT;Sigma u_perp/mean(uT)",
                qt_bin_edges,
                resolution,
            ).Write()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the hadrecoil threshold-scan study.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output_npz", default="threshold_scan_results.npz", help="Output NPZ path.")
    parser.add_argument("--output", default="threshold_scan_results.hists.root", help="Output root file saves histograms")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to process.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        help="Thresholds in GeV for absolute momentum scans of both clusters and truth particles.",
    )
    return parser.parse_args()


def main(args):

    proc = NtupleProcessorRDF_hadrecoil(args.input, args.tree, args.nEvents)
    proc.build_dataframe()
    results = study_threshold_scans(proc, args.thresholds)
    save_threshold_results(args.output_npz, results)

    out_file = ROOT.TFile.Open(args.output, "RECREATE")
    save_threshold_results_as_hists(results)

    out_file.Close()

    return results


if __name__ == "__main__":
    main(parse_args())
