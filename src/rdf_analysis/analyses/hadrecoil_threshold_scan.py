#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.hadrecoil_common import (
    NtupleProcessorRDF_hadrecoil,
    build_threshold_scan_observables,
)


def study_threshold_scans(proc, threshold_values):
    df_qT = proc.to_pandas(["qT"])["qT"].to_numpy()

    df_lep_pt = proc.to_pandas(["lep_pt"])
    df_lep_eta = proc.to_pandas(["lep_eta"])
    df_lep_phi = proc.to_pandas(["lep_phi"])

    df_clus_e_em = proc.to_pandas(["clus_e_em"])
    df_clus_e_lcw = proc.to_pandas(["clus_e_lcw"])
    df_clus_e_ml = proc.to_pandas(["clus_e_ml"])
    df_clus_eta = proc.to_pandas(["clus_eta"])
    df_clus_phi = proc.to_pandas(["clus_phi"])

    df_qx = proc.to_pandas(["q_x"])["q_x"].to_numpy()
    df_qy = proc.to_pandas(["q_y"])["q_y"].to_numpy()
    df_qT_safe = np.maximum(df_qT, 1e-9)

    threshold_results = {}

    for calib_name in ["em", "lcw", "ml"]:
        for thr in threshold_values:
            threshold_results[(calib_name, thr)] = {"qT": [], "u_perp": [], "u_par_plus_qT": []}

    total_events = len(df_qT)

    for event in range(total_events):
        if event % 10000 == 0:
            print(f"[Info]:: Threshold scan event {event}/{total_events}")

        lep_pt = np.array(df_lep_pt.iloc[event]["lep_pt"])
        lep_eta = np.array(df_lep_eta.iloc[event]["lep_eta"])
        lep_phi = np.array(df_lep_phi.iloc[event]["lep_phi"])

        clus_eta = np.array(df_clus_eta.iloc[event]["clus_eta"])
        clus_phi = np.array(df_clus_phi.iloc[event]["clus_phi"])

        qx = df_qx[event]
        qy = df_qy[event]
        qT = df_qT_safe[event]

        for calib_name, df_clus_e in [("em", df_clus_e_em), ("lcw", df_clus_e_lcw), ("ml", df_clus_e_ml)]:
            clus_e = np.array(df_clus_e.iloc[event][f"clus_e_{calib_name}"])
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the hadrecoil threshold-scan study.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="threshold_scan_results.npz", help="Output NPZ path.")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to process.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 5.0],
        help="Cluster-energy thresholds in GeV.",
    )
    return parser.parse_args()


def main(args):
    proc = NtupleProcessorRDF_hadrecoil(args.input, args.tree, args.nEvents)
    proc.build_dataframe()
    results = study_threshold_scans(proc, args.thresholds)
    save_threshold_results(args.output, results)
    return results


if __name__ == "__main__":
    main(parse_args())

