#!/usr/bin/env python3
import argparse
from pathlib import Path
import glob
import ROOT
import sys
import awkward as ak
import logging

# Set the minimum log level to output (default is WARNING)
logging.basicConfig(level=logging.INFO)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.hadrecoil_common import HadRecoilCommon
from rdf_analysis.stats import make_hist_from_bin_contents
from rdf_analysis.stats.fits import (
    calculate_gaussian_fit_params_from_arrays,
    calculate_mean_sigma_in_x_bins,
)


def save_response_histograms(recoil, output_file):
    # pT(ll) bin edges in GeV.
    x_bin_edges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 250]
    out_file = ROOT.TFile.Open(output_file, "RECREATE")
    if not out_file or out_file.IsZombie():
        raise OSError(f"Could not create {output_file}")

    try:
        qT = recoil["qT"]
        # calculate mean and sigma of u_par_plus_qT
        mean_upar_truth, sigma_upar_truth = calculate_gaussian_fit_params_from_arrays(
            qT, recoil["u_par_plus_qT_truth"], x_bin_edges,
            out_file=out_file, y_name="u_par_plus_qT_truth",
        )
        mean_upar_em, sigma_upar_em = calculate_gaussian_fit_params_from_arrays(
            qT, recoil["u_par_plus_qT_em"], x_bin_edges,
            out_file=out_file, y_name="u_par_plus_qT_em",
        )
        mean_upar_ml, sigma_upar_ml = calculate_gaussian_fit_params_from_arrays(
            qT, recoil["u_par_plus_qT_ml"], x_bin_edges,
            out_file=out_file, y_name="u_par_plus_qT_ml",
        )

        # calculate mean and sigma of u_perp
        _, sigma_uperp_truth = calculate_gaussian_fit_params_from_arrays(
            qT, recoil["u_perp_truth"], x_bin_edges,
            out_file=out_file, y_name="u_perp_truth",
        )
        _, sigma_uperp_em = calculate_gaussian_fit_params_from_arrays(
            qT, recoil["u_perp_em"], x_bin_edges,
            out_file=out_file, y_name="u_perp_em",
        )
        _, sigma_uperp_ml = calculate_gaussian_fit_params_from_arrays(
            qT, recoil["u_perp_ml"], x_bin_edges,
            out_file=out_file, y_name="u_perp_ml",
        )

        # uT mean at different scales
        mean_uT_truth, _ = calculate_mean_sigma_in_x_bins(qT, recoil["u_pt_truth"], x_bin_edges)
        mean_uT_em, _ = calculate_mean_sigma_in_x_bins(qT, recoil["u_pt_em"], x_bin_edges)
        mean_uT_ml, _ = calculate_mean_sigma_in_x_bins(qT, recoil["u_pt_ml"], x_bin_edges)

        # Normalize the widths before writing the histograms.
        relative_sigma_upar_truth = sigma_upar_truth / mean_upar_truth
        relative_sigma_upar_em = sigma_upar_em / mean_upar_em
        relative_sigma_upar_ml = sigma_upar_ml / mean_upar_ml
        relative_sigma_uperp_truth = sigma_uperp_truth / mean_uT_truth
        relative_sigma_uperp_em = sigma_uperp_em / mean_uT_em
        relative_sigma_uperp_ml = sigma_uperp_ml / mean_uT_ml

        out_file.cd()

        # save mean and sigmas of u_par_plus_qT and u_perp as histograms
        make_hist_from_bin_contents(
            "h_mean_u_par_plus_qT_truth", "<u_par_plus_qT_truth>;qT;u_par_plus_qT",
            x_bin_edges, mean_upar_truth,
        ).Write()
        make_hist_from_bin_contents(
            "h_sigma_u_par_plus_qT_truth", "Sigma u_par_plus_qT_truth/mean;qT;u_par_plus_qT/mean",
            x_bin_edges, relative_sigma_upar_truth,
        ).Write()
        make_hist_from_bin_contents(
            "h_sigma_u_perp_truth", "Sigma u_perp_truth;qT;Sigma u_perp_truth/mean(uT)",
            x_bin_edges, relative_sigma_uperp_truth,
        ).Write()
        make_hist_from_bin_contents(
            "h_mean_u_par_plus_qT_em", "<u_par_plus_qT_em>;qT;",
            x_bin_edges, mean_upar_em,
        ).Write()
        make_hist_from_bin_contents(
            "h_sigma_u_par_plus_qT_em", "Sigma u_par_plus_qT_em/mean;qT;Sigma u_par_plus_qT_em/mean",
            x_bin_edges, relative_sigma_upar_em,
        ).Write()
        make_hist_from_bin_contents(
            "h_sigma_u_perp_em", "Sigma u_perp_em;qT;Sigma u_perp_em/mean(uT)",
            x_bin_edges, relative_sigma_uperp_em,
        ).Write()
        make_hist_from_bin_contents(
            "h_mean_u_par_plus_qT_ml", "<u_par_plus_qT_ml>;qT;",
            x_bin_edges, mean_upar_ml,
        ).Write()
        make_hist_from_bin_contents(
            "h_sigma_u_par_plus_qT_ml", "Sigma u_par_plus_qT_ml/mean;qT;Sigma u_par_plus_qT_ml/mean",
            x_bin_edges, relative_sigma_upar_ml,
        ).Write()
        make_hist_from_bin_contents(
            "h_sigma_u_perp_ml", "Sigma u_perp_ml;qT;Sigma u_perp_ml/mean(uT)",
            x_bin_edges, relative_sigma_uperp_ml,
        ).Write()

        # Resolution differences relative to truth.
        delta_sigma_upar_em_truth = sigma_upar_em - sigma_upar_truth
        delta_sigma_uperp_em_truth = sigma_uperp_em - sigma_uperp_truth
        delta_sigma_upar_ml_truth = sigma_upar_ml - sigma_upar_truth
        delta_sigma_uperp_ml_truth = sigma_uperp_ml - sigma_uperp_truth
        make_hist_from_bin_contents(
            "h_delta_sigma_upar_em_truth", "Delta sigma(u_par+qT): EM - truth;qT;",
            x_bin_edges, delta_sigma_upar_em_truth,
        ).Write()
        make_hist_from_bin_contents(
            "h_delta_sigma_uperp_em_truth", "Delta sigma(u_perp): EM - truth;qT;",
            x_bin_edges, delta_sigma_uperp_em_truth,
        ).Write()
        make_hist_from_bin_contents(
            "h_delta_sigma_upar_ml_truth", "Delta sigma(u_par+qT): ML - truth;qT;",
            x_bin_edges, delta_sigma_upar_ml_truth,
        ).Write()
        make_hist_from_bin_contents(
            "h_delta_sigma_uperp_ml_truth", "Delta sigma)(u_perp): ML - truth;qT",
            x_bin_edges, delta_sigma_uperp_ml_truth,
        ).Write()
    finally:
        out_file.Close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run hadrecoil response study.")
    parser.add_argument("--input", help="Path to input ROOT file.")
    parser.add_argument("--input-dir", help="Directory containing input root files.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="hadrecoil_response_study.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to process.")
    return parser.parse_args()


def main(args):
    input_files = list(args.input or [])
    if args.input_dir:
        input_files += [str(path) for path in Path(args.input_dir).rglob("*.root")]
    input_files = sorted(set(input_files))
    branches = [
        "mu_pt", "mu_eta", "mu_phi",
        "tu_pt.second", "tu_phi.second",
        "u_pt.second", "u_phi.second",
        "fCluster_truthE", "fCluster_rawE",
        "fCluster_calE", "fCluster_MLE",
        "fClusterEta", "fClusterPhi",
        "fMETTruthPt", "fMETTruthEta",
        "fMETTruthPhi", "fMETTruthE",
        "tboson_pt", "tboson_eta", "tboson_phi",
        "avgMu", "nPrimVtx",
    ]
    logging.info("Processing %d ROOT files", len(input_files))
    analysis = HadRecoilCommon(input_files,args.tree,branches,args.nEvents)
    # Calculate recoil observables for all selected events.
    recoil = analysis.execute()

    logging.info("Selected %d events", len(recoil["qT"]))

    # Save response and resolution histograms.
    save_response_histograms(recoil, args.output)

    logging.info("Histograms saved to %s", args.output)



if __name__ == "__main__":
    main(parse_args())
