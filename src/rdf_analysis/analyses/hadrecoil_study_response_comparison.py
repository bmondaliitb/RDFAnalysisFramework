#!/usr/bin/env python3
import argparse
from pathlib import Path
import ROOT
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.hadrecoil_common import NtupleProcessorRDF_hadrecoil, default_study2_x_bin_edges
from rdf_analysis.stats import make_numpy_hist, make_param_hist


def study_2(proc, x_bin_edges, out_file):
    prev_dir = ROOT.gDirectory
    mean_truth_resp, sigma_truth_upar = proc.calculate_gaussian_fit_params(
        "qT", "u_par_plus_qT_truth", x_bin_edges, out_file=out_file
    )
    mean_base_resp, sigma_base_upar = proc.calculate_gaussian_fit_params(
        "qT", "u_par_plus_qT_em", x_bin_edges, out_file=out_file
    )
    mean_ml_resp, sigma_ml_upar = proc.calculate_gaussian_fit_params(
        "qT", "u_par_plus_qT_ml", x_bin_edges, out_file=out_file
    )

    _, sigma_truth_uperp = proc.calculate_gaussian_fit_params("qT", "u_perp_truth", x_bin_edges, out_file=out_file)
    _, sigma_base_uperp = proc.calculate_gaussian_fit_params("qT", "u_perp_em", x_bin_edges, out_file=out_file)
    _, sigma_ml_uperp = proc.calculate_gaussian_fit_params("qT", "u_perp_ml", x_bin_edges, out_file=out_file)

    delta_sigma_upar = sigma_ml_upar - sigma_base_upar
    delta_sigma_uperp = sigma_ml_uperp - sigma_base_uperp
    delta_mean_resp = mean_ml_resp - mean_base_resp

    df_u_mag_truth = proc.to_pandas(["u_mag_truth"])["u_mag_truth"].to_numpy()
    df_u_mag_em = proc.to_pandas(["u_mag_em"])["u_mag_em"].to_numpy()
    df_u_mag_ml = proc.to_pandas(["u_mag_ml"])["u_mag_ml"].to_numpy()
    delta_u_mag_ml_truth = df_u_mag_ml - df_u_mag_truth
    delta_u_mag_em_truth = df_u_mag_em - df_u_mag_truth

    df_u_par_truth = proc.to_pandas(["u_par_truth"])["u_par_truth"].to_numpy()
    df_u_par_em = proc.to_pandas(["u_par_em"])["u_par_em"].to_numpy()
    df_u_par_ml = proc.to_pandas(["u_par_ml"])["u_par_ml"].to_numpy()
    delta_u_par_ml_truth = df_u_par_ml - df_u_par_truth
    delta_u_par_em_truth = df_u_par_em - df_u_par_truth

    df_u_perp_truth = proc.to_pandas(["u_perp_truth"])["u_perp_truth"].to_numpy()
    df_u_perp_em = proc.to_pandas(["u_perp_em"])["u_perp_em"].to_numpy()
    df_u_perp_ml = proc.to_pandas(["u_perp_ml"])["u_perp_ml"].to_numpy()
    delta_u_perp_ml_truth = df_u_perp_ml - df_u_perp_truth
    delta_u_perp_em_truth = df_u_perp_em - df_u_perp_truth

    hists = [
        make_param_hist(
            "h_delta_sigma_upar_ml_minus_baseline",
            "Delta sigma(u_par+qT): ML - baseline",
            x_bin_edges,
            delta_sigma_upar,
            "Delta Gaussian #sigma",
        ),
        make_param_hist(
            "h_delta_sigma_uperp_ml_minus_baseline",
            "Delta sigma(u_perp): ML - baseline",
            x_bin_edges,
            delta_sigma_uperp,
            "Delta Gaussian #sigma",
        ),
        make_param_hist(
            "h_delta_mean_resp_ml_minus_baseline",
            "Delta mean(u_par+qT): ML - baseline",
            x_bin_edges,
            delta_mean_resp,
            "Delta Gaussian mean",
        ),
        make_numpy_hist("h_delta_u_mag_ml_minus_truth", "Delta u_mag(u_mag): ML - truth", 100, -50, 50, delta_u_mag_ml_truth),
        make_numpy_hist("h_delta_u_mag_em_minus_truth", "Delta u_mag_em(u_mag): EM - truth", 100, -50, 50, delta_u_mag_em_truth),
        make_numpy_hist("h_delta_u_par_ml_minus_truth", "Delta u_par(ML - truth)", 100, -50, 50, delta_u_par_ml_truth),
        make_numpy_hist("h_delta_u_par_em_minus_truth", "Delta u_par(EM - truth)", 100, -50, 50, delta_u_par_em_truth),
        make_numpy_hist("h_delta_u_perp_ml_minus_truth", "Delta u_perp(ML - truth)", 100, -50, 50, delta_u_perp_ml_truth),
        make_numpy_hist("h_delta_u_perp_em_minus_truth", "Delta u_perp(EM - truth)", 100, -50, 50, delta_u_perp_em_truth),
    ]

    prev_dir.cd()
    for hist in hists:
        hist.Write()


def parse_args():
    parser = argparse.ArgumentParser(description="Run hadrecoil study 2.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="hadrecoil_study_2.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to process.")
    return parser.parse_args()


def main(args):
    proc = NtupleProcessorRDF_hadrecoil(args.input, args.tree, args.nEvents)
    proc.build_dataframe()
    out_file = ROOT.TFile(args.output, "RECREATE")
    study_2(proc, default_study2_x_bin_edges(), out_file)
    out_file.Close()
    return proc


if __name__ == "__main__":
    main(parse_args())

