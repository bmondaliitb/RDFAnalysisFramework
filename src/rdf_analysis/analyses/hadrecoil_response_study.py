#!/usr/bin/env python3
import argparse
from pathlib import Path
import ROOT
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.hadrecoil_common import NtupleProcessorRDF_hadrecoil
from rdf_analysis.stats import make_numpy_hist, make_param_hist, make_hist_from_bin_contents


def hadrecoil_response(proc, x_bin_edges, out_file):
    prev_dir = ROOT.gDirectory
    # calculate mean and sigma of u_par_plus_qT
    mean_truth_clus_resp, sigma_truth_clus_upar = proc.calculate_gaussian_fit_params(
        "qT", "u_par_plus_qT_truth", x_bin_edges, out_file=out_file
    )
    mean_em_resp, sigma_em_upar = proc.calculate_gaussian_fit_params(
        "qT", "u_par_plus_qT_em", x_bin_edges, out_file=out_file
    )
    mean_ml_resp, sigma_ml_upar = proc.calculate_gaussian_fit_params(
        "qT", "u_par_plus_qT_ml", x_bin_edges, out_file=out_file
    )

    # calculate mean and sigma of u_perp
    _, sigma_truth_uperp = proc.calculate_gaussian_fit_params("qT", "u_perp_truth", x_bin_edges, out_file=out_file)
    _, sigma_em_uperp = proc.calculate_gaussian_fit_params("qT", "u_perp_em", x_bin_edges, out_file=out_file)
    _, sigma_ml_uperp = proc.calculate_gaussian_fit_params("qT", "u_perp_ml", x_bin_edges, out_file=out_file)

    # uT mean at different scales
    mean_truth_clus_uT, sigma_truth_clus_uT = proc.calculate_mean_sigma_in_x_bins("qT", "u_pt_truth", x_bin_edges)
    mean_em_uT, sigma_em_uT = proc.calculate_mean_sigma_in_x_bins("qT", "u_pt_em", x_bin_edges)
    mean_ml_uT, sigma_ml_uT = proc.calculate_mean_sigma_in_x_bins("qT", "u_pt_ml", x_bin_edges)

    # save mean and sigmas of u_par_plus_qT and u_perp as histograms
    make_hist_from_bin_contents("h_mean_u_par_plus_qT_truth","<u_par_plus_qT_truth>;qT;u_par_plus_qT",
                                x_bin_edges, mean_truth_clus_resp).Write()
    make_hist_from_bin_contents("h_sigma_u_par_plus_qT_truth","Sigma u_par_plus_qT_truth/mean;qT;u_par_plus_qT/mean",
                                x_bin_edges, sigma_truth_clus_upar/mean_truth_clus_resp).Write()
    make_hist_from_bin_contents("h_sigma_u_perp_truth","Sigma u_perp_truth;qT;Sigma u_perp_truth/mean(uT)",
                                x_bin_edges, sigma_truth_uperp/mean_truth_clus_uT).Write()
    make_hist_from_bin_contents("h_mean_u_par_plus_qT_em","<u_par_plus_qT_em>;qT;",
                                x_bin_edges, mean_em_resp).Write()
    make_hist_from_bin_contents("h_sigma_u_par_plus_qT_em","Sigma u_par_plus_qT_em/mean;qT;Sigma u_par_plus_qT_em/mean",
                                x_bin_edges, sigma_em_upar/mean_em_resp).Write()
    make_hist_from_bin_contents("h_sigma_u_perp_em","Sigma u_perp_em;qT;Sigma u_perp_em/mean(uT)",
                                x_bin_edges, sigma_em_uperp/mean_em_uT).Write()
    make_hist_from_bin_contents("h_mean_u_par_plus_qT_ml","<u_par_plus_qT_ml>;qT;",
                                x_bin_edges, mean_ml_resp).Write()
    make_hist_from_bin_contents("h_sigma_u_par_plus_qT_ml","Sigma u_par_plus_qT_ml/mean;qT;Sigma u_par_plus_qT_ml/mean",
                                x_bin_edges, sigma_ml_upar/mean_ml_resp).Write()
    make_hist_from_bin_contents("h_sigma_u_perp_ml","Sigma u_perp_ml;qT;Sigma u_perp_ml/mean(uT)",
                                x_bin_edges, sigma_ml_uperp/mean_ml_uT).Write()

    # delta sigma
    ## em
    delta_sigma_upar_em_truth = sigma_em_upar - sigma_truth_clus_upar
    delta_sigma_uperp_em_truth = sigma_em_uperp - sigma_truth_uperp
    ## ml
    delta_sigma_upar_ml_truth = sigma_ml_upar - sigma_truth_clus_upar
    delta_sigma_uperp_ml_truth = sigma_ml_uperp - sigma_truth_uperp
    ## save these delta values as histograms
    make_hist_from_bin_contents("h_delta_sigma_upar_em_truth","Delta sigma(u_par+qT): EM - truth;qT;",
                                x_bin_edges, delta_sigma_upar_em_truth).Write()
    make_hist_from_bin_contents("h_delta_sigma_uperp_em_truth", "Delta sigma(u_perp): EM - truth;qT;",
                                x_bin_edges, delta_sigma_uperp_em_truth).Write()
    make_hist_from_bin_contents("h_delta_sigma_upar_ml_truth", "Delta sigma(u_par+qT): ML - truth;qT;",
                                x_bin_edges, delta_sigma_upar_ml_truth).Write()
    make_hist_from_bin_contents("h_delta_sigma_uperp_ml_truth", "Delta sigma)(u_perp): ML - truth;qT",
                                x_bin_edges, delta_sigma_uperp_ml_truth).Write()


    #df_u_mag_truth = proc.to_pandas(["u_mag_truth"])["u_mag_truth"].to_numpy()
    #df_u_mag_em = proc.to_pandas(["u_mag_em"])["u_mag_em"].to_numpy()
    #df_u_mag_ml = proc.to_pandas(["u_mag_ml"])["u_mag_ml"].to_numpy()
    #delta_u_mag_ml_truth = df_u_mag_ml - df_u_mag_truth
    #delta_u_mag_em_truth = df_u_mag_em - df_u_mag_truth

    #df_u_par_truth = proc.to_pandas(["u_par_truth"])["u_par_truth"].to_numpy()
    #df_u_par_em = proc.to_pandas(["u_par_em"])["u_par_em"].to_numpy()
    #df_u_par_ml = proc.to_pandas(["u_par_ml"])["u_par_ml"].to_numpy()
    #delta_u_par_ml_truth = df_u_par_ml - df_u_par_truth
    #delta_u_par_em_truth = df_u_par_em - df_u_par_truth

    #df_u_perp_truth = proc.to_pandas(["u_perp_truth"])["u_perp_truth"].to_numpy()
    #df_u_perp_em = proc.to_pandas(["u_perp_em"])["u_perp_em"].to_numpy()
    #df_u_perp_ml = proc.to_pandas(["u_perp_ml"])["u_perp_ml"].to_numpy()
    #delta_u_perp_ml_truth = df_u_perp_ml - df_u_perp_truth
    #delta_u_perp_em_truth = df_u_perp_em - df_u_perp_truth


def parse_args():
    parser = argparse.ArgumentParser(description="Run hadrecoil response study.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="hadrecoil_response_study.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to process.")
    return parser.parse_args()


def main(args):
    proc = NtupleProcessorRDF_hadrecoil(args.input, args.tree, args.nEvents)
    proc.build_dataframe()
    out_file = ROOT.TFile(args.output, "RECREATE")
    # pT(ll) bin edges
    x_bin_edges = [0,5,10,15,20,25,30,40,50,60,70,80,90,100,150,250]
    hadrecoil_response(proc, x_bin_edges, out_file)
    out_file.Close()
    return proc


if __name__ == "__main__":
    main(parse_args())

