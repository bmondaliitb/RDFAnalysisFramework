#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import ROOT

from hadrecoil import *
from helper_function import *

from NtupleProcessorRDF import NtupleProcessorRDF
from hadrecoil import *
from helper_function import *


def study_threshold_scans(proc):

    threshold_values = [0.0, 0.5, 1.0, 2.0, 5.0]

    # event-wise inputs
    df_qT = proc.to_pandas(["qT"])["qT"].to_numpy()

    df_lep_pt = proc.to_pandas(["lep_pt"])
    df_lep_eta = proc.to_pandas(["lep_eta"])
    df_lep_phi = proc.to_pandas(["lep_phi"])

    df_clus_e_em = proc.to_pandas(["clus_e_em"])
    df_clus_e_lcw = proc.to_pandas(["clus_e_lcw"])
    df_clus_e_ml = proc.to_pandas(["clus_e_ml"])
    df_clus_eta = proc.to_pandas(["clus_eta"])
    df_clus_phi = proc.to_pandas(["clus_phi"])

    # boson direction
    df_qx = proc.to_pandas(["q_x"])["q_x"].to_numpy()
    df_qy = proc.to_pandas(["q_y"])["q_y"].to_numpy()
    df_qT_safe = np.maximum(df_qT, 1e-9)

    threshold_results = {}

    for calib_name, df_clus_e in [
        ("em", df_clus_e_em),
        ("lcw", df_clus_e_lcw),
        ("ml", df_clus_e_ml),
    ]:
        for thr in threshold_values:
            threshold_results[(calib_name, thr)] = {
                "qT": [],
                "u_perp": [],
                "u_par_plus_qT": [],
            }

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

        for calib_name, df_clus_e in [
            ("em", df_clus_e_em),
            ("lcw", df_clus_e_lcw),
            ("ml", df_clus_e_ml),
        ]:
            clus_e = np.array(df_clus_e.iloc[event][f"clus_e_{calib_name}"])

            scan_out = build_threshold_scan_observables(
                lep_pt, lep_eta, lep_phi,
                clus_e, clus_eta, clus_phi,
                threshold_values
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


def study_2(proc,x_bin_edges, out_file):
    prev_dir = ROOT.gDirectory
    mean_truth_resp, sigma_truth_upar = proc.calculate_gaussian_fit_params("qT", "u_par_plus_qT_truth", x_bin_edges,
                                                                           out_file=out_file)
    mean_base_resp, sigma_base_upar = proc.calculate_gaussian_fit_params("qT", "u_par_plus_qT_em", x_bin_edges,
                                                                         out_file=out_file)
    mean_ml_resp, sigma_ml_upar = proc.calculate_gaussian_fit_params("qT", "u_par_plus_qT_ml", x_bin_edges,
                                                                     out_file=out_file)

    _, sigma_truth_uperp = proc.calculate_gaussian_fit_params("qT", "u_perp_truth", x_bin_edges, out_file=out_file)
    _, sigma_base_uperp = proc.calculate_gaussian_fit_params("qT", "u_perp_em", x_bin_edges, out_file=out_file)
    _, sigma_ml_uperp = proc.calculate_gaussian_fit_params("qT", "u_perp_ml", x_bin_edges, out_file=out_file)

    # Deltas (ML - baseline)
    delta_sigma_upar = sigma_ml_upar - sigma_base_upar
    delta_sigma_uperp = sigma_ml_uperp - sigma_base_uperp
    delta_mean_resp = mean_ml_resp - mean_base_resp

    #
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

    h_delta_sigma_upar = make_param_hist(
        "h_delta_sigma_upar_ml_minus_baseline",
        "Delta sigma(u_par+qT): ML - baseline",
        x_bin_edges,
        delta_sigma_upar,
        "Delta Gaussian #sigma",
    )
    h_delta_sigma_uperp = make_param_hist(
        "h_delta_sigma_uperp_ml_minus_baseline",
        "Delta sigma(u_perp): ML - baseline",
        x_bin_edges,
        delta_sigma_uperp,
        "Delta Gaussian #sigma",
    )
    h_delta_mean_resp = make_param_hist(
        "h_delta_mean_resp_ml_minus_baseline",
        "Delta mean(u_par+qT): ML - baseline",
        x_bin_edges,
        delta_mean_resp,
        "Delta Gaussian mean",
    )

    h_delta_u_mag_ml_truth = make_numpy_hist(
        "h_delta_u_mag_ml_minus_truth",
        "Delta u_mag(u_mag): ML - truth",
        100, -50, 50,
        delta_u_mag_ml_truth
    )
    h_delta_u_mag_em_truth = make_numpy_hist(
        "h_delta_u_mag_em_minus_truth",
        "Delta u_mag_em(u_mag): EM - truth",
        100, -50, 50,
        delta_u_mag_em_truth
    )
    h_delta_u_par_ml_truth = make_numpy_hist(
        "h_delta_u_par_ml_minus_truth",
        "Delta u_par(ML - truth)",
        100, -50, 50,
        delta_u_par_ml_truth
    )
    h_delta_u_par_em_truth = make_numpy_hist(
        "h_delta_u_par_em_minus_truth",
        "Delta u_par(EM - truth)",
        100, -50, 50,
        delta_u_par_em_truth
    )
    h_delta_u_perp_ml_truth = make_numpy_hist(
        "h_delta_u_perp_ml_minus_truth",
        "Delta u_perp(ML - truth)",
        100, -50, 50,
        delta_u_perp_ml_truth
    )
    h_delta_u_perp_em_truth = make_numpy_hist(
        "h_delta_u_perp_em_minus_truth",
        "Delta u_perp(EM - truth)",
        100, -50, 50,
        delta_u_perp_em_truth
    )

    # go back to base tdirectory
    prev_dir.cd()
    h_delta_sigma_upar.Write()
    h_delta_sigma_uperp.Write()
    h_delta_mean_resp.Write()

    h_delta_u_mag_ml_truth.Write()
    h_delta_u_mag_em_truth.Write()
    h_delta_u_par_ml_truth.Write()
    h_delta_u_par_em_truth.Write()
    h_delta_u_perp_ml_truth.Write()
    h_delta_u_perp_em_truth.Write()


def study_3(proc, outfile):
    #### ---------- study 3 -----------------
    """
    Lets check the ut_pt calculation from hadrecoil.py matches with MiniTreeMaker
    """

    prev_dir = ROOT.gDirectory

    df_u_mag_truth = proc.to_pandas(["u_mag_truth"])["u_mag_truth"].to_numpy()
    df_u_mag_em = proc.to_pandas(["u_mag_em"])["u_mag_em"].to_numpy()
    df_u_mag_ml = proc.to_pandas(["u_mag_ml"])["u_mag_ml"].to_numpy()

    df_lep_pt = proc.to_pandas(["lep_pt"])
    df_lep_eta = proc.to_pandas(["lep_eta"])
    df_lep_phi = proc.to_pandas(["lep_phi"])

    df_clus_e_truth= proc.to_pandas(["clus_e_truth"])
    df_clus_e_em = proc.to_pandas(["clus_e_em"])
    df_clus_e_lcw = proc.to_pandas(["clus_e_lcw"])
    df_clus_e_ml = proc.to_pandas(["clus_e_ml"])
    df_clus_eta = proc.to_pandas(["clus_eta"])
    df_clus_phi = proc.to_pandas(["clus_phi"])

    # truth particles
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


    # creating dictionary of
    # {hadrecoil_pt_by_region = {"abs_eta_lt2pt": []}
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

    total_events = df_lep_pt.shape[0]
    event_counter = 0

    for event in range(0, total_events):
        # print every 10000 events
        if event_counter % 10000 == 0:
            print("[Info]:: Processing event {}/{}".format(event, total_events))
        event_lep_pt = np.array(df_lep_pt.iloc[event]["lep_pt"])
        event_lep_eta = np.array(df_lep_eta.iloc[event]["lep_eta"])
        event_lep_phi = np.array(df_lep_phi.iloc[event]["lep_phi"])

        event_clus_e_truth= np.array(df_clus_e_truth.iloc[event]["clus_e_truth"])
        event_clus_e_em = np.array(df_clus_e_em.iloc[event]["clus_e_em"])
        event_clus_e_ml = np.array(df_clus_e_ml.iloc[event]["clus_e_ml"])
        event_clus_eta = np.array(df_clus_eta.iloc[event]["clus_eta"])
        event_clus_phi = np.array(df_clus_phi.iloc[event]["clus_phi"])

        # met truth particles
        event_truth_particle_pt = np.array(df_truth_particle_pt.iloc[event]["met_truth_particle_pt"])
        event_truth_particle_eta = np.array(df_truth_particle_eta.iloc[event]["met_truth_particle_eta"])
        event_truth_particle_phi = np.array(df_truth_particle_phi.iloc[event]["met_truth_particle_phi"])
        event_truth_particle_e = np.array(df_truth_particle_e.iloc[event]["met_truth_particle_e"])


        abs_clus_eta = np.abs(event_clus_eta)
        abs_truth_eta = np.abs(event_truth_particle_eta)

        for region_name, eta_min, eta_max in eta_regions:
            # creating mask with eta cut
            region_clus_mask = (abs_clus_eta >= eta_min) & (abs_clus_eta < eta_max)
            region_truth_mask = (abs_truth_eta >= eta_min) & (abs_truth_eta < eta_max)

            region_clus_e_truth= event_clus_e_truth[region_clus_mask]
            region_clus_e_em = event_clus_e_em[region_clus_mask]
            region_clus_e_ml = event_clus_e_ml[region_clus_mask]
            region_clus_eta = event_clus_eta[region_clus_mask]
            region_clus_phi = event_clus_phi[region_clus_mask]

            region_truth_particle_pt = event_truth_particle_pt[region_truth_mask]
            region_truth_particle_eta = event_truth_particle_eta[region_truth_mask]
            region_truth_particle_phi = event_truth_particle_phi[region_truth_mask]
            region_truth_particle_e = event_truth_particle_e[region_truth_mask]

            # computing had recoil for different energy scales
            hadrecoil_em_region = hadronic_recoil_from_clusters_python(
                event_lep_pt,
                event_lep_eta,
                event_lep_phi,
                region_clus_e_em,
                region_clus_eta,
                region_clus_phi,
            )
            hadrecoil_ml_region = hadronic_recoil_from_clusters_python(
                event_lep_pt,
                event_lep_eta,
                event_lep_phi,
                region_clus_e_ml,
                region_clus_eta,
                region_clus_phi,
            )
            hadrecoil_truth_region = hadronic_recoil_from_clusters_python(
                event_lep_pt,
                event_lep_eta,
                event_lep_phi,
                region_clus_e_truth,
                region_clus_eta,
                region_clus_phi
            )
            hadrecoil_truth_particle_region = hadrecoil_from_truth_particles(
                region_truth_particle_pt,
                region_truth_particle_eta,
                region_truth_particle_phi,
            )

            # adding had recoil in the dictionary
            hadrecoil_pt_em_by_region[region_name].append(hadrecoil_em_region["hr_pt"])
            hadrecoil_pt_ml_by_region[region_name].append(hadrecoil_ml_region["hr_pt"])
            hadrecoil_pt_truth_by_region[region_name].append(hadrecoil_truth_region["hr_pt"])
            hadrecoil_pt_truth_particle_by_region[region_name].append(hadrecoil_truth_particle_region["met_pt"])

            # adding sum_et in the dictionary
            sumet_truth_by_region[region_name].append(compute_sum_et_from_energy(region_clus_e_truth, region_clus_eta))
            sumet_em_by_region[region_name].append(compute_sum_et_from_energy(region_clus_e_em, region_clus_eta))
            sumet_ml_by_region[region_name].append(compute_sum_et_from_energy(region_clus_e_ml, region_clus_eta))
            sumet_truth_particle_by_region[region_name].append(compute_sum_et_from_energy(region_truth_particle_e, region_truth_particle_eta))

            #sumet_truth_particle_tmp = compute_sum_et(region_truth_particle_pt)
            #sumet_truth_particle_tmp_1 = compute_sum_et_from_energy(region_truth_particle_e, region_truth_particle_eta)
            #print("sumet_truth_particle_tmp", sumet_truth_particle_tmp)
            #print("sumet_truth_particle_tmp_1", sumet_truth_particle_tmp_1)


            # adding cluster energy in the dictionary
            clus_energy_truth_by_region[region_name].extend(region_clus_e_truth)
            clus_energy_em_by_region[region_name].extend(region_clus_e_em)
            clus_energy_ml_by_region[region_name].extend(region_clus_e_ml)
            truth_particle_energy_by_region[region_name].extend(region_truth_particle_e)

        event_counter += 1

    # bin edges
    bin_edges_recoil = get_bins_log(0.5, 150, 40)
    bin_edges_sumet = get_bins_log(0.5, 1000, 40)
    bin_edges_energy = get_bins_log(0.05, 150, 50)

    # make histograms
    for region_name, _, _ in eta_regions:
        prev_dir.cd()
        # Try to get the directory if it exists, otherwise create it
        dir = outfile.Get(region_name)
        if not dir:
            dir = outfile.mkdir(region_name)
        dir.cd()

        h_recoil_pt_cluster_truth = make_numpy_hist(f"h_recoil_pt_cluster_truth", "", bin_edges_recoil, hadrecoil_pt_truth_by_region[region_name])
        h_recoil_pt_cluster_em = make_numpy_hist(f"h_recoil_pt_cluster_em", "", bin_edges_recoil, hadrecoil_pt_em_by_region[region_name])
        h_recoil_pt_cluster_ml = make_numpy_hist(f"h_recoil_pt_cluster_ml", "", bin_edges_recoil, hadrecoil_pt_ml_by_region[region_name])
        h_recoil_pt_truth_particle = make_numpy_hist("h_recoil_pt_truth_particle", "", bin_edges_recoil, hadrecoil_pt_truth_particle_by_region[region_name])

        h_sumet_cluster_truth= make_numpy_hist(f"h_sumet_cluster_truth","", bin_edges_sumet, sumet_truth_by_region[region_name])
        h_sumet_cluster_em = make_numpy_hist(f"h_sumet_cluster_em","", bin_edges_sumet, sumet_em_by_region[region_name])
        h_sumet_cluster_ml = make_numpy_hist(f"h_sumet_cluster_ml","", bin_edges_sumet, sumet_ml_by_region[region_name])
        h_sumet_truth_particle = make_numpy_hist("h_sumet_truth_particle", "", bin_edges_sumet, sumet_truth_particle_by_region[region_name])

        h_clus_energy_em = make_numpy_hist(f"h_clus_energy_em", "", bin_edges_energy, clus_energy_em_by_region[region_name])
        h_clus_energy_ml = make_numpy_hist(f"h_clus_energy_ml", "", bin_edges_energy, clus_energy_ml_by_region[region_name])
        h_clus_energy_truth = make_numpy_hist(f"h_clus_energy_truth", "", bin_edges_energy, clus_energy_truth_by_region[region_name])
        h_clus_energy_truth_particle = make_numpy_hist("h_clus_energy_truth_particle", "", bin_edges_energy, truth_particle_energy_by_region[region_name])

        h_recoil_pt_cluster_truth.Write()
        h_recoil_pt_cluster_em.Write()
        h_recoil_pt_cluster_ml.Write()
        h_recoil_pt_truth_particle.Write()

        h_sumet_cluster_truth.Write()
        h_sumet_cluster_em.Write()
        h_sumet_cluster_ml.Write()
        h_sumet_truth_particle.Write()

        h_clus_energy_em.Write()
        h_clus_energy_ml.Write()
        h_clus_energy_truth.Write()
        h_clus_energy_truth_particle.Write()

        # delete the histograms
        h_recoil_pt_cluster_truth.Delete()
        h_recoil_pt_cluster_em.Delete()
        h_recoil_pt_cluster_ml.Delete()
        h_recoil_pt_truth_particle.Delete()

        h_sumet_cluster_truth.Delete()
        h_sumet_cluster_em.Delete()
        h_sumet_cluster_ml.Delete()
        h_sumet_truth_particle.Delete()

        h_clus_energy_em.Delete()
        h_clus_energy_ml.Delete()
        h_clus_energy_truth.Delete()
        h_clus_energy_truth_particle.Delete()


def main(args):
    do_iqr = False
    do_gauss = True
    n_events = args.nEvents

    proc = NtupleProcessorRDF(args.input, args.tree, n_events)
    proc.build_dataframe()

    x_variable = "qT"
    y_variables = [
        "u_perp_em", "u_perp_lcw", "u_perp_ml",
        "u_par_em", "u_par_lcw", "u_par_ml",
        "u_par_plus_qT_em", "u_par_plus_qT_lcw", "u_par_plus_qT_ml",
    ]
    x_bin_edges = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64)
    out_file = ROOT.TFile(args.output, "RECREATE")
    #study_2(proc, x_bin_edges, out_file)
    study_3(proc, out_file)

    out_file.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process had recoil tuples with RDF and save IQR histograms to ROOT.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="iqr_histograms.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to process.")
    args = parser.parse_args()

    main(args)