#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import ROOT
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.hadrecoil_common import NtupleProcessorRDF_hadrecoil
from rdf_analysis.physics import compute_sum_et_from_energy, hadrecoil_from_truth_particles, hadronic_recoil_from_clusters_python
from rdf_analysis.stats import get_bins_log, make_numpy_hist


def study_3(proc, outfile):
    prev_dir = ROOT.gDirectory

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

        abs_clus_eta = np.abs(event_clus_eta)
        abs_truth_eta = np.abs(event_truth_particle_eta)

        for region_name, eta_min, eta_max in eta_regions:
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

            hadrecoil_pt_em_by_region[region_name].append(hadrecoil_em_region["hr_pt"])
            hadrecoil_pt_ml_by_region[region_name].append(hadrecoil_ml_region["hr_pt"])
            hadrecoil_pt_truth_by_region[region_name].append(hadrecoil_truth_region["hr_pt"])
            hadrecoil_pt_truth_particle_by_region[region_name].append(hadrecoil_truth_particle_region["met_pt"])

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

    for region_name, _, _ in eta_regions:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run hadrecoil study 3.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="hadrecoil_study_3.root", help="Output ROOT file path.")
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

