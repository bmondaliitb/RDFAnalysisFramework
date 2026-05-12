#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import ROOT
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.core import NtupleProcessorRDF
from rdf_analysis.stats import calculate_gaussian_fit_params_from_arrays, make_hist_from_bin_contents

DEBUG = False
ROOT.gROOT.SetBatch(True)


class NtupleProcessorRDF_jet(NtupleProcessorRDF):
    def build_dataframe(self):
        df = self.df

        df = df.Define("d_jet_pt", "jet_pt")
        df = df.Define("d_jet_eta", "jet_eta")
        df = df.Define("d_jet_phi", "jet_phi")
        df = df.Define("d_jet_m", "jet_m")
        df = df.Define("d_jet_e_EM", "jet_e_EM")
        df = df.Define("d_jet_pt_truth", "jet_pt_truth")
        df = df.Define("d_jet_eta_truth", "jet_eta_truth")
        df = df.Define("d_jet_phi_truth", "jet_phi_truth")
        df = df.Define("d_jet_e_truth", "jet_e_truth")

        df = df.Define("d_cluster_pt", "cluster_pt_inJet")
        df = df.Define("d_cluster_eta", "cluster_eta_inJet")
        df = df.Define("d_cluster_phi", "cluster_phi_inJet")
        df = df.Define("d_cluster_e_truth", "cluster_e_truth_inJet")
        df = df.Define("d_cluster_e_EM", "cluster_e_EM_inJet")
        df = df.Define("d_cluster_e_ML", "cluster_e_ML_inJet")
        df = df.Define("d_cluster_e_LC", "cluster_e_LC_inJet")

        self.df = df
        return df


def get_jet_energy(df_e):
    return np.sum(df_e)


def _default_x_bin_edges():
    return [
        20, 30, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
        160, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
        1800, 1900, 2000, 2100, 2300, 2500, 2700, 2900, 3200, 3900, 5000,
    ]


def _collect_event_data(proc):
    columns = [
        "d_jet_pt",
        "d_jet_eta",
        "d_jet_phi",
        "d_jet_e_EM",
        "d_jet_pt_truth",
        "d_jet_e_truth",
        "d_cluster_pt",
        "d_cluster_eta",
        "d_cluster_phi",
        "d_cluster_e_truth",
        "d_cluster_e_EM",
        "d_cluster_e_ML",
        "d_cluster_e_LC",
    ]
    return {column: proc.to_pandas([column]) for column in columns}


def _build_response_arrays(frames):
    df_jet_pt = frames["d_jet_pt"]
    df_jet_e_EM = frames["d_jet_e_EM"]
    df_jet_pt_truth = frames["d_jet_pt_truth"]
    df_jet_e_truth = frames["d_jet_e_truth"]
    df_cluster_e_truth = frames["d_cluster_e_truth"]
    df_cluster_e_EM = frames["d_cluster_e_EM"]
    df_cluster_e_ML = frames["d_cluster_e_ML"]
    df_cluster_e_LC = frames["d_cluster_e_LC"]

    total_events = df_jet_pt.shape[0]
    event_counter = 0

    jet_energy_recal_dict = {"EM": [], "ML": [], "LC": [], "truth": []}
    jet_pt_dict = {"EM": [], "truth": []}
    jet_energy_dict = {"EM": [], "truth": []}

    for event in range(total_events):
        if event_counter % 10000 == 0:
            print("[Info]:: Processing event {}/{}".format(event, total_events))

        event_jet_pt = np.array(df_jet_pt.iloc[event]["d_jet_pt"])
        event_jet_e_EM = np.array(df_jet_e_EM.iloc[event]["d_jet_e_EM"])
        event_jet_pt_truth = np.array(df_jet_pt_truth.iloc[event]["d_jet_pt_truth"])
        event_jet_e_truth = np.array(df_jet_e_truth.iloc[event]["d_jet_e_truth"])

        for jet in range(len(event_jet_pt)):
            event_cluster_e_truth = np.array(df_cluster_e_truth.iloc[event]["d_cluster_e_truth"][jet])
            event_cluster_e_EM = np.array(df_cluster_e_EM.iloc[event]["d_cluster_e_EM"][jet])
            event_cluster_e_ML = np.array(df_cluster_e_ML.iloc[event]["d_cluster_e_ML"][jet])
            event_cluster_e_LC = np.array(df_cluster_e_LC.iloc[event]["d_cluster_e_LC"][jet])

            jet_energy_recal_EM = get_jet_energy(event_cluster_e_EM)
            jet_energy_recal_ML = get_jet_energy(event_cluster_e_ML)
            jet_energy_recal_LC = get_jet_energy(event_cluster_e_LC)
            jet_energy_recal_truth = get_jet_energy(event_cluster_e_truth)

            jet_energy_recal_dict["EM"].append(jet_energy_recal_EM)
            jet_energy_recal_dict["ML"].append(jet_energy_recal_ML)
            jet_energy_recal_dict["LC"].append(jet_energy_recal_LC)
            jet_energy_recal_dict["truth"].append(jet_energy_recal_truth)

            jet_pt_dict["EM"].append(event_jet_pt[jet])
            jet_pt_dict["truth"].append(event_jet_pt_truth[jet])
            jet_energy_dict["EM"].append(event_jet_e_EM[jet])
            jet_energy_dict["truth"].append(event_jet_e_truth[jet])

        event_counter += 1

    if DEBUG:
        print("Shape of jet_pt_dict['EM']: ", np.array(jet_pt_dict["EM"]).shape)
        print("Shape of jet_energy_recal_dict['EM']: ", np.array(jet_energy_recal_dict["EM"]).shape)

    jet_response_dict = {scale: np.array(jet_energy_recal_dict[scale]) / np.array(jet_energy_dict["truth"]) for scale in ["EM", "ML", "LC", "truth"]}
    jet_response_cluster_truth_dict = {
        scale: np.array(jet_energy_recal_dict[scale]) / np.array(jet_energy_recal_dict["truth"])
        for scale in ["EM", "ML", "LC", "truth"]
    }

    return {
        "jet_energy_recal_dict": jet_energy_recal_dict,
        "jet_pt_dict": jet_pt_dict,
        "jet_energy_dict": jet_energy_dict,
        "jet_response_dict": jet_response_dict,
        "jet_response_cluster_truth_dict": jet_response_cluster_truth_dict,
    }


def _fit_response_metrics(response_data, x_bin_edges, out_file):
    jet_energy_dict = response_data["jet_energy_dict"]
    jet_pt_dict = response_data["jet_pt_dict"]
    jet_energy_recal_dict = response_data["jet_energy_recal_dict"]
    jet_response_dict = response_data["jet_response_dict"]
    jet_response_cluster_truth_dict = response_data["jet_response_cluster_truth_dict"]

    fit_specs = {
        "em_jet_response_vs_truth_jet_energy": (jet_energy_dict["truth"], jet_response_dict["EM"]),
        "ml_jet_response_vs_truth_jet_energy": (jet_energy_dict["truth"], jet_response_dict["ML"]),
        "lc_jet_response_vs_truth_jet_energy": (jet_energy_dict["truth"], jet_response_dict["LC"]),
        "em_jet_response_vs_truth_jet_pt": (jet_pt_dict["truth"], jet_response_dict["EM"]),
        "ml_jet_response_vs_truth_jet_pt": (jet_pt_dict["truth"], jet_response_dict["ML"]),
        "lc_jet_response_vs_truth_jet_pt": (jet_pt_dict["truth"], jet_response_dict["LC"]),
        "em_jet_response_cluster_truth_vs_truth_jet_energy": (jet_energy_dict["truth"], jet_response_cluster_truth_dict["EM"]),
        "ml_jet_response_cluster_truth_vs_truth_jet_energy": (jet_energy_dict["truth"], jet_response_cluster_truth_dict["ML"]),
        "lc_jet_response_cluster_truth_vs_truth_jet_energy": (jet_energy_dict["truth"], jet_response_cluster_truth_dict["LC"]),
        "em_jet_response_cluster_truth_vs_truth_jet_pt": (jet_pt_dict["truth"], jet_response_cluster_truth_dict["EM"]),
        "ml_jet_response_cluster_truth_vs_truth_jet_pt": (jet_pt_dict["truth"], jet_response_cluster_truth_dict["ML"]),
        "lc_jet_response_cluster_truth_vs_truth_jet_pt": (jet_pt_dict["truth"], jet_response_cluster_truth_dict["LC"]),
        "em_jet_response_cluster_truth_vs_cluster_truth_jet_energy": (jet_energy_recal_dict["truth"], jet_response_cluster_truth_dict["EM"]),
        "ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy": (jet_energy_recal_dict["truth"], jet_response_cluster_truth_dict["ML"]),
        "lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy": (jet_energy_recal_dict["truth"], jet_response_cluster_truth_dict["LC"]),
    }

    metrics = {}
    for name, (x_values, y_values) in fit_specs.items():
        metrics[name] = calculate_gaussian_fit_params_from_arrays(x_values, y_values, x_bin_edges, out_file=out_file, y_name=name)

    return metrics


def _write_histograms(metrics, x_bin_edges):
    for name, (mean_values, sigma_values) in metrics.items():
        make_hist_from_bin_contents(f"h_sigma_{name}", "", x_bin_edges, sigma_values / mean_values).Write()
        make_hist_from_bin_contents(f"h_mean_{name}", "", x_bin_edges, mean_values).Write()


def main(args, out_file=None):

    proc = NtupleProcessorRDF_jet(args.input, args.tree, args.nEvents)
    proc.build_dataframe()

    frames = _collect_event_data(proc)
    response_data = _build_response_arrays(frames)
    x_bin_edges = _default_x_bin_edges()

    owned_out_file = out_file is None
    if owned_out_file:
        out_file = ROOT.TFile(args.output, "RECREATE")

    base_tdirectory = ROOT.gDirectory
    metrics = _fit_response_metrics(response_data, x_bin_edges, out_file)
    base_tdirectory.cd()
    _write_histograms(metrics, x_bin_edges)

    if owned_out_file:
        out_file.Close()

    return {
        "processor": proc,
        "response_data": response_data,
        "metrics": metrics,
        "x_bin_edges": x_bin_edges,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Process jet tuples with RDF and save response histograms to ROOT.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="iqr_histograms.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to process.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
