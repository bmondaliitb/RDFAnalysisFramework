#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import ROOT
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.core import NtupleProcessor, awkward_to_numpy
from rdf_analysis.stats import (
    calculate_gaussian_fit_params_from_arrays,
    make_hist_from_bin_contents,
)

DEBUG = False
ROOT.gROOT.SetBatch(True)

USE_TOWER_CLUSTER_E_ML_CORRECT = False


class NtupleProcessor_jet(NtupleProcessor):
    pass


def to_numpy(values):
    return awkward_to_numpy(values)


def get_jet_energy(df_e):
    return np.sum(df_e)


def _jet_abs_eta_mask(jet_eta, jet_abs_eta_min, jet_abs_eta_max):
    eta = np.asarray(jet_eta, dtype=np.float64)
    abs_eta = np.abs(eta)
    mask = np.isfinite(abs_eta)

    if jet_abs_eta_min is not None:
        mask &= abs_eta >= jet_abs_eta_min
    if jet_abs_eta_max is not None:
        mask &= abs_eta < jet_abs_eta_max

    return mask


#def _default_x_bin_edges():
#    return [
#        20, 30, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
#        160, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
#        1800, 1900, 2000, 2100, 2300, 2500, 2700, 2900, 3200, 3900, 5000,
#    ]

def _default_x_bin_edges(): # for forward region (high pt jets)
    return [
        20, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1600,
        1800, 2500, 4000
    ]


def _cluster_e_ml_correct_branch(use_tower_cluster_e_ml_correct=USE_TOWER_CLUSTER_E_ML_CORRECT):
    if use_tower_cluster_e_ml_correct:
        return "cluster_e_ML_correct_tower"
    return "cluster_e_ML_correct"


def _collect_event_data(proc, use_tower_cluster_e_ml_correct=USE_TOWER_CLUSTER_E_ML_CORRECT):
    columns = [
        "jet_pt",
        "jet_eta",
        "jet_e_EM",
        "jet_pt_truth",
        "jet_e_truth",
        "cluster_e_truth",
        "cluster_e_EM",
        "cluster_e_ML",
        _cluster_e_ml_correct_branch(use_tower_cluster_e_ml_correct),
        "cluster_e_LC",
        "cluster_eta",
    ]
    return proc.iter_arrays(columns)


def _build_response_arrays(
    frames,
    total_events,
    jet_abs_eta_min=None,
    jet_abs_eta_max=None,
    use_tower_cluster_e_ml_correct=USE_TOWER_CLUSTER_E_ML_CORRECT,
):
    event_counter = 0
    cluster_e_ml_correct_branch = _cluster_e_ml_correct_branch(use_tower_cluster_e_ml_correct)

    jet_energy_recal_dict = {"EM": [], "ML": [], "LC": [], "truth": []}
    jet_pt_dict = {"EM": [], "truth": []}
    jet_energy_dict = {"EM": [], "truth": []}

    for chunk in frames:
        df_jet_pt = chunk["jet_pt"]
        df_jet_eta = chunk["jet_eta"]
        df_jet_e_EM = chunk["jet_e_EM"]
        df_jet_pt_truth = chunk["jet_pt_truth"]
        df_jet_e_truth = chunk["jet_e_truth"]
        df_cluster_e_truth = chunk["cluster_e_truth"]
        df_cluster_e_EM = chunk["cluster_e_EM"]
        df_cluster_e_ML = chunk["cluster_e_ML"]
        df_cluster_e_ML_correct = chunk[cluster_e_ml_correct_branch]
        df_cluster_e_LC = chunk["cluster_e_LC"]
        df_cluster_eta = chunk["cluster_eta"]

        for event in range(len(df_jet_pt)):
            if event_counter % 10000 == 0:
                print("[Info]:: Processing event {}/{}".format(event_counter, total_events))

            event_jet_pt = to_numpy(df_jet_pt[event])
            event_jet_eta = to_numpy(df_jet_eta[event])
            event_jet_e_EM = to_numpy(df_jet_e_EM[event])
            event_jet_pt_truth = to_numpy(df_jet_pt_truth[event])
            event_jet_e_truth = to_numpy(df_jet_e_truth[event])

            # verify size of number of jets and size of cluster branches matches
            if not (
                len(event_jet_pt) == len(df_cluster_e_truth[event])
                == len(df_cluster_e_EM[event]) == len(df_cluster_e_ML[event])
                == len(df_cluster_e_ML_correct[event]) == len(df_cluster_e_LC[event])
                == len(df_cluster_eta[event])
            ):
                print("[Error]:: Number of jets does not match number of cluster energy entries for event {}".format(event_counter))
                sys.exit(1)
            if not (
                len(event_jet_pt) == len(event_jet_eta)
                == len(event_jet_e_EM) == len(event_jet_pt_truth)
                == len(event_jet_e_truth)
            ):
                print("[Error]:: Number of jet entries does not match across branches for event {}".format(event_counter))
                sys.exit(1)

            jet_eta_mask = _jet_abs_eta_mask(
                event_jet_eta,
                jet_abs_eta_min,
                jet_abs_eta_max,
            )

            for jet in range(len(event_jet_pt)):
                if not jet_eta_mask[jet]:
                    continue

                event_cluster_e_truth = to_numpy(df_cluster_e_truth[event][jet])
                event_cluster_e_EM = to_numpy(df_cluster_e_EM[event][jet])
                event_cluster_e_ML = to_numpy(df_cluster_e_ML[event][jet])
                event_cluster_e_ML_correct = to_numpy(df_cluster_e_ML_correct[event][jet])
                event_cluster_e_LC = to_numpy(df_cluster_e_LC[event][jet])
                event_cluster_eta = to_numpy(df_cluster_eta[event][jet])

                if not (
                    len(event_cluster_e_truth) == len(event_cluster_e_EM) ==
                    len(event_cluster_e_ML) == len(event_cluster_e_ML_correct)
                    == len(event_cluster_e_LC) == len(event_cluster_eta)
                ):
                    print("[Error]:: Number of cluster entries does not match across scales for event {}, jet {}".format(event_counter, jet))
                    sys.exit(1)

                event_cluster_e_ML = np.where(
                    np.abs(event_cluster_eta) < 2.5,
                    event_cluster_e_ML,
                    event_cluster_e_ML_correct,
                )

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
        #make_hist_from_bin_contents(f"h_sigma_{name}", "", x_bin_edges, sigma_values / mean_values).Write()
        make_hist_from_bin_contents(f"h_sigma_{name}", "", x_bin_edges, sigma_values).Write()
        make_hist_from_bin_contents(f"h_mean_{name}", "", x_bin_edges, mean_values).Write()


def main(args, out_file=None):
    jet_abs_eta_min = getattr(args, "abs_eta_min", None)
    jet_abs_eta_max = getattr(args, "abs_eta_max", None)
    use_tower_cluster_e_ml_correct = getattr(
        args,
        "use_tower_cluster_e_ml_correct",
        USE_TOWER_CLUSTER_E_ML_CORRECT,
    )

    proc = NtupleProcessor_jet(args.input, args.tree, args.nEvents)
    proc.build_arrays()

    frames = _collect_event_data(
        proc,
        use_tower_cluster_e_ml_correct=use_tower_cluster_e_ml_correct,
    )
    response_data = _build_response_arrays(
        frames,
        proc.n_events_to_process,
        jet_abs_eta_min=jet_abs_eta_min,
        jet_abs_eta_max=jet_abs_eta_max,
        use_tower_cluster_e_ml_correct=use_tower_cluster_e_ml_correct,
    )
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
    parser = argparse.ArgumentParser(description="Process jet tuples with uproot/awkward and save response histograms to ROOT.")
    parser.add_argument("--input", required=True, action='append', help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="iqr_histograms.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=-1, help="Number of events to process.")
    parser.add_argument(
        "--jetAbsEtaMin",
        "--abs-eta-min",
        dest="abs_eta_min",
        type=float,
        default=None,
        help="Minimum jet |eta| to keep. Omit to leave the lower edge uncut.",
    )
    parser.add_argument(
        "--jetAbsEtaMax",
        "--abs-eta-max",
        dest="abs_eta_max",
        type=float,
        default=None,
        help="Maximum jet |eta| to keep, using an exclusive upper bound. Omit to leave the upper edge uncut.",
    )
    parser.add_argument(
        "--useTowerClusterEMLCorrect",
        "--use-tower-cluster-e-ml-correct",
        dest="use_tower_cluster_e_ml_correct",
        action="store_true",
        default=USE_TOWER_CLUSTER_E_ML_CORRECT,
        help="Use cluster_e_ML_correct_tower instead of cluster_e_ML_correct.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
