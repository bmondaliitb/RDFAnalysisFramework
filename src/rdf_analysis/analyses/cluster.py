#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.core import NtupleProcessorRDF
from rdf_analysis.stats import (
    calculate_sigma_iqr_in_x_bins,
    get_bins_log,
    make_iqr_hist,
)

ROOT.gROOT.SetBatch(True)


class NtupleProcessorRDF_cluster(NtupleProcessorRDF):
    def build_dataframe(self):
        df = self.df

        df = df.Define("d_cluster_eta", "cluster_eta")
        df = df.Define("d_cluster_e_truth", "cluster_e_truth")
        df = df.Define("d_cluster_e_EM", "cluster_e_EM")
        df = df.Define("d_cluster_e_ML", "cluster_e_ML_correct")
        df = df.Define("d_cluster_e_LC", "cluster_e_LC")

        self.df = df
        return df



def _cluster_abs_eta_mask(cluster_eta, cluster_abs_eta_min, cluster_abs_eta_max):
    eta = np.asarray(cluster_eta, dtype=np.float64)
    abs_eta = np.abs(eta)
    mask = np.isfinite(abs_eta)

    if cluster_abs_eta_min is not None:
        mask &= abs_eta >= cluster_abs_eta_min
    if cluster_abs_eta_max is not None:
        mask &= abs_eta < cluster_abs_eta_max

    return mask


def _default_cluster_truth_energy_bin_edges(cluster_truth_energy, n_bins=50):
    energies = np.asarray(cluster_truth_energy, dtype=np.float64)
    positive_energies = energies[np.isfinite(energies) & (energies > 0)]

    if positive_energies.size == 0:
        return np.array([1.0, 2.0], dtype=np.float64)

    energy_min = 0.1
    energy_max = 5000

    if energy_max <= energy_min:
        return np.array([energy_min, energy_min * 1.5], dtype=np.float64)

    return get_bins_log(energy_min, energy_max, n_bins)


def _default_cluster_energy_distribution_bin_edges(cluster_energy_dict, n_bins=80):
    all_energies = []
    for values in cluster_energy_dict.values():
        energies = np.asarray(values, dtype=np.float64)
        all_energies.append(energies[np.isfinite(energies) & (energies > 0)])

    positive_energies = np.concatenate(all_energies) if all_energies else np.array([], dtype=np.float64)
    if positive_energies.size == 0:
        return np.array([1.0, 2.0], dtype=np.float64)

    energy_min = max(float(np.min(positive_energies)), 1e-3)
    energy_max = float(np.max(positive_energies))

    if energy_max <= energy_min:
        return np.array([energy_min, energy_min * 1.5], dtype=np.float64)

    return get_bins_log(energy_min, energy_max, n_bins)


def _default_cluster_response_bin_edges(cluster_response_dict, n_bins=10000):
    return np.linspace(0.0, 1000.0, n_bins + 1, dtype=np.float64)


def _collect_event_data(proc):
    columns = [
        "d_cluster_e_truth",
        "d_cluster_e_EM",
        "d_cluster_e_ML",
        "d_cluster_e_LC",
        "d_cluster_eta",
    ]

    return proc.to_numpy(columns)


def _empty_cluster_arrays():
    return {
        "energy": {"EM": [], "ML": [], "LC": [], "truth": []},
        "response": {"EM": [], "ML": [], "LC": []},
    }


def _append_cluster_response(cluster_data, cluster_truth, cluster_em, cluster_ml, cluster_lc):
    positive_truth_mask = np.isfinite(cluster_truth) & (cluster_truth > 0)
    if not np.any(positive_truth_mask):
        return

    truth = cluster_truth[positive_truth_mask]
    em = cluster_em[positive_truth_mask]
    ml = cluster_ml[positive_truth_mask]
    lc = cluster_lc[positive_truth_mask]

    cluster_data["energy"]["truth"].extend(truth.tolist())
    cluster_data["energy"]["EM"].extend(em.tolist())
    cluster_data["energy"]["ML"].extend(ml.tolist())
    cluster_data["energy"]["LC"].extend(lc.tolist())

    cluster_data["response"]["EM"].extend((em / truth).tolist())
    cluster_data["response"]["ML"].extend((ml / truth).tolist())
    cluster_data["response"]["LC"].extend((lc / truth).tolist())


def _to_numpy_cluster_data(cluster_data):
    return {
        "energy": {scale: np.array(values, dtype=np.float64) for scale, values in cluster_data["energy"].items()},
        "response": {scale: np.array(values, dtype=np.float64) for scale, values in cluster_data["response"].items()},
    }


def _build_response_arrays(frames, cluster_abs_eta_min=None, cluster_abs_eta_max=None):
    df_cluster_e_truth = frames["d_cluster_e_truth"]
    df_cluster_e_EM = frames["d_cluster_e_EM"]
    df_cluster_e_ML = frames["d_cluster_e_ML"]
    df_cluster_e_LC = frames["d_cluster_e_LC"]
    df_cluster_eta = frames["d_cluster_eta"]

    cluster_data = _empty_cluster_arrays()

    total_events = df_cluster_e_truth.shape[0]
    for event in range(total_events):
        if event % 10000 == 0:
            print("[Info]:: Processing event {}/{}".format(event, total_events))

        event_cluster_e_truth = df_cluster_e_truth[event]
        event_cluster_e_EM = df_cluster_e_EM[event]
        event_cluster_e_ML = df_cluster_e_ML[event]
        event_cluster_e_LC = df_cluster_e_LC[event]
        event_cluster_eta = df_cluster_eta[event]

        if not (
            len(event_cluster_e_truth) == len(event_cluster_e_EM)
            == len(event_cluster_e_ML) == len(event_cluster_e_LC)
        ):
            print("[Error]:: Number of cluster groups does not match across scales for event {}".format(event))
            sys.exit(1)

        for jet in range(len(event_cluster_e_truth)):
            cluster_truth = np.array(event_cluster_e_truth[jet])
            cluster_em = np.array(event_cluster_e_EM[jet])
            cluster_ml = np.array(event_cluster_e_ML[jet])
            cluster_lc = np.array(event_cluster_e_LC[jet])
            cluster_eta = np.array(event_cluster_eta[jet])

            if not (
                len(cluster_truth) == len(cluster_em)
                == len(cluster_ml) == len(cluster_lc)
            ):
                print("[Error]:: Number of cluster entries does not match across scales for event {}, jet {}".format(event, jet))
                sys.exit(1)

            eta_mask = _cluster_abs_eta_mask(
                cluster_eta,
                cluster_abs_eta_min,
                cluster_abs_eta_max,
            )
            cluster_truth = cluster_truth[eta_mask]
            cluster_em = cluster_em[eta_mask]
            cluster_ml = cluster_ml[eta_mask]
            cluster_lc = cluster_lc[eta_mask]

            _append_cluster_response(cluster_data, cluster_truth, cluster_em, cluster_ml, cluster_lc)

    arrays = _to_numpy_cluster_data(cluster_data)

    return {
        "cluster_energy_dict": arrays["energy"],
        "cluster_response_dict": arrays["response"],
    }

def _build_response_arrays_outjet_clusters(frames, cluster_abs_eta_min=None, cluster_abs_eta_max=None):
    df_cluster_e_truth = frames["d_cluster_e_truth"]
    df_cluster_e_EM = frames["d_cluster_e_EM"]
    df_cluster_e_ML = frames["d_cluster_e_ML"]
    df_cluster_e_LC = frames["d_cluster_e_LC"]
    df_cluster_eta = frames["d_cluster_eta"]

    cluster_data = _empty_cluster_arrays()

    total_events = df_cluster_e_truth.shape[0]
    for event in range(total_events):
        if event % 10000 == 0:
            print("[Info]:: Processing event {}/{}".format(event, total_events))

        event_cluster_e_truth = df_cluster_e_truth[event]
        event_cluster_e_EM = df_cluster_e_EM[event]
        event_cluster_e_ML = df_cluster_e_ML[event]
        event_cluster_e_LC = df_cluster_e_LC[event]
        event_cluster_eta = df_cluster_eta[event]

        if not (
            len(event_cluster_e_truth) == len(event_cluster_e_EM)
            == len(event_cluster_e_ML) == len(event_cluster_e_LC)
        ):
            print("[Error]:: Number of cluster groups does not match across scales for event {}".format(event))
            sys.exit(1)

        cluster_truth = np.array(event_cluster_e_truth)
        cluster_em = np.array(event_cluster_e_EM)
        cluster_ml = np.array(event_cluster_e_ML)
        cluster_lc = np.array(event_cluster_e_LC)
        cluster_eta = np.array(event_cluster_eta)

        eta_mask = _cluster_abs_eta_mask(
            cluster_eta,
            cluster_abs_eta_min,
            cluster_abs_eta_max,
        )
        cluster_truth = cluster_truth[eta_mask]
        cluster_em = cluster_em[eta_mask]
        cluster_ml = cluster_ml[eta_mask]
        cluster_lc = cluster_lc[eta_mask]

        _append_cluster_response(cluster_data, cluster_truth, cluster_em, cluster_ml, cluster_lc)

    arrays = _to_numpy_cluster_data(cluster_data)

    return {
        "cluster_energy_dict": arrays["energy"],
        "cluster_response_dict": arrays["response"],
    }



def _cluster_resolution_metrics(cluster_energy_dict, cluster_response_dict, x_bin_edges):
    cluster_truth_energy = cluster_energy_dict["truth"]
    metrics = {}
    for scale in ["EM", "ML", "LC"]:
        median_values, sigma_iqr_values = calculate_sigma_iqr_in_x_bins(
            cluster_truth_energy,
            cluster_response_dict[scale],
            x_bin_edges,
        )
        metrics[scale] = {
            "median": median_values,
            "sigma_iqr": sigma_iqr_values,
        }

    return metrics


def _write_cluster_resolution_histograms(cluster_metrics, x_bin_edges, region_label=None):
    for scale, values in cluster_metrics.items():
        region_prefix = f"{region_label}_" if region_label else ""
        title_prefix = f"{region_label} " if region_label else ""
        hist = make_iqr_hist(
            f"h_cluster_resolution_iqr_{region_prefix}{scale.lower()}_vs_truth_cluster_energy",
            f"{title_prefix}{scale} cluster resolution vs truth cluster energy",
            x_bin_edges,
            values["sigma_iqr"] / 2*values["median"], # sigma_iqr/2*median
            x_title="E_{cluster}^{truth} [GeV]",
            y_title=f"#sigma_{{IQR}}(E_{{cluster}}^{{{scale}}} / E_{{cluster}}^{{truth}})",
        )
        hist.Write()


def _write_cluster_energy_histograms(cluster_energy_dict, x_bin_edges, region_label=None):
    region_prefix = f"{region_label}_" if region_label else ""
    title_prefix = f"{region_label} " if region_label else ""
    edges = np.asarray(x_bin_edges, dtype=np.float64)

    for scale in ["truth", "EM", "ML", "LC"]:
        hist = ROOT.TH1D(
            f"h_cluster_energy_distribution_{region_prefix}{scale.lower()}",
            f"{title_prefix}{scale} cluster energy distribution",
            len(edges) - 1,
            edges,
        )
        hist.GetXaxis().SetTitle(f"E_{{cluster}}^{{{scale}}} [GeV]")
        hist.GetYaxis().SetTitle("Clusters")

        values = np.asarray(cluster_energy_dict[scale], dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        for value in values:
            hist.Fill(float(value))

        hist.Write()


def _write_2d_histogram(name, title, x_edges, y_edges, x_values, y_values):
    hist = ROOT.TH2D(
        name,
        title,
        len(x_edges) - 1,
        x_edges,
        len(y_edges) - 1,
        y_edges,
    )

    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    for x_value, y_value in zip(x_values[finite_mask], y_values[finite_mask]):
        hist.Fill(float(x_value), float(y_value))

    hist.Write()


def _write_cluster_scatter_histograms(
    cluster_energy_dict,
    cluster_response_dict,
    energy_bin_edges,
    response_bin_edges,
    region_label=None,
):
    region_prefix = f"{region_label}_" if region_label else ""
    title_prefix = f"{region_label} " if region_label else ""

    _write_2d_histogram(
        f"h2_cluster_response_{region_prefix}ml_vs_em",
        f"{title_prefix}ML cluster response vs EM cluster response;E_{{cluster}}^{{EM}} / E_{{cluster}}^{{truth}};E_{{cluster}}^{{ML}} / E_{{cluster}}^{{truth}}",
        response_bin_edges,
        response_bin_edges,
        cluster_response_dict["EM"],
        cluster_response_dict["ML"],
    )
    _write_2d_histogram(
        f"h2_cluster_energy_{region_prefix}ml_vs_truth",
        f"{title_prefix}ML cluster energy vs truth cluster energy;E_{{cluster}}^{{truth}} [GeV];E_{{cluster}}^{{ML}} [GeV]",
        energy_bin_edges,
        energy_bin_edges,
        cluster_energy_dict["truth"],
        cluster_energy_dict["ML"],
    )
    _write_2d_histogram(
        f"h2_cluster_energy_{region_prefix}em_vs_truth",
        f"{title_prefix}EM cluster energy vs truth cluster energy;E_{{cluster}}^{{truth}} [GeV];E_{{cluster}}^{{EM}} [GeV]",
        energy_bin_edges,
        energy_bin_edges,
        cluster_energy_dict["truth"],
        cluster_energy_dict["EM"],
    )


def main(args, out_file=None):
    cluster_abs_eta_min = args.abs_eta_min
    cluster_abs_eta_max = args.abs_eta_max

    proc = NtupleProcessorRDF_cluster(args.input, args.tree, args.nEvents)
    proc.build_dataframe()

    frames = _collect_event_data(proc)
    # For OutJetTree, the clusters are vector<double> per event, while for InJetEvents vector<vector<double>>
    if args.tree == "InJetEvents":
      response_data = _build_response_arrays(
          frames,
          cluster_abs_eta_min=cluster_abs_eta_min,
          cluster_abs_eta_max=cluster_abs_eta_max,
      )
    if args.tree == "OutJetEvents":
        response_data = _build_response_arrays_outjet_clusters(
            frames,
            cluster_abs_eta_min=cluster_abs_eta_min,
            cluster_abs_eta_max=cluster_abs_eta_max,
        )

    cluster_truth_energy_bin_edges = _default_cluster_truth_energy_bin_edges(
        response_data["cluster_energy_dict"]["truth"]
    )

    cluster_energy_distribution_bin_edges = _default_cluster_energy_distribution_bin_edges(
        response_data["cluster_energy_dict"]
    )
    cluster_response_bin_edges = _default_cluster_response_bin_edges(
        response_data["cluster_response_dict"]
    )

    owned_out_file = out_file is None
    if owned_out_file:
        out_file = ROOT.TFile(args.output, "RECREATE")

    cluster_metrics = _cluster_resolution_metrics(
        response_data["cluster_energy_dict"],
        response_data["cluster_response_dict"],
        cluster_truth_energy_bin_edges,
    )

    _write_cluster_energy_histograms(response_data["cluster_energy_dict"], cluster_energy_distribution_bin_edges)

    _write_cluster_resolution_histograms(cluster_metrics, cluster_truth_energy_bin_edges)

    _write_cluster_scatter_histograms(
        response_data["cluster_energy_dict"],
        response_data["cluster_response_dict"],
        cluster_energy_distribution_bin_edges,
        cluster_response_bin_edges,
    )

    if owned_out_file:
        out_file.Close()

    return {
        "processor": proc,
        "response_data": response_data,
        "cluster_metrics": cluster_metrics,
        "cluster_truth_energy_bin_edges": cluster_truth_energy_bin_edges,
        "cluster_energy_distribution_bin_edges": cluster_energy_distribution_bin_edges,
        "cluster_response_bin_edges": cluster_response_bin_edges,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Process cluster tuples with RDF and save response histograms to ROOT.")
    parser.add_argument("--input", required=True, action="append", help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="cluster.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=-1, help="Number of events to process.")
    parser.add_argument(
        "--clusterAbsEtaMin",
        "--abs-eta-min",
        dest="abs_eta_min",
        type=float,
        default=None,
        help="Minimum cluster |eta| to keep. Omit to leave the lower edge uncut.",
    )
    parser.add_argument(
        "--clusterAbsEtaMax",
        "--abs-eta-max",
        dest="abs_eta_max",
        type=float,
        default=None,
        help="Maximum cluster |eta| to keep, using an exclusive upper bound. Omit to leave the upper edge uncut.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(parse_args())
