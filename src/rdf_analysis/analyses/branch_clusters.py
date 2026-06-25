#!/usr/bin/env python3

"""
Find branch clusters from the E_truth vs E_ML correlation plot.
"""

import argparse
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.cluster import _default_cluster_truth_energy_bin_edges
from rdf_analysis.core import NtupleProcessor, awkward_to_numpy

ROOT.gROOT.SetBatch(True)


CLUSTER_MOMENT_BRANCHES = [
    "cluster_eta",
    "cluster_SIGNIFICANCE",
    "cluster_time",
    "cluster_SECOND_TIME",
    "cluster_CENTER_LAMBDA",
    "cluster_CENTER_MAG",
    "cluster_ENG_FRAC_EM_INCL",
    "cluster_FIRST_ENG_DENS",
    "cluster_LONGITUDINAL",
    "cluster_LATERAL",
    "cluster_PTD",
    "cluster_ISOLATION",
    "cluster_closestDeltaR",
    "nprimvtx",
    "avgmu"
]


CLUSTER_TOWER_MOMENT_BRANCHES = [
    "cluster_FIRST_TOWER_PHI",
    "cluster_FIRST_TOWER_ETA",
    "cluster_SECOND_TOWER_PHI",
    "cluster_SECOND_TOWER_ETA",
    "cluster_FIRST_TOWER_R",
    "cluster_SECOND_TOWER_R",
    "cluster_TOWER_PTD",
]


def to_numpy(values):
    return awkward_to_numpy(values)


class NtupleProcessor_branch_clusters(NtupleProcessor):
    pass


@dataclass
class BranchClusterResults:
    moment_values_all: dict
    moment_values_offdiag: dict
    moment_values_diag: dict
    all_truth: list
    all_ml: list
    selected_truth: list
    selected_ml: list
    diag_truth: list
    diag_ml: list

    @classmethod
    def empty(cls, moment_branches):
        return cls(
            {branch: [] for branch in moment_branches},
            {branch: [] for branch in moment_branches},
            {branch: [] for branch in moment_branches},
            [],
            [],
            [],
            [],
            [],
            [],
        )


def get_moment_branches(proc, tree_name, include_tower_moments=False):
    branches = proc.branch_names()
    required = ["cluster_e_truth", "cluster_e_ML_correct"]
    if tree_name == "InJetEvents":
        required.append("jet_eta")
    elif tree_name == "OutJetEvents":
        required.append("cluster_eta")
    else:
        print("[Error]:: Unsupported tree {}. Expected InJetEvents or OutJetEvents.".format(tree_name))
        sys.exit(1)

    missing = [branch for branch in required if branch not in branches]

    if missing:
        print("[Error]:: Missing required branches: {}".format(", ".join(missing)))
        sys.exit(1)

    moment_branches = [branch for branch in CLUSTER_MOMENT_BRANCHES if branch in branches]
    print("[Info]:: Found {} moment branches".format(len(moment_branches)))
    if include_tower_moments:
        moment_branches += [branch for branch in CLUSTER_TOWER_MOMENT_BRANCHES if branch in branches]

    if not moment_branches:
        print("[Error]:: No cluster moment branches found.")
        sys.exit(1)

    return moment_branches


def eta_mask(jet_eta, abs_eta_min=None, abs_eta_max=None):
    mask = np.isfinite(jet_eta)

    if abs_eta_min is not None:
        mask &= np.abs(jet_eta) >= abs_eta_min
    if abs_eta_max is not None:
        mask &= np.abs(jet_eta) < abs_eta_max

    return mask


def cluster_energy_mask(cluster_truth, cluster_ml):
    cluster_truth = np.asarray(cluster_truth, dtype=np.float64)
    cluster_ml = np.asarray(cluster_ml, dtype=np.float64)
    return np.isfinite(cluster_truth) & np.isfinite(cluster_ml) & (cluster_truth > 0) & (cluster_ml > 0)


def offdiag_mask(cluster_truth, cluster_ml, width=0.2, scale="log", side="both", x_range=None, y_range=None):
    cluster_truth = np.asarray(cluster_truth, dtype=np.float64)
    cluster_ml = np.asarray(cluster_ml, dtype=np.float64)
    valid_energy = cluster_energy_mask(cluster_truth, cluster_ml)
    distance = np.full(cluster_truth.shape, np.nan, dtype=np.float64)

    if scale == "log":
        distance[valid_energy] = np.log10(cluster_ml[valid_energy] / cluster_truth[valid_energy])
    else:
        distance[valid_energy] = cluster_ml[valid_energy] - cluster_truth[valid_energy]

    if side == "above":
        mask = valid_energy & (distance > width)
    elif side == "below":
        mask = valid_energy & (distance < -width)
    else:
        mask = valid_energy & (np.abs(distance) > width)

    # Apply x and y range masks if provided
    if x_range is not None:
        mask = mask & (x_range[0] < cluster_truth) & (cluster_truth < x_range[1])

    if y_range is not None:
        mask = mask & (y_range[0] < cluster_ml) & (cluster_ml < y_range[1])

    return mask


def diagonal_mask(cluster_truth, cluster_ml, width=0.2, scale="log", x_range=None, y_range=None):
    cluster_truth = np.asarray(cluster_truth, dtype=np.float64)
    cluster_ml = np.asarray(cluster_ml, dtype=np.float64)
    valid_energy = cluster_energy_mask(cluster_truth, cluster_ml)
    distance = np.full(cluster_truth.shape, np.nan, dtype=np.float64)

    if scale == "log":
        distance[valid_energy] = np.log10(cluster_ml[valid_energy] / cluster_truth[valid_energy])
    else:
        distance[valid_energy] = cluster_ml[valid_energy] - cluster_truth[valid_energy]

    mask = valid_energy & (np.abs(distance) <= width)

    if x_range is not None:
        mask = mask & (x_range[0] < cluster_truth) & (cluster_truth < x_range[1])

    if y_range is not None:
        mask = mask & (y_range[0] < cluster_ml) & (cluster_ml < y_range[1])

    return mask


def hist_range(values):
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 1.0

    low = float(np.min(values))
    high = float(np.max(values))

    if low == high:
        width = max(abs(low) * 0.1, 1.0)
        return low - width, high + width

    padding = 0.05 * (high - low)
    return low - padding, high + padding


def fill_th2(name, title, x_values, y_values, bin_edges, write=True):
    bins = array("d", bin_edges.tolist())
    hist = ROOT.TH2D(name, title, len(bin_edges) - 1, bins, len(bin_edges) - 1, bins)
    if not write:
        hist.SetDirectory(0)

    mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0) & (y_values > 0)

    for x_value, y_value in zip(x_values[mask], y_values[mask]):
        hist.Fill(float(x_value), float(y_value))

    if write:
        hist.Write()
    return hist


def write_moment_histogram(hist_name, title, values, n_bins, range_values=None):
    values = np.asarray(values, dtype=np.float64)
    range_values = values if range_values is None else np.asarray(range_values, dtype=np.float64)
    low, high = hist_range(range_values)
    hist = ROOT.TH1D(
        hist_name,
        title,
        n_bins,
        low,
        high,
    )

    for value in values[np.isfinite(values)]:
        hist.Fill(float(value))

    hist.Write()


def write_moment_histograms(moment_values_all, moment_values_offdiag, moment_values_diag, n_bins):
    for branch, all_values in moment_values_all.items():
        offdiag_values = moment_values_offdiag[branch]
        diag_values = moment_values_diag[branch]
        range_values = all_values

        write_moment_histogram(
            f"h_branch_cluster_all_{branch}",
            f"All selected cluster {branch};{branch};Clusters",
            all_values,
            n_bins,
            range_values=range_values,
        )
        write_moment_histogram(
            f"h_branch_cluster_offdiag_{branch}",
            f"Off-diagonal cluster {branch};{branch};Clusters",
            offdiag_values,
            n_bins,
            range_values=range_values,
        )
        write_moment_histogram(
            f"h_branch_cluster_very_diagonal_{branch}",
            f"Very diagonal cluster {branch};{branch};Clusters",
            diag_values,
            n_bins,
            range_values=range_values,
        )


def write_energy_plot(all_truth, all_ml, selected_truth, selected_ml, diag_truth, diag_ml, args):
    all_truth = np.asarray(all_truth, dtype=np.float64)
    all_ml = np.asarray(all_ml, dtype=np.float64)
    selected_truth = np.asarray(selected_truth, dtype=np.float64)
    selected_ml = np.asarray(selected_ml, dtype=np.float64)
    diag_truth = np.asarray(diag_truth, dtype=np.float64)
    diag_ml = np.asarray(diag_ml, dtype=np.float64)
    bin_edges = _default_cluster_truth_energy_bin_edges(all_truth, n_bins=args.energy_n_bins)

    h_all = fill_th2(
        "h2_cluster_energy_truth_vs_ml_all",
        "All selected clusters;E_{cluster}^{truth};E_{cluster}^{ML}",
        all_truth,
        all_ml,
        bin_edges,
    )
    fill_th2(
        "h2_cluster_energy_truth_vs_ml_offdiag",
        "Selected off-diagonal clusters;E_{cluster}^{truth};E_{cluster}^{ML}",
        selected_truth,
        selected_ml,
        bin_edges,
    )
    fill_th2(
        "h2_cluster_energy_truth_vs_ml_very_diagonal",
        "Selected very diagonal clusters;E_{cluster}^{truth};E_{cluster}^{ML}",
        diag_truth,
        diag_ml,
        bin_edges,
    )

    canvas = ROOT.TCanvas("c_cluster_energy_truth_vs_ml_offdiag", "E_truth vs E_ML off-diagonal selection", 900, 800)
    canvas.SetLogx()
    canvas.SetLogy()
    canvas.SetLogz()
    h_all.Draw("COLZ")

    selected_mask = np.isfinite(selected_truth) & np.isfinite(selected_ml) & (selected_truth > 0) & (selected_ml > 0)
    n_selected = int(np.count_nonzero(selected_mask))
    if n_selected > 0:
        graph = ROOT.TGraph(
            n_selected,
            array("d", selected_truth[selected_mask].tolist()),
            array("d", selected_ml[selected_mask].tolist()),
        )
    else:
        graph = ROOT.TGraph()

    graph.SetName("g_cluster_energy_truth_vs_ml_offdiag")
    graph.SetTitle("Selected off-diagonal clusters")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(0.45)
    graph.SetMarkerColor(ROOT.kRed + 1)
    if graph.GetN() > 0:
        graph.Draw("P SAME")

    diagonal = ROOT.TLine(bin_edges[0], bin_edges[0], bin_edges[-1], bin_edges[-1])
    diagonal.SetLineColor(ROOT.kBlack)
    diagonal.SetLineStyle(2)
    diagonal.Write("l_cluster_energy_truth_vs_ml_diagonal")
    diagonal.Draw("SAME")

    if args.offdiag_scale == "log":
        factor = 10.0 ** args.offdiag_width
        upper = ROOT.TLine(bin_edges[0], bin_edges[0] * factor, bin_edges[-1], bin_edges[-1] * factor)
        lower = ROOT.TLine(bin_edges[0], bin_edges[0] / factor, bin_edges[-1], bin_edges[-1] / factor)
    else:
        upper = ROOT.TLine(bin_edges[0], bin_edges[0] + args.offdiag_width, bin_edges[-1], bin_edges[-1] + args.offdiag_width)
        lower = ROOT.TLine(bin_edges[0], bin_edges[0] - args.offdiag_width, bin_edges[-1], bin_edges[-1] - args.offdiag_width)

    for line, name in ((upper, "l_cluster_energy_truth_vs_ml_upper_offdiag"), (lower, "l_cluster_energy_truth_vs_ml_lower_offdiag")):
        line.SetLineColor(ROOT.kRed + 1)
        line.SetLineStyle(2)
        line.Write(name)

    if args.offdiag_side in {"both", "above"}:
        upper.Draw("SAME")
    if args.offdiag_side in {"both", "below"}:
        lower.Draw("SAME")

    graph.Write()
    canvas.Write()

    png_path = Path(args.output).with_suffix(".energy_truth_vs_ml.png")
    canvas.SaveAs(str(png_path))
    return png_path


class BranchClusterAnalysis:
    def __init__(self, args):
        self.args = args
        self.proc = NtupleProcessor_branch_clusters(args.input, args.tree, args.nEvents)
        self.proc.build_arrays()
        self.moment_branches = get_moment_branches(self.proc, args.tree, args.include_tower_moments)
        self.results = BranchClusterResults.empty(self.moment_branches)

    def run(self):
        print("[Info]:: Moment branches: {}".format(", ".join(self.moment_branches)))

        if self.args.tree == "InJetEvents":
            self.process_injet_events()
        elif self.args.tree == "OutJetEvents":
            self.process_outjet_events()

        self.write_outputs()

    def append_cluster_selection(self, cluster_truth, cluster_ml, moment_arrays, event_counter, location):
        if len(cluster_truth) != len(cluster_ml):
            print("[Error]:: Cluster size mismatch for event {}{}".format(event_counter, location))
            sys.exit(1)

        args = self.args
        results = self.results
        valid_energy = cluster_energy_mask(cluster_truth, cluster_ml)
        results.all_truth.extend(cluster_truth[valid_energy].tolist())
        results.all_ml.extend(cluster_ml[valid_energy].tolist())

        mask = offdiag_mask(
            cluster_truth,
            cluster_ml,
            width=args.offdiag_width,
            scale=args.offdiag_scale,
            side=args.offdiag_side,
            x_range=args.offdiag_xrange,
            y_range=args.offdiag_yrange,
        )
        diag_mask = diagonal_mask(
            cluster_truth,
            cluster_ml,
            width=args.diag_width,
            scale=args.diag_scale,
            x_range=args.diag_xrange,
            y_range=args.diag_yrange,
        )

        results.selected_truth.extend(cluster_truth[mask].tolist())
        results.selected_ml.extend(cluster_ml[mask].tolist())
        results.diag_truth.extend(cluster_truth[diag_mask].tolist())
        results.diag_ml.extend(cluster_ml[diag_mask].tolist())

        for branch in self.moment_branches:
            cluster_moment = moment_arrays[branch]
            if len(cluster_moment) != len(mask):
                print(
                    "[Error]:: Moment size mismatch for branch {}, event {}{}".format(
                        branch,
                        event_counter,
                        location,
                    )
                )
                sys.exit(1)

            results.moment_values_all[branch].extend(cluster_moment[valid_energy].tolist())
            results.moment_values_offdiag[branch].extend(cluster_moment[mask].tolist())
            results.moment_values_diag[branch].extend(cluster_moment[diag_mask].tolist())

    def process_injet_events(self):
        args = self.args
        n_events = self.proc.n_events_to_process
        columns = ["jet_eta", "cluster_e_truth", "cluster_e_ML_correct"] + self.moment_branches
        event_counter = 0

        for chunk in self.proc.iter_arrays(columns):
            df_jet_eta = chunk["jet_eta"]
            df_cluster_e_truth = chunk["cluster_e_truth"]
            df_cluster_e_ML = chunk["cluster_e_ML_correct"]

            for event in range(len(df_jet_eta)):
                if event_counter % 10000 == 0:
                    print("[Info]:: Processing event {}/{}".format(event_counter, n_events))

                jet_eta = to_numpy(df_jet_eta[event])
                keep_jet = eta_mask(jet_eta, args.abs_eta_min, args.abs_eta_max)

                if not (len(keep_jet) == len(df_cluster_e_truth[event]) == len(df_cluster_e_ML[event])):
                    print(
                        "[Error]:: Number of jets does not match number of cluster energy entries for event {}".format(
                            event_counter,
                        )
                    )
                    sys.exit(1)

                for branch in self.moment_branches:
                    if len(chunk[branch][event]) != len(keep_jet):
                        print(
                            "[Error]:: Number of jets does not match number of moment entries for branch {}, event {}".format(
                                branch,
                                event_counter,
                            )
                        )
                        sys.exit(1)

                for jet in np.where(keep_jet)[0]:
                    jet = int(jet)
                    cluster_truth = to_numpy(df_cluster_e_truth[event][jet])
                    cluster_ml = to_numpy(df_cluster_e_ML[event][jet])
                    moment_arrays = {
                        branch: to_numpy(chunk[branch][event][jet])
                        for branch in self.moment_branches
                    }

                    self.append_cluster_selection(
                        cluster_truth,
                        cluster_ml,
                        moment_arrays,
                        event_counter,
                        ", jet {}".format(jet),
                    )

                event_counter += 1

    def process_outjet_events(self):
        args = self.args
        n_events = self.proc.n_events_to_process
        columns = ["cluster_eta", "cluster_e_truth", "cluster_e_ML_correct"] + self.moment_branches
        event_counter = 0

        for chunk in self.proc.iter_arrays(columns):
            df_cluster_eta = chunk["cluster_eta"]
            df_cluster_e_truth = chunk["cluster_e_truth"]
            df_cluster_e_ML = chunk["cluster_e_ML_correct"]

            for event in range(len(df_cluster_e_truth)):
                if event_counter % 10000 == 0:
                    print("[Info]:: Processing event {}/{}".format(event_counter, n_events))

                cluster_eta = to_numpy(df_cluster_eta[event])
                cluster_truth = to_numpy(df_cluster_e_truth[event])
                cluster_ml = to_numpy(df_cluster_e_ML[event])

                if not (len(cluster_eta) == len(cluster_truth) == len(cluster_ml)):
                    print("[Error]:: Cluster size mismatch for event {}".format(event_counter))
                    sys.exit(1)

                keep_cluster = eta_mask(cluster_eta, args.abs_eta_min, args.abs_eta_max)
                cluster_truth = cluster_truth[keep_cluster]
                cluster_ml = cluster_ml[keep_cluster]

                moment_arrays = {}
                for branch in self.moment_branches:
                    cluster_moment = to_numpy(chunk[branch][event])
                    if len(cluster_moment) != len(keep_cluster):
                        print("[Error]:: Moment size mismatch for branch {}, event {}".format(branch, event_counter))
                        sys.exit(1)

                    moment_arrays[branch] = cluster_moment[keep_cluster]

                self.append_cluster_selection(
                    cluster_truth,
                    cluster_ml,
                    moment_arrays,
                    event_counter,
                    "",
                )

                event_counter += 1

    def write_outputs(self):
        args = self.args
        results = self.results
        output_file = ROOT.TFile(args.output, "RECREATE")
        write_moment_histograms(
            results.moment_values_all,
            results.moment_values_offdiag,
            results.moment_values_diag,
            args.n_bins,
        )
        png_path = write_energy_plot(
            results.all_truth,
            results.all_ml,
            results.selected_truth,
            results.selected_ml,
            results.diag_truth,
            results.diag_ml,
            args,
        )
        output_file.Close()

        print(
            "[Info]:: Wrote moment histograms for {} all clusters, {} off-diagonal clusters, and {} very diagonal clusters to {}".format(
                len(results.all_truth),
                len(results.selected_truth),
                len(results.diag_truth),
                args.output,
            )
        )
        print("[Info]:: Wrote E_truth vs E_ML selection plot to {}".format(png_path))


def main(args):
    BranchClusterAnalysis(args).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cluster moments for off-diagonal E_truth vs E_ML clusters.")
    parser.add_argument("--input", required=True, action="append", help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="branch_cluster_moments.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=-1, help="Number of events to process.")
    parser.add_argument("--jetAbsEtaMin", "--clusterAbsEtaMin", "--abs-eta-min", dest="abs_eta_min", type=float, default=None)
    parser.add_argument("--jetAbsEtaMax", "--clusterAbsEtaMax", "--abs-eta-max", dest="abs_eta_max", type=float, default=None)
    parser.add_argument("--offdiagWidth", "--offdiag-width", dest="offdiag_width", type=float, default=0.2)
    parser.add_argument("--offdiagScale", "--offdiag-scale", dest="offdiag_scale", choices=["log", "linear"], default="log")
    parser.add_argument("--offdiagSide", "--offdiag-side", dest="offdiag_side", choices=["both", "above", "below"], default="both")
    parser.add_argument("--offdiagXrange", "--offdiag-xrange", dest="offdiag_xrange", type=float, nargs='+', default=None)
    parser.add_argument("--offdiagYrange", "--offdiag-yrange", dest="offdiag_yrange", type=float, nargs='+', default=None)
    parser.add_argument("--diagWidth", "--diag-width", dest="diag_width", type=float, default=0.2)
    parser.add_argument("--diagScale", "--diag-scale", dest="diag_scale", choices=["log", "linear"], default="log")
    parser.add_argument("--diagXrange", "--diag-xrange", dest="diag_xrange", type=float, nargs='+', default=None)
    parser.add_argument("--diagYrange", "--diag-yrange", dest="diag_yrange", type=float, nargs='+', default=None)
    parser.add_argument("--includeTowerMoments", "--include-tower-moments", dest="include_tower_moments", action="store_true")
    parser.add_argument("--nBins", "--n-bins", dest="n_bins", type=int, default=100)
    parser.add_argument("--energyNBins", "--energy-n-bins", dest="energy_n_bins", type=int, default=100)

    main(parser.parse_args())
