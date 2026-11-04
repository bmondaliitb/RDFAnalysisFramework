#!/usr/bin/env python3
"""
Performance of hadronic recoil on including forward clusters
"""

import argparse
import json
from pathlib import Path
from typing import Mapping
from types import SimpleNamespace
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.physics.hadrecoil import hadronic_recoil_from_clusters_python
from rdf_analysis.core import NtupleProcessor
from rdf_analysis.analyses.met_inputs_common import METConfig
import rdf_analysis.analyses.met_inputs_common as met_inputs_common
from rdf_analysis.analyses.met_inputs_common  import TransverseVector

import numpy as np

import logging

# Set the minimum log level to output (default is WARNING)
logging.basicConfig(level=logging.INFO)

from rdf_analysis.stats.fits import calculate_sigma_iqr_in_x_bins
from rdf_analysis.stats.histograms import make_hist_from_bin_contents, make_numpy_hist

QT_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 160, 200, 260]
MU_BINS = list(range(0, 85, 5))
VERTEX_BINS = list(range(0, 51, 2))


def cluster_recoil(leptons, clusters, scale, mask, args=None, energy=None):
    """Apply the selected energy/pT threshold and cone correction to one subset."""
    if args is None:
        args = default_settings()
    if energy is None:
        energy = clusters.energy[scale]
    selected = mask & np.isfinite(energy) & np.isfinite(clusters.phi)
    selected &= np.isfinite(clusters.eta)
    result = hadronic_recoil_from_clusters_python(
        leptons.pt, leptons.eta, leptons.phi,
        energy[selected], clusters.eta[selected], clusters.phi[selected],
        threshold=args.cluster_threshold,
        threshold_on_momentum=(args.threshold_on == "momentum"),
        abs_eta_min=0.0,
        abs_eta_max=args.eta_max,
        dr_cone=args.lepton_cone,
        min_dist_cone=args.random_cone_distance,
    )
    return TransverseVector(result["hr_px"], result["hr_py"])


def default_settings():
    return SimpleNamespace(eta_min=2.5, eta_max=4.9, jet_pt_min=30.0,
        jet_radius=0.4, cluster_threshold=0.0, threshold_on="momentum",
        lepton_cone=0.2, random_cone_distance=0.4, min_avg_mu=0.0)


class HadRecoil:
    def __init__(self, files, tree, n_events, args=None):
        self.processor = NtupleProcessor(files, tree, n_events)
        self.args = args if args is not None else default_settings()
        self.event_counts = dict.fromkeys([
            "total", "valid_met_inputs", "one_muon", "valid_w",
            "pileup_selection", "valid_truth_met", "valid_clusters",
            "selected", "with_selected_forward_jets"], 0)
        self.event_values = {key: [] for key in ("qT", "avgMu", "nPrimVtx")}
        names = ["standard_met", "tree_truth", "tree_em", "tree_lcw", "tree_ml"]
        for scale in METConfig.CLUSTER_SCALES:
            names += [f"central_clusters_{scale}", f"all_clusters_{scale}",
                f"keep_jets_forward_clusters_{scale}",
                f"replace_forward_jets_with_all_forward_clusters_{scale}"]
        self.values = {}
        for name in names:
            self.values[name] = {key: [] for key in (
                "u_pt", "u_par", "u_perp", "balance", "residual_x", "residual_y",
                "met_residual_x", "met_residual_y")}

    def set_branches(self, branches):
        self.branches = list(dict.fromkeys(branches))

    def execute(self):
        # Get the branches coming in in chunks.
        for array in self.processor.iter_arrays(self.branches):
            self.process_chunk(array)

    def save_histograms(self, output):
        event_values = {key: np.asarray(value, dtype=float)
                        for key, value in self.event_values.items()}
        values = {name: {key: np.asarray(value, dtype=float)
                        for key, value in observables.items()}
                  for name, observables in self.values.items()}
        save_histograms(event_values, values, self.event_counts, output, self.args)

    def calculate_event_recoils(self, met_inputs, leptons, clusters, ml_energy):
        """Build all comparisons for the same event, retaining MET jet weights."""
        args = self.args
        current_met = met_inputs.calculate_met()
        lepton_sum = met_inputs.muons.visible_vector() + met_inputs.electrons.visible_vector()
        standard = -current_met - lepton_sum
        recoils = {"standard_met": standard}

        forward_indices = met_inputs.forward_jet_indices(args.eta_min, args.eta_max)
        jet_pt = met_inputs.jets.pt[forward_indices]
        selected_jets = forward_indices[np.isfinite(jet_pt) & (jet_pt > args.jet_pt_min)]
        jet_sum = met_inputs.jets.visible_vector(selected_jets)

        central = clusters.in_abs_eta_range(0.0, args.eta_min)
        forward = clusters.in_abs_eta_range(args.eta_min, args.eta_max)
        outside_jets = clusters.away_from_jets(
            met_inputs.jets.eta[selected_jets], met_inputs.jets.phi[selected_jets], args.jet_radius,
        )
        for scale in METConfig.CLUSTER_SCALES:
            central_recoil = cluster_recoil(leptons, clusters, scale, central, args)
            full_recoil = cluster_recoil(
                leptons, clusters, scale, central | forward, args,
                energy=ml_energy if scale == "ml" else None,
            )
            extra_recoil = cluster_recoil(leptons, clusters, scale, forward & outside_jets, args)
            forward_recoil = cluster_recoil(leptons, clusters, scale, forward, args)
            recoils[f"central_clusters_{scale}"] = central_recoil
            recoils[f"all_clusters_{scale}"] = full_recoil
            recoils[f"keep_jets_forward_clusters_{scale}"] = standard + extra_recoil
            recoils[f"replace_forward_jets_with_all_forward_clusters_{scale}"] = standard - jet_sum + forward_recoil
        return recoils, lepton_sum, len(selected_jets)


    def process_chunk(self, array: Mapping[str, object]) -> None:
        args = self.args
        for event in range(len(array["mu_pt"])):
            if self.event_counts["total"] % 1000 == 0:
                logging.info(f"Processing event {self.event_counts['total']}")
            self.event_counts["total"] += 1
            met_inputs = met_inputs_common.build_met_inputs(array, event)
            if not met_inputs.has_consistent_size():
                continue
            self.event_counts["valid_met_inputs"] += 1
            leptons = met_inputs_common.build_leptons(array, event)
            if leptons.count < 1 or not (len(leptons.eta) == len(leptons.phi) == leptons.count):
                continue
            if not (np.isfinite(leptons.pt).all() and np.isfinite(leptons.eta).all()
                    and np.isfinite(leptons.phi).all()):
                continue
            self.event_counts["one_muon"] += 1

            qT = float(array["tboson_pt"][event]) / 1000.0
            w_eta = float(array["tboson_eta"][event])
            w_phi = float(array["tboson_phi"][event])
            if not np.isfinite([qT, w_eta, w_phi]).all() or abs(w_eta) >= 4.9:
                continue
            self.event_counts["valid_w"] += 1
            avg_mu = float(array["avgMu"][event])
            vertices = float(array["nPrimVtx"][event])
            if not np.isfinite([avg_mu, vertices]).all() or avg_mu < args.min_avg_mu:
                continue
            self.event_counts["pileup_selection"] += 1
            truth_met = met_inputs_common.build_truth_met(array, event)
            if truth_met is None or not truth_met.has_direction:
                continue
            self.event_counts["valid_truth_met"] += 1
            clusters = met_inputs_common.build_clusters(array, event)
            if not clusters.has_consistent_size():
                continue
            self.event_counts["valid_clusters"] += 1

            # Central ML energies are converted to GeV by build_clusters().
            # The forward ML branch is already in GeV and follows the same cluster order.
            forward_ml_energy = np.asarray(array["cluster_e_ML_forward"][event], dtype=float)
            if forward_ml_energy.shape != clusters.energy["ml"].shape:
                raise ValueError("cluster_e_ML_forward must have one energy per cluster.")
            forward = clusters.in_abs_eta_range(args.eta_min, args.eta_max)
            ml_energy = np.where(forward, forward_ml_energy, clusters.energy["ml"])
            recoils, lepton_sum, n_forward_jets = self.calculate_event_recoils(
                met_inputs, leptons, clusters, ml_energy,
            )
            # Compare the reconstructed scenarios with the recoil stored in the tree.
            tree_recoil = self.calculate_recoil(array, event)
            for scale in ("truth", "em", "lcw", "ml"):
                recoils[f"tree_{scale}"] = TransverseVector(
                    tree_recoil[f"u_x_{scale}"], tree_recoil[f"u_y_{scale}"])
            if not all(np.isfinite([u.x, u.y]).all() for u in recoils.values()):
                continue
            self.event_counts["selected"] += 1
            if n_forward_jets:
                self.event_counts["with_selected_forward_jets"] += 1
            self.event_values["qT"].append(qT)
            self.event_values["avgMu"].append(avg_mu)
            self.event_values["nPrimVtx"].append(vertices)

            cos_phi = np.cos(w_phi)
            sin_phi = np.sin(w_phi)
            for name, u in recoils.items():
                u_par = u.x * cos_phi + u.y * sin_phi
                u_perp = -u.x * sin_phi + u.y * cos_phi
                # Ideal recoil is -pT(W), so u_parallel + qT and u_perp vanish.
                met_residual = -u - lepton_sum - truth_met.vector
                self.values[name]["u_pt"].append(u.pt)
                self.values[name]["u_par"].append(u_par)
                self.values[name]["u_perp"].append(u_perp)
                self.values[name]["balance"].append(u_par + qT)
                self.values[name]["residual_x"].append(u.x + qT * cos_phi)
                self.values[name]["residual_y"].append(u.y + qT * sin_phi)
                self.values[name]["met_residual_x"].append(met_residual.x)
                self.values[name]["met_residual_y"].append(met_residual.y)


    def calculate_recoil(self, data: Mapping[str, object], event: int) -> dict:
        """Calculate the truth W momentum and recoil components in GeV."""


        qT = data["tboson_pt"][event]/1000
        tboson_phi = data["tboson_phi"][event]
        q_x = qT*np.cos(tboson_phi)
        q_y = qT*np.sin(tboson_phi)

        recoil = dict()
        recoil["q_x"] = q_x
        recoil["q_y"] = q_y
        recoil["qT"] = qT

        # Truth recoil (slot 0).
        u_pt = data["tu_pt.second"][event][0] / 1000.0
        u_phi = data["tu_phi.second"][event][0]
        u_x = u_pt * np.cos(u_phi)
        u_y = u_pt * np.sin(u_phi)
        u_par = (u_x * q_x + u_y * q_y) / qT
        u_perp = (q_x * u_y - q_y * u_x) / qT

        recoil["u_pt_truth"] = u_pt
        recoil["u_x_truth"] = u_x
        recoil["u_y_truth"] = u_y
        recoil["u_mag_truth"] = np.sqrt(u_x**2 + u_y**2)
        recoil["u_par_truth"] = u_par
        recoil["u_perp_truth"] = u_perp
        recoil["u_par_plus_qT_truth"] = u_par + qT
        recoil["u_perp_truth_alt"] = u_x * (-q_y / qT) + u_y * (q_x / qT)

        # EM recoil (slot 9).
        u_pt = data["u_pt.second"][event][9] / 1000.0
        u_phi = data["u_phi.second"][event][9]
        u_x = u_pt * np.cos(u_phi)
        u_y = u_pt * np.sin(u_phi)
        u_par = (u_x * q_x + u_y * q_y) / qT
        u_perp = (q_x * u_y - q_y * u_x) / qT

        recoil["u_pt_em"] = u_pt
        recoil["u_x_em"] = u_x
        recoil["u_y_em"] = u_y
        recoil["u_mag_em"] = np.sqrt(u_x**2 + u_y**2)
        recoil["u_par_em"] = u_par
        recoil["u_perp_em"] = u_perp
        recoil["u_par_plus_qT_em"] = u_par + qT
        recoil["u_perp_em_alt"] = u_x * (-q_y / qT) + u_y * (q_x / qT)

        # LCW recoil (slot 10).
        u_pt = data["u_pt.second"][event][10] / 1000.0
        u_phi = data["u_phi.second"][event][10]
        u_x = u_pt * np.cos(u_phi)
        u_y = u_pt * np.sin(u_phi)
        u_par = (u_x * q_x + u_y * q_y) / qT
        u_perp = (q_x * u_y - q_y * u_x) / qT

        recoil["u_pt_lcw"] = u_pt
        recoil["u_x_lcw"] = u_x
        recoil["u_y_lcw"] = u_y
        recoil["u_mag_lcw"] = np.sqrt(u_x**2 + u_y**2)
        recoil["u_par_lcw"] = u_par
        recoil["u_perp_lcw"] = u_perp
        recoil["u_par_plus_qT_lcw"] = u_par + qT
        recoil["u_perp_lcw_alt"] = u_x * (-q_y / qT) + u_y * (q_x / qT)

        # ML recoil (slot 11).
        u_pt = data["u_pt.second"][event][11] / 1000.0
        u_phi = data["u_phi.second"][event][11]
        u_x = u_pt * np.cos(u_phi)
        u_y = u_pt * np.sin(u_phi)
        u_par = (u_x * q_x + u_y * q_y) / qT
        u_perp = (q_x * u_y - q_y * u_x) / qT

        recoil["u_pt_ml"] = u_pt
        recoil["u_x_ml"] = u_x
        recoil["u_y_ml"] = u_y
        recoil["u_mag_ml"] = np.sqrt(u_x**2 + u_y**2)
        recoil["u_par_ml"] = u_par
        recoil["u_perp_ml"] = u_perp
        recoil["u_par_plus_qT_ml"] = u_par + qT
        recoil["u_perp_ml_alt"] = u_x * (-q_y / qT) + u_y * (q_x / qT)

        return recoil


def calculate_binned_response(x, qT, u_par, edges):
    """R = -mean(u_parallel)/mean(qT); an ideal response is one."""
    response = np.full(len(edges) - 1, np.nan)
    entries = np.zeros(len(edges) - 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        upper = x <= high if index == len(edges) - 2 else x < high
        selected = np.isfinite(x) & (x >= low) & upper
        selected &= np.isfinite(qT) & np.isfinite(u_par)
        entries[index] = np.count_nonzero(selected)
        if entries[index] >= 10 and np.mean(qT[selected]) > 0:
            response[index] = -np.mean(u_par[selected]) / np.mean(qT[selected])
    return response, entries


def save_histograms(event_values, values, counts, output_path, args):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.AddDirectory(False)
    output = ROOT.TFile.Open(output_path, "RECREATE")
    if not output or output.IsZombie():
        raise OSError(f"Could not create {output_path}")
    try:
        settings = dict(vars(args))
        settings["recoil_convention"] = "visible hadronic sum; ideal u = -pT(W)"
        settings["resolution"] = "(q84 - q16) / 2; at least 10 entries per bin"
        settings["cluster_scale_factors"] = METConfig.CLUSTER_SCALE_FACTORS
        settings["truth_met_components"] = METConfig.TRUTH_MET_REFERENCE_COMPONENTS
        ROOT.TNamed("settings", json.dumps(settings)).Write()
        cutflow = ROOT.TH1D("h_cutflow", "Event selection", len(counts), 0, len(counts))
        for index, (name, count) in enumerate(counts.items(), start=1):
            cutflow.GetXaxis().SetBinLabel(index, name)
            cutflow.SetBinContent(index, count)
        cutflow.Write()

        for name, observables in values.items():
            for key in ("u_pt", "balance", "u_perp", "residual_x", "residual_y",
                        "met_residual_x", "met_residual_y"):
                low, high = (0, 300) if key == "u_pt" else (-150, 150)
                make_numpy_hist(f"h_{key}_{name}", f"{name};{key} [GeV];Events",
                                120, low, high, observables[key]).Write()
            for variable, edges, axis_title in (
                ("qT", QT_BINS, "p_{T}^{W} [GeV]"),
                ("avgMu", MU_BINS, "<#mu>"),
                ("nPrimVtx", VERTEX_BINS, "N_{PV}"),
            ):
                x = event_values[variable]
                response, entries = calculate_binned_response(
                    x, event_values["qT"], observables["u_par"], edges,
                )
                make_hist_from_bin_contents(f"h_response_vs_{variable}_{name}",
                    f"{name};{axis_title};-<u_{{||}}>/<p_{{T}}^{{W}}>", edges, response).Write()
                make_hist_from_bin_contents(f"h_entries_vs_{variable}_{name}",
                    f"{name};{axis_title};Events", edges, entries).Write()
                for key in ("balance", "u_perp", "residual_x", "residual_y",
                            "met_residual_x", "met_residual_y"):
                    median, width = calculate_sigma_iqr_in_x_bins(x, observables[key], edges)
                    sigma = width / 2.0
                    corrected = np.full_like(sigma, np.nan)
                    np.divide(sigma, response, out=corrected,
                              where=np.isfinite(response) & (response > 0))
                    make_hist_from_bin_contents(f"h_median_{key}_vs_{variable}_{name}",
                        f"{name};{axis_title};median({key}) [GeV]", edges, median).Write()
                    make_hist_from_bin_contents(f"h_sigma_{key}_vs_{variable}_{name}",
                        f"{name};{axis_title};#sigma_{{IQR}}({key}) [GeV]", edges, sigma).Write()
                    if key in ("balance", "u_perp", "residual_x", "residual_y"):
                        make_hist_from_bin_contents(f"h_corrected_sigma_{key}_vs_{variable}_{name}",
                            f"{name};{axis_title};#sigma_{{IQR}}({key})/R [GeV]", edges, corrected).Write()
                valid = np.isfinite(response) & (response > 0)
                make_hist_from_bin_contents(f"h_valid_response_vs_{variable}_{name}",
                    f"{name};{axis_title};Valid response", edges, valid.astype(float)).Write()
    finally:
        output.Close()
    print(f"Saved histograms to {output_path}")
    print(json.dumps(counts, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", help="Input ROOT file; repeat for multiple files.")
    parser.add_argument("--input-dir", help="Recursively read ROOT files from this directory.")
    parser.add_argument("--tree", default="treeAnaWZ")
    parser.add_argument("--nEvents", "--n-events", dest="n_events", type=int, default=-1)
    parser.add_argument("--output", default="hadronic_recoil_highmu_forward.root")
    parser.add_argument("--eta-min", type=float, default=2.5, help="Start of the forward region.")
    parser.add_argument("--eta-max", type=float, default=4.9, help="Exclusive end of cluster acceptance.")
    parser.add_argument("--jet-pt-min", type=float, default=30.0, help="Strict forward-jet pT cut in GeV.")
    parser.add_argument("--jet-radius", type=float, default=0.4)
    parser.add_argument("--cluster-threshold", type=float, default=0.0, help="Inclusive threshold in GeV.")
    parser.add_argument("--threshold-on", choices=["momentum", "pt"], default="momentum")
    parser.add_argument("--lepton-cone", type=float, default=0.2)
    parser.add_argument("--random-cone-distance", type=float, default=0.4)
    parser.add_argument("--min-avg-mu", type=float, default=0.0, help="Optional high-pileup selection.")
    args = parser.parse_args()
    numeric = [args.eta_min, args.eta_max, args.jet_pt_min, args.jet_radius,
               args.cluster_threshold, args.lepton_cone, args.random_cone_distance, args.min_avg_mu]
    if not np.isfinite(numeric).all() or min(numeric) < 0:
        parser.error("Cuts must be finite and nonnegative.")
    if args.eta_min >= args.eta_max or args.jet_radius == 0:
        parser.error("Require eta-min < eta-max and jet-radius > 0.")
    if args.random_cone_distance >= np.pi:
        parser.error("random-cone-distance must be smaller than pi.")
    return args


def main():
    args = parse_args()
    input_files = list(args.input or [])
    if args.input_dir:
        input_files += [str(path) for path in Path(args.input_dir).rglob("*.root")]
    input_files = sorted(set(input_files))
    if not input_files:
        raise ValueError("Provide --input or --input-dir.")
    if Path(args.output).resolve() in [Path(path).resolve() for path in input_files]:
        raise ValueError("The output must not overwrite an input file.")

    branches = list(METConfig.MET_INPUT_BRANCHES.values())
    branches += list(METConfig.LEPTON_BRANCHES.values())
    branches += list(METConfig.CLUSTER_DIRECTION_BRANCHES.values())
    branches += [branch for _, branch in METConfig.CLUSTER_SCALES.values()]
    branches += ["cluster_e_ML_forward"]
    branches += ["tboson_pt", "tboson_eta", "tboson_phi", "avgMu", "nPrimVtx"]
    branches += [METConfig.TRUTH_MET_PT_BRANCH, METConfig.TRUTH_MET_PHI_BRANCH]
    branches += ["tu_pt.second", "tu_phi.second", "u_pt.second", "u_phi.second"]

    analysis = HadRecoil(input_files, args.tree, args.n_events, args)
    analysis.set_branches(branches)
    analysis.execute()
    analysis.save_histograms(args.output)


if __name__ == "__main__":
    main()
