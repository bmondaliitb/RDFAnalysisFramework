#!/usr/bin/env python3
"""Study MET resolution for alternative forward-cluster inputs."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Union

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.met_inputs_common import (
    BinnedPerformance,
    METConfig,
    ObservableValues,
    build_clusters,
    build_leptons,
    build_met_inputs,
    build_truth_met,
    calculate_binned_performance,
    make_histogram_1d,
    make_histogram_2d,
)
from rdf_analysis.core import NtupleProcessor
from rdf_analysis.stats import make_hist_from_bin_contents



class Config:
    """Selection and histogram settings for the resolution study."""

    Z_MASS = 91.1876
    Z_WINDOW = 20.0
    FORWARD_ETA_MIN = 2.5
    FORWARD_ETA_MAX = 4.9
    CLUSTER_RADIUS = 0.4
    PROGRESS_INTERVAL = 10000

    PTZ_BIN_EDGES = (0,10,20,30,40,50,60,70,80,90,100,120,160,200,260)

    MET_BINS = 120
    MET_MIN = 0.0
    MET_MAX = 300.0
    PZ_BINS = 120
    PZ_MIN = -120.0
    PZ_MAX = 120.0


def parse_bin_edges(text: str) -> np.ndarray:
    """Parse comma-separated, strictly increasing pTZ bin edges."""
    edges = np.asarray([float(value) for value in text.split(",")])
    if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise argparse.ArgumentTypeError(
            "pTZ bins must be a comma-separated increasing list"
        )
    return edges


class METResolutionStudy:
    """Compare the requested MET definitions in forward-jet Z events."""

    BASE_VARIANT_LABELS = {"current": "current MET"}

    def __init__(
        self,
        input_files: Union[str, List[str]],
        tree_name: str,
        output_path: str,
        max_events: int = -1,
        eta_min: float = Config.FORWARD_ETA_MIN,
        eta_max: float = Config.FORWARD_ETA_MAX,
        cluster_radius: float = Config.CLUSTER_RADIUS,
        ptz_bin_edges: Iterable[float] = Config.PTZ_BIN_EDGES,
        apply_z_mass_window: bool = True,
    ):
        self.processor = NtupleProcessor(
            input_files,
            tree_name,
            max_events,
        )
        self.output_path = output_path
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.cluster_radius = cluster_radius
        self.ptz_bin_edges = np.asarray(ptz_bin_edges, dtype=np.float64)
        self.apply_z_mass_window = apply_z_mass_window
        self.available_branches = self.processor.branch_names()
        self.truth_pt_branch = METConfig.TRUTH_MET_PT_BRANCH
        self.truth_phi_branch = METConfig.TRUTH_MET_PHI_BRANCH
        self.truth_direction_branches = METConfig.TRUTH_MET_PT_BRANCH

        self.variant_labels = self._build_variant_labels()
        self.values = {
            name: ObservableValues()
            for name in self.variant_labels
        }
        self.performance: Dict[str, BinnedPerformance] = {}
        self.histograms = {}
        self.event_counts = {
            "total": 0,
            "valid_met_inputs": 0,
            "two_muons": 0,
            "z_mass_window": 0,
            "valid_z": 0,
            "valid_truth_met": 0,
            "valid_truth_direction": 0,
            "has_forward_met_input_jets": 0,
            "valid_global_clusters": 0,
        }

    @staticmethod
    def _build_variant_labels() -> Dict[str, str]:
        labels = dict(METResolutionStudy.BASE_VARIANT_LABELS)
        for scale, (scale_label, _) in METConfig.CLUSTER_SCALES.items():
            labels[f"keep_jets_forward_clusters_{scale}"] = (
                "forward jets kept + non-overlapping forward clusters, "
                "{} scale".format(scale_label)
            )
            labels[f"clusters_away_{scale}"] = (
                "forward jets removed + clusters away, {} scale".format(
                    scale_label
                )
            )
            labels[f"all_clusters_{scale}"] = (
                "forward jets removed + all clusters, {} scale".format(
                    scale_label
                )
            )
        return labels

    """
    List of tree branches to be read
    """
    @property
    def columns(self) -> List[str]:
        branches = (
            list(METConfig.MET_INPUT_BRANCHES.values())
            + [self.truth_pt_branch]
            + [self.truth_phi_branch]
            + list(METConfig.LEPTON_BRANCHES.values())
            + list(METConfig.CLUSTER_DIRECTION_BRANCHES.values())
            + [
                branch
                for _, branch in METConfig.CLUSTER_SCALES.values()
            ]
        )
        return sorted(set(branches))

    def run(self) -> None:
        """Process events and construct resolution histograms."""
        print("Starting forward-jet MET resolution study...")
        for index, input_file in enumerate(self.processor.input_file, start=1):
            remaining_events = (
                self.processor.n_events_to_process
                - self.event_counts["total"]
            )
            if remaining_events <= 0:
                break

            print(
                "[Info] Processing input file {}/{}: {}".format(
                    index,
                    len(self.processor.input_file),
                    input_file,
                ),
                flush=True,
            )
            file_processor = NtupleProcessor(
                input_file,
                self.processor.tree_name,
                remaining_events,
                step_size=self.processor.step_size,
            )
            for arrays in file_processor.iter_arrays(self.columns):
                self._process_chunk(arrays)

        for values in self.values.values():
            values.convert_to_numpy()
        self._calculate_performance()
        self._build_histograms()


    def _process_chunk(self, arrays: Mapping[str, object]) -> None:
        first_branch = METConfig.LEPTON_BRANCHES["pt"]
        for event in range(len(arrays[first_branch])):
            if self.event_counts["total"] % Config.PROGRESS_INTERVAL == 0:
                print(f"[Info] Processing event {self.event_counts["total"]}")
            self.event_counts["total"] += 1

            met_inputs = build_met_inputs(arrays, event)
            if not met_inputs.has_consistent_size():
                continue
            self.event_counts["valid_met_inputs"] += 1

            leptons = build_leptons(arrays, event)
            if leptons.count < 2:
                continue
            self.event_counts["two_muons"] += 1

            if self.apply_z_mass_window and not (
                Config.Z_MASS - Config.Z_WINDOW
                < leptons.dilepton_mass
                < Config.Z_MASS + Config.Z_WINDOW
            ):
                continue
            self.event_counts["z_mass_window"] += 1

            z_vector = leptons.z_vector
            if not np.isfinite(z_vector.pt) or z_vector.pt == 0.0:
                continue
            self.event_counts["valid_z"] += 1

            truth_met = build_truth_met(
                arrays,
                event,
                pt_branch=self.truth_pt_branch,
                phi_branch=self.truth_phi_branch,
            )
            if truth_met is None:
                continue
            self.event_counts["valid_truth_met"] += 1
            if truth_met.has_direction:
                self.event_counts["valid_truth_direction"] += 1
            forward_indices = met_inputs.forward_jet_indices(
                self.eta_min,
                self.eta_max,
            )
            # with this we can keep only events which has forward jets,
            if len(forward_indices) == 0:
                continue
            self.event_counts["has_forward_met_input_jets"] += 1

            clusters = build_clusters(arrays, event)
            if not clusters.has_consistent_size():
                continue
            self.event_counts["valid_global_clusters"] += 1

            current_met = met_inputs.calculate_met()
            self.values["current"].append(
                z_vector,
                current_met,
                truth_met,
            )

            jets_removed_met = met_inputs.met_without_jets(
                current_met,
                forward_indices,
            )
            away_mask = clusters.away_from_jets(
                met_inputs.jets.eta[forward_indices],
                met_inputs.jets.phi[forward_indices],
                self.cluster_radius,
            )
            away_vectors = clusters.visible_vectors(away_mask)
            all_vectors = clusters.visible_vectors(
                np.ones(clusters.count, dtype=bool)
            )
            forward_nonoverlap_vectors = clusters.visible_vectors(
                clusters.in_abs_eta_range(self.eta_min, self.eta_max)
                & away_mask
            )

            for scale in METConfig.CLUSTER_SCALES:
                self.values[f"keep_jets_forward_clusters_{scale}"].append(
                    z_vector,
                    current_met - forward_nonoverlap_vectors[scale],
                    truth_met,
                )
                self.values[f"clusters_away_{scale}"].append(
                    z_vector,
                    jets_removed_met - away_vectors[scale],
                    truth_met,
                )
                self.values[f"all_clusters_{scale}"].append(
                    z_vector,
                    jets_removed_met - all_vectors[scale],
                    truth_met,
                )

    def _calculate_performance(self) -> None:
        self.performance = {
            name: calculate_binned_performance(
                values.ptz,
                values.pz,
                self.ptz_bin_edges,
            )
            for name, values in self.values.items()
        }

    def _build_histograms(self) -> None:
        for name, values in self.values.items():
            self._build_variant_histograms(name, values)
        self._build_improvement_histograms()

    def _build_variant_histograms(
        self,
        name: str,
        values: ObservableValues,
    ) -> None:
        label = self.variant_labels[name]
        kinematic_finite = (
            np.isfinite(values.ptz)
            & np.isfinite(values.pz)
            & np.isfinite(values.met)
        )
        pz_residual_finite = (
            np.isfinite(values.ptz) & np.isfinite(values.pz_residual)
        )
        met_error_finite = np.isfinite(values.met_error)

        self.histograms[f"h_met_{name}"] = make_histogram_1d(
            f"h_met_{name}",
            f"{label};p_{{T}}^{{miss}} [GeV];Events",
            Config.MET_BINS,
            Config.MET_MIN,
            Config.MET_MAX,
            values.met[kinematic_finite],
        )
        self.histograms[f"h_pz_{name}"] = make_histogram_1d(
            f"h_pz_{name}",
            f"{label};P^{{Z}} [GeV];Events",
            Config.PZ_BINS,
            Config.PZ_MIN,
            Config.PZ_MAX,
            values.pz[kinematic_finite],
        )
        self.histograms[f"h_met_error_{name}"] = make_histogram_1d(
            f"h_met_error_{name}",
            (
                f"{label};"
                "|#bf{p}_{T}^{miss}-#bf{p}_{T}^{miss,true}| [GeV];Events"
            ),
            Config.MET_BINS,
            Config.MET_MIN,
            Config.MET_MAX,
            values.met_error[met_error_finite],
        )
        self.histograms[f"h_pz_residual_{name}"] = make_histogram_1d(
            f"h_pz_residual_{name}",
            (
                f"{label};"
                "(#bf{p}_{T}^{miss}-#bf{p}_{T}^{miss,true})"
                "#upoint#hat{A}^{Z} [GeV];Events"
            ),
            Config.PZ_BINS,
            Config.PZ_MIN,
            Config.PZ_MAX,
            values.pz_residual[pz_residual_finite],
        )
        self.histograms[f"h2_pz_vs_pTZ_{name}"] = make_histogram_2d(
            f"h2_pz_vs_pTZ_{name}",
            f"{label};p_{{T}}^{{Z}} [GeV];P^{{Z}} [GeV]",
            self.ptz_bin_edges,
            np.linspace(Config.PZ_MIN, Config.PZ_MAX, Config.PZ_BINS + 1),
            values.ptz[kinematic_finite],
            values.pz[kinematic_finite],
        )
        self.histograms[f"h2_pz_residual_vs_pTZ_{name}"] = (
            make_histogram_2d(
                f"h2_pz_residual_vs_pTZ_{name}",
                (
                    f"{label};p_{{T}}^{{Z}} [GeV];"
                    "(#bf{p}_{T}^{miss}-#bf{p}_{T}^{miss,true})"
                    "#upoint#hat{A}^{Z} [GeV]"
                ),
                self.ptz_bin_edges,
                np.linspace(Config.PZ_MIN, Config.PZ_MAX, Config.PZ_BINS + 1),
                values.ptz[pz_residual_finite],
                values.pz_residual[pz_residual_finite],
            )
        )

        performance = self.performance[name]
        quantities = {
            "entries": ("Events", performance.entries),
            "mean_pz": ("<P^{Z}> [GeV]", performance.mean_pz),
            "response": ("C^{Z}", performance.response),
            "raw_resolution": (
                "RMS(P^{Z}) [GeV]",
                performance.raw_resolution,
            ),
            "resolution": (
                "RMS(P^{Z})/C^{Z} [GeV]",
                performance.resolution,
            ),
        }
        for quantity, (axis_title, contents) in quantities.items():
            histogram_name = f"h_{quantity}_vs_pTZ_{name}"
            self.histograms[histogram_name] = make_hist_from_bin_contents(
                histogram_name,
                f"{label};p_{{T}}^{{Z}} [GeV];{axis_title}",
                self.ptz_bin_edges,
                contents,
            )

    def _build_improvement_histograms(self) -> None:
        current_resolution = self.performance["current"].resolution
        for name, performance in self.performance.items():
            if name == "current":
                continue
            histogram_name = f"h_resolution_improvement_{name}"
            self.histograms[histogram_name] = make_hist_from_bin_contents(
                histogram_name,
                (
                    f"{self.variant_labels[name]};p_{{T}}^{{Z}} [GeV];"
                    "#sigma_{current}-#sigma_{scenario} [GeV]"
                ),
                self.ptz_bin_edges,
                current_resolution - performance.resolution,
            )

    def write_histograms(self) -> None:
        """Write all study histograms to the configured ROOT file."""
        output = ROOT.TFile.Open(self.output_path, "RECREATE")
        if not output or output.IsZombie():
            raise OSError("Could not create {}".format(self.output_path))
        for histogram in self.histograms.values():
            histogram.Write()
        output.Close()
        print("Histograms saved to {}".format(self.output_path))

    def print_summary(self) -> None:
        print("\nFORWARD-JET MET RESOLUTION STUDY")
        for name, count in self.event_counts.items():
            print("  {:40s}: {:10d}".format(name, count))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Study MET resolution with alternative forward-cluster inputs "
            "at multiple energy scales."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        help="Input ROOT ntuple; repeat for multiple files.",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory searched recursively for ROOT ntuples.",
    )
    parser.add_argument(
        "--tree",
        default="treeAnaWZ",
        help="Input TTree name.",
    )
    parser.add_argument(
        "--nEvents",
        "--n-events",
        dest="n_events",
        type=int,
        default=-1,
        help="Maximum events to process; negative means all.",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=Config.FORWARD_ETA_MIN,
        help="Minimum absolute eta of a forward MET-input jet.",
    )
    parser.add_argument(
        "--eta-max",
        type=float,
        default=Config.FORWARD_ETA_MAX,
        help="Maximum absolute eta of a forward MET-input jet.",
    )
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=Config.CLUSTER_RADIUS,
        help="Minimum delta-R for the clusters-away scenario.",
    )
    parser.add_argument(
        "--ptz-bins",
        type=parse_bin_edges,
        default=np.asarray(Config.PTZ_BIN_EDGES, dtype=np.float64),
        help="Comma-separated pTZ bin edges in GeV.",
    )
    parser.add_argument(
        "--no-z-window",
        action="store_true",
        help="Disable the |m_mumu - mZ| < 20 GeV selection.",
    )
    parser.add_argument(
        "--output",
        default="met_resolution_study.root",
        help="Output ROOT histogram file.",
    )
    args = parser.parse_args()

    if args.eta_min < 0.0 or args.eta_min >= args.eta_max:
        parser.error("require 0 <= eta-min < eta-max")
    if args.cluster_radius <= 0.0:
        parser.error("cluster-radius must be positive")
    return args


def input_files_from_args(args) -> List[str]:
    input_files = list(args.input or [])
    if args.input_dir:
        for root, _, files in os.walk(args.input_dir):
            input_files.extend(
                os.path.join(root, name)
                for name in files
                if name.endswith(".root")
            )

    input_files = sorted(set(input_files))
    if not input_files:
        raise ValueError("Provide at least one --input or --input-dir.")
    return input_files


def main() -> None:
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.AddDirectory(False)
    args = parse_args()
    study = METResolutionStudy(
        input_files=input_files_from_args(args),
        tree_name=args.tree,
        output_path=args.output,
        max_events=args.n_events,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        cluster_radius=args.cluster_radius,
        ptz_bin_edges=args.ptz_bins,
        apply_z_mass_window=not args.no_z_window,
    )
    study.run()
    study.write_histograms()
    study.print_summary()


if __name__ == "__main__":
    main()
