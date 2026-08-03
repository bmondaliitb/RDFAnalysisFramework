#!/usr/bin/env python3
"""Validate MET reconstructed from stored inputs against METMaker."""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Union

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.met_inputs_common import (
    METConfig,
    TransverseVector,
    build_maker_met,
    build_met_inputs,
    make_histogram_1d,
    wrap_delta_phi,
)
from rdf_analysis.core import NtupleProcessor


class Config:
    """Validation histogram settings."""

    PROGRESS_INTERVAL = 10000
    RESIDUAL_BINS = 200
    RESIDUAL_MIN = -1.0
    RESIDUAL_MAX = 1.0
    PHI_RESIDUAL_MIN = -0.02
    PHI_RESIDUAL_MAX = 0.02


@dataclass
class METResiduals:
    """Calculated-minus-METMaker closure values."""

    mpx: List[float] = field(default_factory=list)
    mpy: List[float] = field(default_factory=list)
    et: List[float] = field(default_factory=list)
    phi: List[float] = field(default_factory=list)

    def append(
        self,
        calculated: TransverseVector,
        reference: TransverseVector,
    ) -> None:
        self.mpx.append(calculated.x - reference.x)
        self.mpy.append(calculated.y - reference.y)
        self.et.append(calculated.pt - reference.pt)
        self.phi.append(wrap_delta_phi(calculated.phi - reference.phi))

    def convert_to_numpy(self) -> None:
        for component in ("mpx", "mpy", "et", "phi"):
            values = np.asarray(getattr(self, component), dtype=np.float64)
            setattr(self, component, values)

    def statistics(self, component: str) -> Dict[str, float]:
        values = getattr(self, component)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return {}
        return {
            "mean": float(np.mean(finite)),
            "rms": float(np.sqrt(np.mean(finite**2))),
            "max_abs": float(np.max(np.abs(finite))),
        }


class METInputValidationAnalysis:
    """Calculate MET from its inputs and check closure against METMaker."""

    def __init__(
        self,
        input_files: Union[str, List[str]],
        tree_name: str,
        output_path: str,
        max_events: int = -1,
    ):
        self.processor = NtupleProcessor(input_files, tree_name, max_events)
        self.output_path = output_path
        self.residuals = METResiduals()
        self.histograms = {}
        self.event_counts = {
            "total": 0,
            "valid_met_inputs": 0,
        }

    @property
    def columns(self) -> List[str]:
        input_branches = set(METConfig.MET_INPUT_BRANCHES.values())
        input_branches.remove(METConfig.MET_INPUT_BRANCHES["jet_eta"])
        return sorted(input_branches | set(METConfig.MET_MAKER_BRANCHES.values()))

    def run(self) -> None:
        """Process events and construct closure histograms."""
        self._validate_input_branches()
        print("Starting MET-input validation...")
        for arrays in self.processor.iter_arrays(self.columns):
            self._process_chunk(arrays)

        self.residuals.convert_to_numpy()
        self._build_histograms()

    def _validate_input_branches(self) -> None:
        missing = sorted(set(self.columns) - self.processor.branch_names())
        if missing:
            raise KeyError(
                "Input tree is missing required branches: {}".format(
                    ", ".join(missing)
                )
            )

    def _process_chunk(self, arrays: Mapping[str, object]) -> None:
        first_branch = METConfig.MET_INPUT_BRANCHES["electron_pt"]
        for event in range(len(arrays[first_branch])):
            if self.event_counts["total"] % Config.PROGRESS_INTERVAL == 0:
                print(
                    "[Info] Processing event {}".format(
                        self.event_counts["total"]
                    ),
                    flush=True,
                )
            self.event_counts["total"] += 1

            met_inputs = build_met_inputs(
                arrays,
                event,
                include_jet_eta=False,
            )
            if not met_inputs.has_consistent_size():
                continue
            self.event_counts["valid_met_inputs"] += 1

            self.residuals.append(
                met_inputs.calculate_met(),
                build_maker_met(arrays, event),
            )

    def _build_histograms(self) -> None:
        settings = {
            "mpx": (
                "#Delta p_{x}^{miss} [GeV]",
                Config.RESIDUAL_MIN,
                Config.RESIDUAL_MAX,
            ),
            "mpy": (
                "#Delta p_{y}^{miss} [GeV]",
                Config.RESIDUAL_MIN,
                Config.RESIDUAL_MAX,
            ),
            "et": (
                "#Delta p_{T}^{miss} [GeV]",
                Config.RESIDUAL_MIN,
                Config.RESIDUAL_MAX,
            ),
            "phi": (
                "#Delta #phi^{miss} [rad]",
                Config.PHI_RESIDUAL_MIN,
                Config.PHI_RESIDUAL_MAX,
            ),
        }
        for component, (axis_title, low, high) in settings.items():
            name = f"h_delta_{component}_maker"
            self.histograms[name] = make_histogram_1d(
                name,
                "calculated - METMaker;{};Events".format(axis_title),
                Config.RESIDUAL_BINS,
                low,
                high,
                getattr(self.residuals, component),
            )

    def write_histograms(self) -> None:
        """Write closure histograms to the configured ROOT file."""
        output = ROOT.TFile.Open(self.output_path, "RECREATE")
        if not output or output.IsZombie():
            raise OSError("Could not create {}".format(self.output_path))
        for histogram in self.histograms.values():
            histogram.Write()
        output.Close()
        print("Histograms saved to {}".format(self.output_path))

    def print_summary(self) -> None:
        print("\nMET INPUT VALIDATION")
        for component in ("mpx", "mpy", "et", "phi"):
            result = self.residuals.statistics(component)
            if not result:
                print("  {:>3s}: no finite entries".format(component))
                continue
            unit = "rad" if component == "phi" else "GeV"
            print(
                "  {:>3s}: mean={:+.4e} {}, RMS={:.4e} {}, "
                "max|delta|={:.4e} {}".format(
                    component,
                    result["mean"],
                    unit,
                    result["rms"],
                    unit,
                    result["max_abs"],
                    unit,
                )
            )
        print("\nEVENT COUNTS")
        for name, count in self.event_counts.items():
            print("  {:32s}: {:10d}".format(name, count))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate MET calculated from stored inputs against METMaker."
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
        "--output",
        default="validate_met_inputs.root",
        help="Output ROOT histogram file.",
    )
    return parser.parse_args()


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
    analysis = METInputValidationAnalysis(
        input_files=input_files_from_args(args),
        tree_name=args.tree,
        output_path=args.output,
        max_events=args.n_events,
    )
    analysis.run()
    analysis.write_histograms()
    analysis.print_summary()


if __name__ == "__main__":
    main()
