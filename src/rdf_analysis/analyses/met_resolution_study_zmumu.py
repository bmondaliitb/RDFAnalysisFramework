#!/usr/bin/env python3
"""Z→μμ MET resolution study with alternative forward-cluster inputs."""

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.met_inputs_common import build_leptons
from rdf_analysis.analyses.met_resolution_common import (
    BosonCandidate,
    METResolutionConfig,
    METResolutionStudyBase,
    build_argument_parser,
    input_files_from_args,
)


class Config(METResolutionConfig):
    """Z-specific selection settings."""

    Z_MASS = 91.1876
    Z_WINDOW = 20.0
    APPLY_Z_MASS_WINDOW = True


class METResolutionStudy(METResolutionStudyBase):
    """MET resolution study using a reconstructed dimuon Z boson."""

    CONFIG = Config
    BOSON_SYMBOL = "Z"

    def boson_columns(self) -> List[str]:
        return []

    def selection_event_counts(self) -> Dict[str, int]:
        return {
            "two_muons": 0,
            "z_mass_window": 0,
            "valid_z": 0,
            "pz_constraint": 0,
        }

    def select_boson(
        self,
        arrays: Mapping[str, object],
        event: int,
    ) -> Optional[BosonCandidate]:
        leptons = build_leptons(arrays, event)
        if leptons.count <= 1:
            return None
        self.event_counts["two_muons"] += 1

        if self.apply_z_mass_window and not (
            Config.Z_MASS - Config.Z_WINDOW
            < leptons.dilepton_mass
            < Config.Z_MASS + Config.Z_WINDOW
        ):
            return None
        self.event_counts["z_mass_window"] += 1

        vector = leptons.z_vector
        eta = leptons.z_eta
        if not np.isfinite(vector.pt) or vector.pt == 0.0 or not np.isfinite(eta):
            return None
        self.event_counts["valid_z"] += 1
        return BosonCandidate(vector=vector, eta=eta)


def parse_args():
    parser = build_argument_parser(
        "Study Z MET resolution with alternative forward-cluster inputs.",
    )
    return parser.parse_args()


def main() -> None:
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.AddDirectory(False)
    args = parse_args()
    study = METResolutionStudy(
        input_files=input_files_from_args(args),
        tree_name=args.tree,
        output_path=args.output,
        max_events=args.n_events,
    )
    study.run()
    study.write_histograms()
    study.print_summary()


if __name__ == "__main__":
    main()
