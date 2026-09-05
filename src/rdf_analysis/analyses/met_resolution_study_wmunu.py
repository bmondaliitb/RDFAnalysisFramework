#!/usr/bin/env python3
"""W→μν MET resolution study with alternative forward-cluster inputs."""

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.met_inputs_common import (
    METConfig,
    TransverseVector,
    build_leptons,
)
from rdf_analysis.analyses.met_resolution_common import (
    BosonCandidate,
    METResolutionConfig,
    METResolutionStudyBase,
    build_argument_parser,
    input_files_from_args,
)


class Config(METResolutionConfig):
    """W-specific truth-boson branch settings."""

    BOSON_PT_BRANCH = "tboson_pt"
    BOSON_ETA_BRANCH = "tboson_eta"
    BOSON_PHI_BRANCH = "tboson_phi"


class METResolutionStudy(METResolutionStudyBase):
    """MET resolution study using the truth W boson direction."""

    CONFIG = Config
    BOSON_SYMBOL = "W"

    def boson_columns(self) -> List[str]:
        return [
            Config.BOSON_PT_BRANCH,
            Config.BOSON_ETA_BRANCH,
            Config.BOSON_PHI_BRANCH,
        ]

    def selection_event_counts(self) -> Dict[str, int]:
        return {"one_muon": 0, "valid_boson": 0}

    def select_boson(
        self,
        arrays: Mapping[str, object],
        event: int,
    ) -> Optional[BosonCandidate]:
        leptons = build_leptons(arrays, event)
        if leptons.count < 1:
            return None
        self.event_counts["one_muon"] += 1

        pt = float(arrays[Config.BOSON_PT_BRANCH][event]) * METConfig.GEV
        eta = float(arrays[Config.BOSON_ETA_BRANCH][event])
        phi = float(arrays[Config.BOSON_PHI_BRANCH][event])
        if (
            not np.isfinite(pt)
            or pt == 0.0
            or not np.isfinite(eta)
            or not np.isfinite(phi)
        ):
            return None
        vector = TransverseVector(
            pt * float(np.cos(phi)),
            pt * float(np.sin(phi)),
        )
        self.event_counts["valid_boson"] += 1
        return BosonCandidate(vector=vector, eta=eta)


def parse_args():
    parser = build_argument_parser(
        "Study W MET resolution with alternative forward-cluster inputs.",
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
