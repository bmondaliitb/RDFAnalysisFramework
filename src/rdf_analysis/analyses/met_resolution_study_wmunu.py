#!/usr/bin/env python3
"""W→μν MET resolution study with alternative forward-cluster inputs."""

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import ROOT
import argparse
import os

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

    def select_boson(self, arrays: Mapping[str, object], event: int,) -> Optional[BosonCandidate]:
        leptons = build_leptons(arrays, event)
        if leptons.count < 1:
            return None
        self.event_counts["one_muon"] += 1

        pt = float(arrays[Config.BOSON_PT_BRANCH][event]) * METConfig.GEV
        eta = float(arrays[Config.BOSON_ETA_BRANCH][event])
        phi = float(arrays[Config.BOSON_PHI_BRANCH][event])
        if (not np.isfinite(pt) or pt == 0.0 or not np.isfinite(eta) or not np.isfinite(phi)):
            return None
        vector = TransverseVector(pt * float(np.cos(phi)), pt * float(np.sin(phi)),)
        self.event_counts["valid_boson"] += 1
        return BosonCandidate(vector=vector, eta=eta)

def input_files_from_args(args) -> List[str]:
    """Resolve explicit and recursively discovered input ROOT files."""
    input_files = list(args.input or [])
    if args.input_dir:
        for root, _, files in os.walk(args.input_dir):
            input_files.extend(os.path.join(root, name) for name in files if name.endswith(".root"))
    input_files = sorted(set(input_files))
    if not input_files:
        raise ValueError("Provide at least one --input or --input-dir.")
    return input_files


def main() -> None:
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.AddDirectory(False)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", action="append", help="Input ROOT ntuple; repeat for multiple files.",)
    parser.add_argument("--input-dir", help="Directory searched recursively for ROOT ntuples.",)
    parser.add_argument("--tree", default="treeAnaWZ", help="Input TTree name.")
    parser.add_argument("--nEvents", "--n-events", dest="n_events", type=int, default=-1, help="Maximum events to process; negative means all.",)
    parser.add_argument("--output", default="output.root", help="Output ROOT histogram file.",)
    args = parser.parse_args()
    study = METResolutionStudy(input_files=input_files_from_args(args), tree_name=args.tree, output_path=args.output, max_events=args.n_events,)
    study.run()
    study.write_histograms()


if __name__ == "__main__":
    main()
