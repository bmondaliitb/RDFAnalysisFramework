#!/usr/bin/env python3
"""
Performance of hadronic recoil on including forward clusters
"""

import argparse
import glob
from pathlib import Path
from typing import Mapping
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

def cluster_recoil(leptons, clusters, scale, mask):
    energy = clusters.energy[scale]
    selected = mask & np.isfinite(energy) & np.isfinite(clusters.phi)
    selected &= np.isfinite(clusters.eta)
    result = hadronic_recoil_from_clusters_python(leptons.pt, leptons.eta, leptons.phi,
        energy[selected], clusters.eta[selected], clusters.phi[selected])
    return TransverseVector(result["hr_px"], result["hr_py"])



class HadRecoil:
    def __init__(self,files,tree,n_events):
        self.processor = NtupleProcessor(files,tree,n_events)
        self.event_counts={
            "total":0,
        }

    def set_branches(self, branches):
        self.branches = branches

    def execute(self):
        # get the branches coming in in chunks
        for array in self.processor.iter_arrays(self.branches):
            self.process_chunk(array)

    def process_chunk(self, array: Mapping[str, object]) -> None:

        for event in range(len(array["mu_pt"])):
            if self.event_counts["total"]%1000==0:
                logging.info(f"Processing event {event}")
            self.event_counts["total"] += 1

            # Skip events that do not have at least 2 muons
            #if len(array["mu_pt"][event]) < 2: continue

            #lets build recoil for this event
            clusters = met_inputs_common.build_clusters(array,event)
            leptons = met_inputs_common.build_leptons(array, event)

            hadrecoil = {}
            for energy_scale in METConfig.CLUSTER_SCALES:
                hadrecoil[energy_scale] = hadronic_recoil_from_clusters_python(leptons.pt, leptons.eta,leptons.phi,clusters.energy[energy_scale], clusters.eta,clusters.phi)
                print(f"energy scale {energy_scale} \t HR: {hadrecoil[energy_scale]['hr_pt']}")

            # hadreciol from tree
            hadrecoil_from_tree = self.calculate_recoil(array, event)
            logging.info(f"HR from tree: {hadrecoil_from_tree['u_pt_em']}")

            # Lets build different scenarios





    def calculate_recoil(self, data: Mapping[str, object], event: int) -> dict:
        """Calculate the dimuon momentum and recoil components in GeV."""
        #mu_pt = data["mu_pt"][event] / 1000.0
        #mu_phi = data["mu_phi"][event]

        ## Add the transverse momenta of the two muons.
        #mu1_px = mu_pt[0] * np.cos(mu_phi[0])
        #mu1_py = mu_pt[0] * np.sin(mu_phi[0])
        #mu2_px = mu_pt[1] * np.cos(mu_phi[1])
        #mu2_py = mu_pt[1] * np.sin(mu_phi[1])
        #q_x = mu1_px + mu2_px
        #q_y = mu1_py + mu2_py
        #qT = np.sqrt(q_x**2 + q_y**2)
        #qT_safe = np.maximum(qT, 1e-9)

        # For W analysis sue tboson_pt (truth boson pt)
        # boson pT
        qT = data["tboson_pt"][event]/1000
        logging.info(f"Truth boson pT: {qT}")
        # boson phi
        tboson_phi = data["tboson_phi"][event]
        q_x = qT*np.cos(tboson_phi)
        q_y = qT*np.sin(tboson_phi)

        recoil = dict()
        #recoil["mu_pt0"] = mu_pt[0]
        #recoil["lep_pt"] = mu_pt
        #recoil["lep_phi"] = mu_phi
        recoil["q_x"] = q_x
        recoil["q_y"] = q_y
        recoil["qT"] = qT
        #recoil["z_x"] = q_x
        #recoil["z_y"] = q_y
        #recoil["z_r"] = qT
        #recoil["qhat_x"] = q_x / qT
        #recoil["qhat_y"] = q_y / qT
        #recoil["qhat_perp_x"] = -q_y / qT
        #recoil["qhat_perp_y"] = q_x / qT

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


if __name__=="__main__":
    parser =  argparse.ArgumentParser("Performance of hadronic recoil on including forward clusters")
    parser.add_argument("--input", action="append", help="Input ROOT file; repeat for multiple files.")
    parser.add_argument("--input-dir", help="Recursively read ROOT files from this directory.")
    parser.add_argument("--tree", default="treeAnaWZ")
    parser.add_argument("--n-events", "--n-events", dest="n_events", type=int, default=-1)
    parser.add_argument("--output", default="hadronic_recoil_highmu_forward.root")

    args = parser.parse_args()

    input_files = list(args.input or [])
    if args.input_dir:
        files = glob.glob("**/*.root", recursive=True)
        input_files.append(f"{args.input_dir}/{file}" for file in files)
        input_files = list(Path(args.input_dir).rglob("*.root"))

    print(f"Input files: {input_files}")

    # data columns to read
    branches = list(METConfig.MET_INPUT_BRANCHES.values())
    branches += list(METConfig.LEPTON_BRANCHES.values())
    branches += list(METConfig.CLUSTER_DIRECTION_BRANCHES.values())
    branches += [branch for _, branch in METConfig.CLUSTER_SCALES.values()]
    branches += ["tboson_pt", "tboson_eta", "tboson_phi", "avgMu", "nPrimVtx"]
    branches += [METConfig.TRUTH_MET_PT_BRANCH, METConfig.TRUTH_MET_PHI_BRANCH]
    branches += ["mu_pt", "mu_phi", "tu_pt.second", "tu_phi.second", "u_pt.second", "u_phi.second"]
    branches += ["tboson_pt", "tboson_phi"]

    analysis = HadRecoil(input_files, args.tree, args.n_events)
    analysis.set_branches(branches)
    analysis.execute()
