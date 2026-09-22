#!/usr/bin/env python3
"""
Forward jet used in MET, evaluate their performance
why using ML contituents helps so much
"""

import os
import argparse
import math
from typing import List

import ROOT
import numpy as np

from rdf_analysis.stats.histograms import get_bins, get_bins_log


from rdf_analysis.analyses.met_inputs_common import (
    METConfig, TransverseVector, build_clusters, build_leptons,
    build_met_inputs, build_truth_met,
)
from rdf_analysis.core import NtupleProcessor, awkward_to_numpy
from rdf_analysis.stats import make_numpy_hist

GEV = METConfig.GEV


def dr(eta1, phi1, eta2, phi2):
    dphi = math.atan2(math.sin(phi1 - phi2), math.cos(phi1 - phi2))
    return math.hypot(eta1 - eta2, dphi)


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


def main():
    ROOT.gROOT.SetBatch(True)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", action="append", help="Input ROOT ntuple; repeat for multiple files.",)
    parser.add_argument("--input-dir", help="Directory searched recursively for ROOT ntuples.",)
    parser.add_argument("--tree", default="treeAnaWZ", help="Input TTree name.")
    parser.add_argument("--nEvents", "--n-events", dest="n_events", type=int, default=-1, help="Maximum events to process; negative means all.",)
    parser.add_argument("--output", default="output.root", help="Output ROOT histogram file.",)
    args = parser.parse_args()
    files = input_files_from_args(args)
    proc = NtupleProcessor(files, args.tree, args.n_events)

    columns = [
        "mu_pt", "mu_eta", "mu_phi", "tboson_pt", "tboson_eta", "tboson_phi",
        "tu_pt.second", "tu_phi.second", "avgMu", "nPrimVtx",
        "met_input_el_pt", "met_input_el_phi", "met_input_el_weight",
        "met_input_mu_pt", "met_input_mu_phi", "met_input_mu_weight",
        "met_input_jet_pt", "met_input_jet_eta", "met_input_jet_phi", "met_input_jet_weight",
        "met_softtrk_mpx", "met_softtrk_mpy",
        "fClusterEta", "fClusterPhi", "fCluster_rawE", "fCluster_calE",
        "fCluster_MLE", "cluster_e_ML_forward", "fCluster_truthE",
        "jet_pt", "jet_eta", "jet_phi",
        "jet_pileup_pt", "jet_pileup_e",
        "tjet_pt", "tjet_eta", "tjet_phi", "tjet_e",
    ]

    # define histograms
    # forward jet pt

    fjet_pt = []
    n_fjets = []
    fjet_summing_from_cluster_energy = {scale: [] for scale in METConfig.CLUSTER_SCALES}
    fjet_summing_from_cluster_pt = {scale: [] for scale in METConfig.CLUSTER_SCALES}

    ## cluster energy distributions at different scales
    cluster_energy_inside_jet = {scale: [] for scale in METConfig.CLUSTER_SCALES}

    event_counter = 0
    for arrays in proc.iter_arrays(columns):
        for event in range(len(arrays["mu_pt"])):
            event_counter = event_counter + 1
            if(event%1000 == 0):
                print(f"Processing event {event_counter}")
            # find forward jets used in MET
            met = build_met_inputs(arrays, event)
            forward_indices = met.forward_jet_indices(2.5,4.9)
            if (len(forward_indices)==0): continue
            # loop over forward jet and
            clusters = build_clusters(arrays, event)

            n_fjets.append(len(forward_indices))
            for fjet in forward_indices:
                fjet_pt.append(met.jets.pt[fjet])
                eta = math.fabs(met.jets.eta[fjet])
                cosh_eta = math.cosh(eta)
                # get clusters corresponding to this forward MET jet
                energy_inside_jet, cluster_energies = clusters.energy_inside_jet(
                    met.jets.eta[fjet],
                    met.jets.phi[fjet],
                    radius=0.4,
                )
                for scale, energy in energy_inside_jet.items():
                    fjet_summing_from_cluster_energy[scale].append(energy)
                    fjet_summing_from_cluster_pt[scale].append(energy/cosh_eta)

                for scale, energy in cluster_energies.items():
                    cluster_energy_inside_jet[scale].extend(energy)

    # create output root files
    tfile = ROOT.TFile(args.output, "RECREATE")
    bin_edges_jet_energy = get_bins_log(0.1, 1500, 30)
    bin_edges_jet_pt = get_bins_log(0.1, 300, 30)
    bin_edges_cluster_energy = get_bins_log(0.1, 500, 50)
    h_fjet_pt = make_numpy_hist("h_fjet_pt", ";p_{T}^{forward jet} [GeV];Events", bin_edges_jet_pt, fjet_pt)
    h_fjet_pt.Write()
    h_nfjet = make_numpy_hist("h_nfjet", "number of forward jets", 15, 0, 15,n_fjets)
    h_nfjet.Write()

    for scale, values in fjet_summing_from_cluster_energy.items():
        hist = make_numpy_hist(f"h_fjet_summing_cluster_energy_{scale}", f"jet energy summing clusters inside jet at {scale} scale", bin_edges_jet_energy, np.array(values),)
        hist.Write()
    for scale, values in fjet_summing_from_cluster_pt.items():
        hist = make_numpy_hist(f"h_fjet_summing_cluster_pt_{scale}", f"jet pT summing clusters inside jet at {scale}", bin_edges_jet_pt, np.array(values),)
        hist.Write()
    for scale, values in cluster_energy_inside_jet.items():
        hist = make_numpy_hist(f"h_cluster_energy_inside_jet_{scale}", f"cluster energy distribution at {scale}", bin_edges_cluster_energy, np.array(values),)
        hist.Write()
    tfile.Close()

if __name__ == "__main__":
    main()
