#!/usr/bin/env python3
"""Truth-match forward MET jets and test removal of unmatched (PU) jets."""

import csv
import math
from pathlib import Path
import sys

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.met_inputs_common import (
    METConfig, TransverseVector, build_clusters, build_leptons,
    build_met_inputs, build_truth_met,
)
from rdf_analysis.analyses.met_resolution_common import (
    build_argument_parser, input_files_from_args,
)
from rdf_analysis.core import NtupleProcessor, awkward_to_numpy


ETA_MIN, ETA_MAX, MATCH_DR = 2.5, 4.9, 0.4
GEV = METConfig.GEV


def dr(eta1, phi1, eta2, phi2):
    dphi = math.atan2(math.sin(phi1 - phi2), math.cos(phi1 - phi2))
    return math.hypot(eta1 - eta2, dphi)


def nearest(eta, phi, etas, phis):
    if len(etas) == 0:
        return None, np.nan
    distances = np.asarray([dr(eta, phi, e, p) for e, p in zip(etas, phis)])
    i = int(np.argmin(distances))
    return (i, float(distances[i])) if distances[i] < MATCH_DR else (None, float(distances[i]))


def main():
    ROOT.gROOT.SetBatch(True)
    parser = build_argument_parser("Diagnose forward MET jets with truth matching.")
    parser.add_argument("--csv", required=True, help="Per-forward-jet CSV output")
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
        "jet_pt", "jet_eta", "jet_phi", "jet_cluster_truthE",
        "jet_pileup_pt", "jet_pileup_e",
        "tjet_pt", "tjet_eta", "tjet_phi", "tjet_e",
    ]

    h_ratio = ROOT.TH1D("h_edep_over_truth", "E_{dep}^{jet}/E_{truth}^{jet};E_{dep}/E_{truth};Jets", 100, 0, 2)
    h_pz = {k: ROOT.TH1D(f"h_pz_{k}", f"{k};P_{{||}}^{{reco-truth}} [GeV];Events", 120, -120, 120) for k in ("current", "drop_unmatched")}
    counts = {"events": 0, "selected": 0, "forward_jets": 0, "matched": 0, "unmatched": 0}

    with open(args.csv, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=(
            "event", "jet", "pt_GeV", "eta", "phi", "truth_match", "dr_truth",
            "E_dep_GeV", "E_truth_GeV", "E_dep_over_E_truth",
            "jet_pileup_pt_GeV", "jet_pileup_e_GeV",
        ))
        writer.writeheader()
        for arrays in proc.iter_arrays(columns):
            for event in range(len(arrays["mu_pt"])):
                counts["events"] += 1
                met = build_met_inputs(arrays, event)
                mu = build_leptons(arrays, event)
                pt = float(arrays["tboson_pt"][event]) * GEV
                eta = float(arrays["tboson_eta"][event])
                phi = float(arrays["tboson_phi"][event])
                truth_met = build_truth_met(arrays, event)
                if (not met.has_consistent_size() or mu.count < 1 or
                    not np.isfinite(pt) or pt == 0 or not np.isfinite(eta) or
                    not np.isfinite(phi) or not (abs(eta) < ETA_MAX) or
                    truth_met is None or truth_met.vector is None):
                    continue
                clusters = build_clusters(arrays, event)
                if not clusters.has_consistent_size():
                    continue
                counts["selected"] += 1
                truth = truth_met.vector
                w = TransverseVector(pt * np.cos(phi), pt * np.sin(phi))
                current = met.calculate_met()
                forward = met.forward_jet_indices(ETA_MIN, ETA_MAX)
                reco_eta = awkward_to_numpy(arrays["jet_eta"][event])
                reco_phi = awkward_to_numpy(arrays["jet_phi"][event])
                truth_eta = awkward_to_numpy(arrays["tjet_eta"][event])
                truth_phi = awkward_to_numpy(arrays["tjet_phi"][event])
                truth_e = awkward_to_numpy(arrays["tjet_e"][event]) * GEV
                pu_pt = awkward_to_numpy(arrays["jet_pileup_pt"][event]) * GEV
                pu_e = awkward_to_numpy(arrays["jet_pileup_e"][event]) * GEV
                cluster_truth = arrays["jet_cluster_truthE"][event]
                pu_indices = []
                for j in forward:
                    j = int(j); counts["forward_jets"] += 1
                    reco_i, _ = nearest(met.jets.eta[j], met.jets.phi[j], reco_eta, reco_phi)
                    if reco_i is None:
                        truth_i, match_dr = None, np.nan
                    else:
                        truth_i, match_dr = nearest(reco_eta[reco_i], reco_phi[reco_i], truth_eta, truth_phi)
                    matched = truth_i is not None
                    if reco_i is not None and reco_i < len(cluster_truth):
                        values = awkward_to_numpy(cluster_truth[reco_i])
                        e_dep = float(np.sum(np.maximum(values, 0))) * GEV
                    else:
                        # Some ntuples leave the nested jet-cluster branch empty;
                        # use the equivalent global truth-cluster cone in that case.
                        cdr = np.asarray([
                            dr(met.jets.eta[j], met.jets.phi[j], e, p)
                            for e, p in zip(clusters.eta, clusters.phi)
                        ])
                        mask = (cdr < MATCH_DR) & np.isfinite(clusters.energy["truth"])
                        e_dep = float(np.sum(np.maximum(clusters.energy["truth"][mask], 0)))
                    if matched:
                        counts["matched"] += 1
                        e_truth = float(truth_e[truth_i])
                        ratio = e_dep / e_truth if np.isfinite(e_dep) and e_truth > 0 else np.nan
                        if np.isfinite(ratio):
                            h_ratio.Fill(ratio)
                    else:
                        counts["unmatched"] += 1; pu_indices.append(j)
                        e_truth = ratio = np.nan
                    writer.writerow(dict(event=counts["events"] - 1, jet=j,
                        pt_GeV=float(met.jets.pt[j]), eta=float(met.jets.eta[j]),
                        phi=float(met.jets.phi[j]), truth_match=int(matched),
                        dr_truth=match_dr, E_dep_GeV=e_dep, E_truth_GeV=e_truth,
                        E_dep_over_E_truth=ratio,
                        jet_pileup_pt_GeV=float(pu_pt[reco_i]) if reco_i is not None and reco_i < len(pu_pt) else np.nan,
                        jet_pileup_e_GeV=float(pu_e[reco_i]) if reco_i is not None and reco_i < len(pu_e) else np.nan))
                residual = current - truth
                h_pz["current"].Fill(residual.dot(w.unit()))
                pu_vector = met.jets.visible_vector(pu_indices)
                h_pz["drop_unmatched"].Fill((current + pu_vector - truth).dot(w.unit()))

    out = ROOT.TFile(args.output, "RECREATE")
    h_ratio.Write(); h_pz["current"].Write(); h_pz["drop_unmatched"].Write(); out.Close()
    print(counts)
    print("unmatched fraction =", counts["unmatched"] / counts["forward_jets"] if counts["forward_jets"] else np.nan)


if __name__ == "__main__":
    main()
