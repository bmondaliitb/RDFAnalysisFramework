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
ANGLE_BINS = np.linspace(-np.pi, np.pi, 25)
FORWARD_SUM_PT_BINS = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]
FORWARD_CLUSTER_BINS = [-0.5, 0.5, 10.5, 25.5, 50.5, 100.5, 200.5, 400.5, 800.5, 1600.5]
FORWARD_JET_BINS = np.arange(-0.5, 10.5, 1.0)
MT_BINS = np.linspace(0, 200, 41)
TRUTH_MT_BRANCHES = tuple(f"{prefix}_{field}" for prefix in ("tlep_bare", "tneutrino")
                          for field in ("pt", "phi", "pdgId"))
MIN_MEAN_ENTRIES = 10
TREE_RECOIL_SCALES = ("truth", "em", "lcw", "ml", "pfoEM")


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
            "selected", "with_selected_forward_jets", "valid_truth_mT", "invalid_truth_mT"], 0)
        self.event_values = {key: [] for key in ("qT", "avgMu", "nPrimVtx",
            "deltaPhi_W_mu", "forward_sum_pt_em", "n_forward_clusters_em", "n_forward_jets", "mT_truth")}
        names = ["standard_met", *(f"tree_{scale}" for scale in TREE_RECOIL_SCALES)]
        for scale in METConfig.CLUSTER_SCALES:
            names += [f"central_clusters_{scale}", f"all_clusters_{scale}",
                f"keep_jets_forward_clusters_{scale}",
                f"replace_forward_jets_with_all_forward_clusters_{scale}"]
        self.values = {}
        for name in names:
            self.values[name] = {key: [] for key in (
                "u_pt", "u_par", "u_perp", "balance", "residual_x", "residual_y",
                "met_residual_x", "met_residual_y", "met", "mT", "mT_residual")}

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
            for scale in TREE_RECOIL_SCALES:
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
            # Use the leading muon as the W lepton when several are present.
            lepton_index = int(np.argmax(leptons.pt))
            lepton_pt = leptons.pt[lepton_index]
            lepton_phi = leptons.phi[lepton_index]
            self.event_values["deltaPhi_W_mu"].append(
                met_inputs_common.wrap_delta_phi(w_phi - lepton_phi))
            forward_sum_pt, forward_count = forward_activity(clusters, args)
            self.event_values["forward_sum_pt_em"].append(forward_sum_pt)
            self.event_values["n_forward_clusters_em"].append(forward_count)
            self.event_values["n_forward_jets"].append(n_forward_jets)
            truth_mt = truth_transverse_mass(array, event)
            self.event_values["mT_truth"].append(truth_mt)
            self.event_counts["valid_truth_mT" if np.isfinite(truth_mt) else "invalid_truth_mT"] += 1
            for name, u in recoils.items():
                u_par = u.x * cos_phi + u.y * sin_phi
                u_perp = -u.x * sin_phi + u.y * cos_phi
                # Ideal recoil is -pT(W), so u_parallel + qT and u_perp vanish.
                missing = -u - lepton_sum
                met_residual = missing - truth_met.vector
                mt_squared = 2.0 * lepton_pt * missing.pt * (
                    1.0 - np.cos(lepton_phi - missing.phi))
                self.values[name]["met"].append(missing.pt)
                reco_mt = np.sqrt(max(0.0, mt_squared))
                self.values[name]["mT"].append(reco_mt)
                self.values[name]["mT_residual"].append(reco_mt - truth_mt)
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
        u_par = u_x * np.cos(tboson_phi) + u_y * np.sin(tboson_phi)
        u_perp = u_y * np.cos(tboson_phi) - u_x * np.sin(tboson_phi)

        recoil["u_pt_truth"] = u_pt
        recoil["u_x_truth"] = u_x
        recoil["u_y_truth"] = u_y
        recoil["u_mag_truth"] = np.sqrt(u_x**2 + u_y**2)
        recoil["u_par_truth"] = u_par
        recoil["u_perp_truth"] = u_perp
        recoil["u_par_plus_qT_truth"] = u_par + qT
        recoil["u_perp_truth_alt"] = -u_x * np.sin(tboson_phi) + u_y * np.cos(tboson_phi)

        # EM recoil (slot 11).
        u_pt = data["u_pt.second"][event][11] / 1000.0
        u_phi = data["u_phi.second"][event][11]
        u_x = u_pt * np.cos(u_phi)
        u_y = u_pt * np.sin(u_phi)
        u_par = u_x * np.cos(tboson_phi) + u_y * np.sin(tboson_phi)
        u_perp = u_y * np.cos(tboson_phi) - u_x * np.sin(tboson_phi)

        recoil["u_pt_em"] = u_pt
        recoil["u_x_em"] = u_x
        recoil["u_y_em"] = u_y
        recoil["u_mag_em"] = np.sqrt(u_x**2 + u_y**2)
        recoil["u_par_em"] = u_par
        recoil["u_perp_em"] = u_perp
        recoil["u_par_plus_qT_em"] = u_par + qT
        recoil["u_perp_em_alt"] = -u_x * np.sin(tboson_phi) + u_y * np.cos(tboson_phi)

        # LCW recoil (slot 12).
        u_pt = data["u_pt.second"][event][12] / 1000.0
        u_phi = data["u_phi.second"][event][12]
        u_x = u_pt * np.cos(u_phi)
        u_y = u_pt * np.sin(u_phi)
        u_par = u_x * np.cos(tboson_phi) + u_y * np.sin(tboson_phi)
        u_perp = u_y * np.cos(tboson_phi) - u_x * np.sin(tboson_phi)

        recoil["u_pt_lcw"] = u_pt
        recoil["u_x_lcw"] = u_x
        recoil["u_y_lcw"] = u_y
        recoil["u_mag_lcw"] = np.sqrt(u_x**2 + u_y**2)
        recoil["u_par_lcw"] = u_par
        recoil["u_perp_lcw"] = u_perp
        recoil["u_par_plus_qT_lcw"] = u_par + qT
        recoil["u_perp_lcw_alt"] = -u_x * np.sin(tboson_phi) + u_y * np.cos(tboson_phi)

        # ML recoil (slot 13).
        u_pt = data["u_pt.second"][event][13] / 1000.0
        u_phi = data["u_phi.second"][event][13]
        u_x = u_pt * np.cos(u_phi)
        u_y = u_pt * np.sin(u_phi)
        u_par = u_x * np.cos(tboson_phi) + u_y * np.sin(tboson_phi)
        u_perp = u_y * np.cos(tboson_phi) - u_x * np.sin(tboson_phi)

        recoil["u_pt_ml"] = u_pt
        recoil["u_x_ml"] = u_x
        recoil["u_y_ml"] = u_y
        recoil["u_mag_ml"] = np.sqrt(u_x**2 + u_y**2)
        recoil["u_par_ml"] = u_par
        recoil["u_perp_ml"] = u_perp
        recoil["u_par_plus_qT_ml"] = u_par + qT
        recoil["u_perp_ml_alt"] = -u_x * np.sin(tboson_phi) + u_y * np.cos(tboson_phi)

        # recoil pfoEM(slot 19).
        u_pt = data["u_pt.second"][event][19] / 1000.0
        u_phi = data["u_phi.second"][event][19]
        u_x = u_pt * np.cos(u_phi)
        u_y = u_pt * np.sin(u_phi)
        u_par = u_x * np.cos(tboson_phi) + u_y * np.sin(tboson_phi)
        u_perp = u_y * np.cos(tboson_phi) - u_x * np.sin(tboson_phi)

        recoil["u_pt_pfoEM"] = u_pt
        recoil["u_x_pfoEM"] = u_x
        recoil["u_y_pfoEM"] = u_y
        recoil["u_mag_pfoEM"] = np.sqrt(u_x**2 + u_y**2)
        recoil["u_par_pfoEM"] = u_par
        recoil["u_perp_pfoEM"] = u_perp
        recoil["u_par_plus_qT_pfoEM"] = u_par + qT
        recoil["u_perp_pfoEM_alt"] = -u_x * np.sin(tboson_phi) + u_y * np.cos(tboson_phi)

        return recoil


def truth_transverse_mass(array, event):
    """Bare W-muon + W-neutrino mT in GeV; no reconstructed inputs.

    These branches are assumed to describe the same W decay. Accept scalar or
    singleton values, not arbitrary pairings from multi-particle collections.
    Missing branches fail explicitly; invalid event records return NaN.
    """
    particles = []
    for prefix in ("tlep_bare", "tneutrino"):
        fields = [np.asarray(array[f"{prefix}_{field}"][event], dtype=float)
                  for field in ("pt", "phi", "pdgId")]
        if any(value.size != 1 for value in fields):
            return np.nan
        pt, phi, pdg_id = (float(value.reshape(-1)[0]) for value in fields)
        if not np.isfinite([pt, phi, pdg_id]).all() or pt < 0:
            return np.nan
        particles.append((pt * METConfig.GEV, phi, pdg_id))
    (mu_pt, mu_phi, mu_id), (nu_pt, nu_phi, nu_id) = particles
    if abs(mu_id) != 13 or abs(nu_id) != 14 or mu_id * nu_id >= 0:
        return np.nan
    # Stable equivalent of sqrt(2*pT_mu*pT_nu*(1-cos(delta_phi))).
    return float(2 * np.sqrt(mu_pt) * np.sqrt(nu_pt) * abs(np.sin((mu_phi - nu_phi)/2)))


def mass_migration(truth, reco, edges):
    """Counts and P(reco bin | truth bin), including both ROOT flow bins.

    Rows index truth, columns reconstruction. Empty rows remain NaN in the
    normalized matrix. ROOT convention: values at the final edge overflow.
    """
    truth, reco = np.asarray(truth), np.asarray(reco)
    if truth.ndim != 1 or truth.shape != reco.shape:
        raise ValueError("Migration inputs must be aligned one-dimensional arrays")
    finite = np.isfinite(truth) & np.isfinite(reco)
    size = len(edges) + 1
    counts = np.zeros((size, size), dtype=float)
    np.add.at(counts, (np.searchsorted(edges, truth[finite], side="right"),
                      np.searchsorted(edges, reco[finite], side="right")), 1)
    totals = counts.sum(axis=1)
    normalized = np.full_like(counts, np.nan)
    np.divide(counts, totals[:, None], out=normalized, where=totals[:, None] > 0)
    return counts, normalized, totals


def save_mass_diagnostics(event_values, values):
    import ROOT

    truth = event_values["mT_truth"]
    hist = make_numpy_hist("h_mT_truth", "Bare truth W decay;m_{T}^{truth} [GeV];Events",
                          MT_BINS, truth[np.isfinite(truth)])
    hist.GetXaxis().SetTitle("m_{T}^{truth} [GeV]")
    hist.Write()
    baseline = values["standard_met"]["mT_residual"]
    for name, obs in values.items():
        residual = obs["mT_residual"]
        hist = make_numpy_hist(f"h_mT_residual_{name}",
                              f"{name};m_{{T}}^{{reco}}-m_{{T}}^{{truth}} [GeV];Events",
                              120, -150, 150, residual[np.isfinite(residual)])
        hist.GetXaxis().SetTitle("m_{T}^{reco}-m_{T}^{truth} [GeV]")
        hist.Write()
        for variable, edges, title in (("qT", QT_BINS, "p_{T}^{W} [GeV]"),
                                       ("nPrimVtx", VERTEX_BINS, "N_{PV}")):
            x = event_values[variable]
            for key, y in (("mT_residual", residual),
                           ("paired_delta_abs_mT_residual", np.abs(residual)-np.abs(baseline))):
                means, errors, entries = calculate_binned_mean(x, y, edges)
                suffix = f"{key}_vs_{variable}_{name}"
                write_diagnostic(f"h_mean_{suffix}", f"{name};{title};Mean {key} [GeV]", edges, means, errors)
                write_diagnostic(f"h_mean_entries_{suffix}", f"{name};{title};Finite events", edges, entries)
                write_diagnostic(f"h_mean_valid_{suffix}", f"{name};{title};Valid mean", edges, np.isfinite(means).astype(float))
            _, width = calculate_sigma_iqr_in_x_bins(x, residual, edges)
            write_diagnostic(f"h_sigma_mT_residual_vs_{variable}_{name}",
                             f"{name};{title};#sigma_{{IQR}}(m_{{T}}^{{reco}}-m_{{T}}^{{truth}}) [GeV]",
                             edges, width / 2)
        counts, normalized, totals = mass_migration(truth, obs["mT"], MT_BINS)
        for suffix, contents in (("counts", counts), ("normalized", normalized)):
            hist = ROOT.TH2D(f"h_mT_truth_vs_reco_{suffix}_{name}",
                             f"{name};m_{{T}}^{{truth}} [GeV];m_{{T}}^{{reco}} [GeV]",
                             len(MT_BINS)-1, MT_BINS, len(MT_BINS)-1, MT_BINS)
            # Include flow cells so large reconstructed tails do not disappear
            # from row normalization. Errors are Poisson / marginal binomial.
            for i in range(len(totals)):
                for j in range(len(totals)):
                    value = contents[i, j]
                    error = (np.sqrt(value) if suffix == "counts" else
                             np.sqrt(value*(1-value)/totals[i]) if totals[i] > 0 else np.nan)
                    hist.SetBinContent(i, j, float(value))
                    hist.SetBinError(i, j, float(error))
            hist.SetEntries(float(counts.sum()))
            hist.Write()
        for suffix, contents in (("entries", totals), ("valid", (totals > 0).astype(float))):
            hist = make_hist_from_bin_contents(f"h_mT_migration_truth_{suffix}_{name}",
                                               f"{name};m_{{T}}^{{truth}} [GeV];{suffix}",
                                               MT_BINS, contents[1:-1])
            hist.SetBinContent(0, float(contents[0]))
            hist.SetBinContent(len(MT_BINS), float(contents[-1]))
            hist.Write()


def forward_activity(clusters, args):
    """Fixed EM activity before lepton-cone/UE corrections, in GeV."""
    energy = clusters.energy["em"]
    mask = clusters.in_abs_eta_range(args.eta_min, args.eta_max)
    mask &= np.isfinite(energy) & np.isfinite(clusters.phi) & (energy >= 0)
    energy, eta = energy[mask], clusters.eta[mask]
    pt = energy / np.cosh(eta)
    selected = (energy if args.threshold_on == "momentum" else pt) >= args.cluster_threshold
    return float(np.sum(pt[selected])), int(np.count_nonzero(selected))


def calculate_binned_mean(x, y, edges, min_entries=MIN_MEAN_ENTRIES):
    """Unweighted mean and sample standard error; final edge is inclusive."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Mean inputs must be aligned one-dimensional event arrays")
    means = np.full(len(edges) - 1, np.nan)
    errors = np.full_like(means, np.nan)
    counts = np.zeros(len(means), dtype=int)
    finite = np.isfinite(x) & np.isfinite(y)
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = finite & (x >= low) & (x <= high if i == len(means)-1 else x < high)
        sample = y[selected]
        counts[i] = len(sample)
        if len(sample) >= max(2, min_entries):
            means[i] = np.mean(sample)
            errors[i] = np.std(sample, ddof=1) / np.sqrt(len(sample))
    return means, errors, counts


def angular_flatness(means, reference_counts, valid):
    """Count-weighted angular variation and offset, NOT recoil resolution."""
    valid = np.asarray(valid) & np.isfinite(means) & (reference_counts > 0)
    if np.count_nonzero(valid) < 2:
        return np.nan, np.nan
    offset = np.average(means[valid], weights=reference_counts[valid])
    return np.sqrt(np.average((means[valid] - offset)**2,
                             weights=reference_counts[valid])), offset


def write_diagnostic(name, title, edges, contents, errors=None):
    # The shared utility maps NaNs to zero. Restore NaNs so invalid estimates
    # cannot be mistaken for perfect balance; companion masks aid ROOT plotting.
    hist = make_hist_from_bin_contents(name, title, edges, contents)
    for i, value in enumerate(contents, 1):
        hist.SetBinContent(i, float(value))
        hist.SetBinError(i, float(errors[i-1]) if errors is not None else 0.0)
    hist.Write()


def save_bias_diagnostics(event_values, values):
    axes = {
        "qT": (QT_BINS, "p_{T}^{W} [GeV]"),
        "avgMu": (MU_BINS, "<#mu>"),
        "nPrimVtx": (VERTEX_BINS, "N_{PV}"),
        "deltaPhi_W_mu": (ANGLE_BINS, "#Delta#phi(W,#mu) [rad]"),
        "forward_sum_pt_em": (FORWARD_SUM_PT_BINS, "Forward EM #Sigma p_{T} [GeV]"),
        "n_forward_clusters_em": (FORWARD_CLUSTER_BINS, "Forward EM cluster multiplicity"),
        "n_forward_jets": (FORWARD_JET_BINS, "Selected forward jet multiplicity"),
    }
    n_events = len(event_values["qT"])
    if any(len(x) != n_events for x in event_values.values()) or any(
            len(y) != n_events for obs in values.values() for y in obs.values()):
        raise ValueError("Scenario and event arrays must be aligned")
    angle = event_values["deltaPhi_W_mu"]
    # Common support and fixed counts prevent scenario-dependent reweighting.
    common = np.isfinite(angle)
    for obs in values.values():
        common &= np.isfinite(obs["u_perp"])
    reference_counts = np.histogram(angle[common], ANGLE_BINS)[0]
    write_diagnostic("h_angular_reference_entries", "Angular reference;#Delta#phi(W,#mu) [rad];Events",
                     ANGLE_BINS, reference_counts)
    angular = {name: calculate_binned_mean(angle[common], obs["u_perp"][common], ANGLE_BINS)
               for name, obs in values.items()}
    valid = reference_counts >= MIN_MEAN_ENTRIES
    for means, _, _ in angular.values():
        valid &= np.isfinite(means)
    write_diagnostic("h_angular_flatness_valid", "Common angular support;#Delta#phi(W,#mu) [rad];Valid",
                     ANGLE_BINS, valid.astype(float))
    hist = make_numpy_hist("h_deltaPhi_W_mu", "W and leading muon;#Delta#phi(W,#mu) [rad];Events",
                          ANGLE_BINS, angle[np.isfinite(angle)])
    hist.GetXaxis().SetTitle("#Delta#phi(W,#mu) [rad]")
    hist.Write()
    baseline = values["standard_met"]
    for name, obs in values.items():
        jobs = [(key, variable, obs[key]) for variable in axes
                for key in (("balance", "u_perp") if variable in
                            ("qT", "avgMu", "nPrimVtx", "deltaPhi_W_mu") else ("balance",))]
        jobs.append(("abs_balance", "forward_sum_pt_em", np.abs(obs["balance"])))
        for key, variables in (("balance", ("nPrimVtx", "qT")), ("u_perp", ("nPrimVtx",))):
            delta = np.abs(obs[key]) - np.abs(baseline[key])
            jobs.extend((f"paired_delta_abs_{key}", variable, delta) for variable in variables)
        for key, variable, y in jobs:
            edges, title = axes[variable]
            means, errors, counts = calculate_binned_mean(event_values[variable], y, edges)
            suffix = f"{key}_vs_{variable}_{name}"
            write_diagnostic(f"h_mean_{suffix}", f"{name};{title};Mean {key} [GeV]", edges, means, errors)
            write_diagnostic(f"h_mean_entries_{suffix}", f"{name};{title};Finite events", edges, counts)
            write_diagnostic(f"h_mean_valid_{suffix}", f"{name};{title};Valid mean", edges, np.isfinite(means).astype(float))
        flatness, offset = angular_flatness(angular[name][0], reference_counts, valid)
        for key, value in (("angular_flatness_u_perp", flatness), ("angular_offset_u_perp", offset)):
            write_diagnostic(f"h_{key}_{name}", f"{name};Summary;{key} [GeV]", [0, 1], [value])


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
        settings["scenario_met"] = "missing transverse momentum = -recoil - visible lepton sum"
        settings["transverse_mass"] = "sqrt(2 * leading_muon_pt * scenario_met * (1 - cos(delta_phi)))"
        settings["mean_statistics"] = "Unweighted mean; sample SEM; >=10 finite events; invalid bins NaN with entries/valid companions"
        settings["angular_flatness"] = "Count-weighted RMS of bin means about offset on common finite support; >=2 valid bins; not resolution; no noise subtraction"
        settings["forward_activity"] = "Fixed EM E/cosh(eta), nonnegative finite energies, configured threshold and eta range, before cone/UE correction"
        settings["truth_mT_reference"] = "Bare truth muon tlep_bare_{pt,phi,pdgId} and tneutrino_{pt,phi,pdgId}; pt assumed MeV, converted to GeV; scalar/singleton W daughters; |PDG|=13,14 with opposite signs. Common W parent assumed from ntuple semantics, not independently verified; no reco-truth matching imposed. Invalid truth entries excluded only from mass diagnostics."
        settings["mT_migration"] = "X=truth, Y=reco; rows normalized over all reco bins INCLUDING flows; final edge overflows (ROOT convention); empty rows NaN with validity/count histograms."
        settings["tree_truth_mT"] = "Existing tree_truth scenario still uses reconstructed leptons; h_mT_truth is the independent daughter-level truth reference."
        settings["ml_energy_selection"] = "Existing scenarios preserved: all_clusters_ml combines central fCluster_MLE (MeV/1000) and forward cluster_e_ML_forward (GeV); other ml subsets use fCluster_MLE; ml_forward uses cluster_e_ML_forward."
        settings["zero_qT_projection"] = "Uses stored truth W phi even at qT=0; direction there is conventional"
        ROOT.TNamed("settings", json.dumps(settings)).Write()
        cutflow = ROOT.TH1D("h_cutflow", "Event selection", len(counts), 0, len(counts))
        for index, (name, count) in enumerate(counts.items(), start=1):
            cutflow.GetXaxis().SetBinLabel(index, name)
            cutflow.SetBinContent(index, count)
        cutflow.Write()

        save_bias_diagnostics(event_values, values)
        save_mass_diagnostics(event_values, values)

        for name, observables in values.items():
            for key in ("u_pt", "balance", "u_perp", "residual_x", "residual_y",
                        "met_residual_x", "met_residual_y", "met", "mT"):
                low, high = (0, 300) if key in ("u_pt", "met", "mT") else (-150, 150)
                axis_title = {"met": "E_{T}^{miss}", "mT": "m_{T}"}.get(key, key)
                make_numpy_hist(f"h_{key}_{name}", f"{name};{axis_title} [GeV];Events",
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
    branches += list(TRUTH_MT_BRANCHES)
    branches += ["tboson_pt", "tboson_eta", "tboson_phi", "avgMu", "nPrimVtx"]
    branches += [METConfig.TRUTH_MET_PT_BRANCH, METConfig.TRUTH_MET_PHI_BRANCH]
    branches += ["tu_pt.second", "tu_phi.second", "u_pt.second", "u_phi.second"]

    analysis = HadRecoil(input_files, args.tree, args.n_events, args)
    analysis.set_branches(branches)
    analysis.execute()
    analysis.save_histograms(args.output)


if __name__ == "__main__":
    main()
