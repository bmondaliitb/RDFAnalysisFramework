#!/usr/bin/env python3

import sys
import os

from pathlib import Path
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import ROOT

from rdf_analysis.core import NtupleProcessor, awkward_to_numpy
from rdf_analysis.stats.fits import *
from rdf_analysis.stats.histograms import *



class Config:
    """All analysis configuration in one place."""

    DEBUG = False
    
    # Z boson
    Z_MASS = 91.1876
    Z_WINDOW = 20.0
    
    # Jet selection
    JET_PT_MIN = 10.0
    JET_Z_DPHI_MIN = 2.8
    SECOND_JET_ABS_PT_MAX = 12.0
    SECOND_JET_REF_PT_FRACTION_MAX = 0.10

    # Matching
    TRUTH_MATCH_DR = 0.4
    
    # Units
    GEV = 0.001
    JET_ENERGY_HIST_MAX = 4000.0
    
    # Energy scales
    ENERGY_SCALES = (
        ("cluster", "cluster E", "jet_clusterE", 1.0),
        ("truth", "cluster truth E", "jet_cluster_truthE", GEV),
        ("raw", "cluster raw E", "jet_cluster_rawE", 1.0),
        ("cal", "cluster calibrated E", "jet_cluster_calE", 1.0),
        ("ml", "cluster ML E", "jet_cluster_MLE", 1.0),
    )
    
    CLUSTER_PT_SCALES = (
        ("raw", "cluster raw E"),
        ("cal", "cluster calibrated E"),
        ("truth", "cluster truth E"),
        ("ml", "cluster ML E"),
    )

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Jet:
    """Single reconstructed jet collection."""
    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    energy: np.ndarray

    @property
    def to_array(self):
        jet_array = []
        for i in range(len(self.pt)):
            jet_array.append(Jet(pt=self.pt[i],eta=self.eta[i],phi=self.phi[i], energy=self.energy[i]) )

        return jet_array

    @property
    def leading_jet(self):
        if len(self.pt) == 0:
            raise ValueError("Cannot get leading jet from empty jet collection.")

        leading_jet_index = self.leading_jet_index

        return Jet(
            pt=self.pt[leading_jet_index],
            eta=self.eta[leading_jet_index],
            phi=self.phi[leading_jet_index],
            energy=self.energy[leading_jet_index],
        )

    @property
    def leading_jet_index(self):
        return np.argmax(self.pt)

    @property
    def subleading_jet(self):
        if len(self.pt) < 2:
            raise ValueError("Cannot get subleading jet from fewer than 2 jets.")

        # argsort sorts ascending, so [-2] is the second-highest pt
        subleading_jet_index = np.argsort(self.pt)[-2]

        return Jet(
            pt=self.pt[subleading_jet_index],
            eta=self.eta[subleading_jet_index],
            phi=self.phi[subleading_jet_index],
            energy=self.energy[subleading_jet_index],
        )

    def is_leading_jet_truth_matched(self, truth_jets: "Jet") -> bool:
        """Check if leading jet is matched to any truth jet."""
        leading_jet = self.leading_jet
        dr=[]
        passed_dr=False
        for truth_jet in truth_jets.to_array:
            dr.append(delta_r(leading_jet.eta, leading_jet.phi, truth_jet.eta, truth_jet.phi))

        dr=np.array(dr)
        return np.any(dr < Config.TRUTH_MATCH_DR)

    def is_subleading_jet_truth_matched(self, truth_jets: "Jet") -> bool:
        """Check if subleading jet is matched to any truth jet."""
        subleading_jet = self.subleading_jet
        dr=[]
        for truth_jet in truth_jets.to_array:
            dr.append(delta_r(subleading_jet.eta, subleading_jet.phi, truth_jet.eta, truth_jet.phi))

        dr=np.array(dr)
        return np.any(dr < Config.TRUTH_MATCH_DR)

    #def get_truth_jet_matched_to_leading_jet(self, truth_jets: "Jet") -> Optional["Jet"]:
    #    """Return the truth jet matched to the leading jet, if any."""
    #    leading_jet = self.leading_jet
    #    truth_matched_jets = []
    #    for truth_jet in truth_jets.to_array:
    #        if delta_r(leading_jet.eta, leading_jet.phi, truth_jet.eta, truth_jet.phi) < Config.TRUTH_MATCH_DR:
    #            truth_matched_jets.append(truth_jet)

    #    if len(truth_matched_jets) == 0:
    #        return None
    #    if len(truth_matched_jets) >= 1:
    #        # find the closest pt matched one
    #        closest_jet = min(truth_matched_jets, key=lambda jet: abs(jet.pt - leading_jet.pt))
    #        return closest_jet

    def get_truth_jet_matched_to_leading_jet(
            self,
            truth_jets: "Jet",
    ) -> Optional["Jet"]:
        leading_jet = self.leading_jet
        candidates = []

        for truth_jet in truth_jets.to_array:
            dr = delta_r(
                leading_jet.eta,
                leading_jet.phi,
                truth_jet.eta,
                truth_jet.phi,
            )

            if dr < Config.TRUTH_MATCH_DR:
                pt_difference = abs(truth_jet.pt - leading_jet.pt)
                candidates.append((dr, pt_difference, truth_jet))

        if not candidates:
            return None

        return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]

@dataclass
class Lepton:
    """Collection of leptons in event."""
    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray

    @property
    def count(self) -> int:
        return len(self.pt)

@dataclass
class Cluster:
    """Cluster data for all energy scales."""
    eta: np.ndarray
    phi: np.ndarray
    energy_em: np.ndarray
    energy_ml: np.ndarray
    energy_lcw: np.ndarray
    energy_truth: np.ndarray

    @property
    def jet_energy_em(self):
        """ Construct jet energy from cluster"""
        return np.sum(self.energy_em, where=self.energy_em > 0)

    @property
    def jet_energy_lcw(self):
        return np.sum(self.energy_lcw, where=self.energy_lcw > 0)

    @property
    def jet_energy_ml(self):
        return np.sum(self.energy_ml, where=self.energy_ml > 0)

    @property
    def jet_energy_truth(self):
        return np.sum(self.energy_truth, where=self.energy_truth > 0)

# ============================================================================
# PHYSICS UTILITIES
# ============================================================================

def delta_phi(phi1: float, phi2: float) -> float:
    """Delta-phi in [-pi, pi]."""
    return ROOT.TVector2.Phi_mpi_pi(phi1 - phi2)

def dilepton_mass(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> float:
    """Invariant mass of two leptons (massless)."""
    if len(pt) < 2:
        return np.nan
    deta = eta[0] - eta[1]
    dphi = delta_phi(phi[0], phi[1])
    mass2 = 2.0 * pt[0] * pt[1] * (np.cosh(deta) - np.cos(dphi))
    return np.sqrt(max(mass2, 0.0))

def z_transverse_vector(lep_pt: np.ndarray, lep_phi: np.ndarray) -> Tuple[float, float, float]:
    """Z boson transverse momentum vector."""
    lep_pt = np.asarray(lep_pt[:2], dtype=float)
    lep_phi = np.asarray(lep_phi[:2], dtype=float)
    z_x = np.sum(lep_pt * np.cos(lep_phi))
    z_y = np.sum(lep_pt * np.sin(lep_phi))
    z_pt = np.hypot(z_x, z_y)
    return z_x, z_y, z_pt

def z_phi(lep_pt: np.ndarray, lep_phi: np.ndarray) -> float:
    """Z boson azimuthal angle."""
    z_x, z_y, z_pt = z_transverse_vector(lep_pt, lep_phi)
    if z_pt == 0.0:
        return np.nan
    return np.arctan2(z_y, z_x)

def delta_phi_to_z(jet_phi: float, lep_pt: np.ndarray, lep_phi: np.ndarray) -> float:
    """Absolute delta-phi between jet and Z."""
    event_z_phi = z_phi(lep_pt, lep_phi)
    if not np.isfinite(event_z_phi):
        return np.nan
    return abs(delta_phi(jet_phi, event_z_phi))

def project_jet_on_z_axis(jet_pt: float, jet_phi: float, lep_pt: np.ndarray, lep_phi: np.ndarray) -> float:
    """Jet pT projected onto Z direction."""
    z_x, z_y, z_pt = z_transverse_vector(lep_pt, lep_phi)
    if z_pt == 0.0:
        return np.nan
    jet_x = jet_pt * np.cos(jet_phi)
    jet_y = jet_pt * np.sin(jet_phi)
    return abs((jet_x * z_x + jet_y * z_y) / z_pt)


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """Delta-R distance."""
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.hypot(deta, dphi)

def calculate_pt_from_energy(energy, eta):
    """
    Calculates jet pT from Energy and Pseudorapidity (eta).
    Assumes massless jet approximation.
    """
    return energy / np.cosh(eta)

def compare_energy_scales(sigma_reco, sigma_truth):
    return np.sqrt(np.abs(sigma_reco**2 - sigma_truth**2))

# ============================================================================
# EVENT PROCESSOR
# ============================================================================

class ZJetsAnalysis:

    def __init__(self, input_files: List[str], tree_name: str, output_path: str, max_events: int = -1):
        self.processor = NtupleProcessor(input_files, tree_name, max_events)
        self.output_path = output_path
        self.histograms = {}
        self.event_counts = {
            "total": 0,
            "nmu_gt_1": 0,
            "has_jets": 0,
            "z_mass_window": 0,
            "delta_phi_cut": 0,
            "subleading_jet_cut": 0,
            "truth_matched": 0
        }

    def run(self):
        """Execute the full analysis."""
        print(f"Starting analysis of Z+jets events...")
        
        columns = [
            "mu_pt", "mu_eta", "mu_phi",
            "jet_pt", "jet_eta", "jet_phi",
            "tjet_pt", "tjet_eta", "tjet_phi", "tjet_e",
            "jet_cluster_eta", "jet_cluster_phi",
            "jet_clusterE", "jet_cluster_truthE", "jet_cluster_rawE", "jet_cluster_calE", "jet_cluster_MLE",
        ]

        # histograms
        leading_jet_pt=[]
        z_pt=[]
        truth_jet_pt=[]
        truth_jet_energy=[]
        # pt
        leading_jet_from_cluster_pt_em = []
        leading_jet_from_cluster_pt_lcw = []
        leading_jet_from_cluster_pt_ml = []
        leading_jet_from_cluster_pt_truth = []
        # pt_ref
        pt_ref_from_cluster_pt_em = []
        pt_ref_from_cluster_pt_lcw = []
        pt_ref_from_cluster_pt_ml = []
        pt_ref_from_cluster_pt_truth = []
        # energy
        leading_jet_from_cluster_energy_em = []
        leading_jet_from_cluster_energy_lcw = []
        leading_jet_from_cluster_energy_ml = []
        leading_jet_from_cluster_energy_truth = []

        # pTref
        pt_ref_list = []

        for arrays in self.processor.iter_arrays(columns):
            for (
                mu_pt, mu_eta, mu_phi,
                jet_pt, jet_eta, jet_phi,
                tjet_pt, tjet_eta, tjet_phi, tjet_e,
                jet_cluster_eta, jet_cluster_phi,
                jet_clusterE, jet_cluster_truthE, jet_cluster_rawE, jet_cluster_calE, jet_cluster_MLE,
            ) in zip(
                arrays["mu_pt"], arrays["mu_eta"], arrays["mu_phi"],
                arrays["jet_pt"], arrays["jet_eta"], arrays["jet_phi"],
                arrays["tjet_pt"], arrays["tjet_eta"], arrays["tjet_phi"], arrays["tjet_e"],
                arrays["jet_cluster_eta"], arrays["jet_cluster_phi"],
                arrays["jet_clusterE"], arrays["jet_cluster_truthE"], arrays["jet_cluster_rawE"],
                arrays["jet_cluster_calE"], arrays["jet_cluster_MLE"],
            ):
                if self.event_counts["total"] % 10000 == 0:
                    print(f"  Events processed: {self.event_counts['total']}")

                self.event_counts["total"] += 1

                leptons = Lepton(
                    pt=awkward_to_numpy(mu_pt)*Config.GEV,
                    eta=awkward_to_numpy(mu_eta),
                    phi=awkward_to_numpy(mu_phi),
                )
                
                reco_jets = Jet(
                    pt=awkward_to_numpy(jet_pt) * Config.GEV,
                    eta=awkward_to_numpy(jet_eta),
                    phi=awkward_to_numpy(jet_phi),
                    energy=np.zeros(len(jet_pt)),
                )
                
                truth_jets = Jet(
                    pt=awkward_to_numpy(tjet_pt) * Config.GEV,
                    eta=awkward_to_numpy(tjet_eta),
                    phi=awkward_to_numpy(tjet_phi),
                    energy=awkward_to_numpy(tjet_e) * Config.GEV,
                )

                ## We have structured data, now do the physics
                ### n Muons > 1
                if not leptons.count>1: continue
                self.event_counts["nmu_gt_1"] += 1
                if not len(reco_jets.to_array) > 0: continue
                self.event_counts["has_jets"] += 1
                dilep_mass = dilepton_mass(leptons.pt, leptons.eta, leptons.phi)
                if not ((Config.Z_MASS - 20)< dilep_mass and (dilep_mass < (Config.Z_MASS + 20)) ): continue
                self.event_counts["z_mass_window"] += 1

                # delta phi (leading jet and Z) > 2.8
                if delta_phi_to_z(reco_jets.leading_jet.phi, leptons.pt, leptons.phi) <= 2.8: continue
                self.event_counts["delta_phi_cut"] += 1

                # calculate pTZ, pTref
                pt_ref = project_jet_on_z_axis(reco_jets.leading_jet.pt, reco_jets.leading_jet.phi,
                                               leptons.pt, leptons.phi)
                pt_ref_list.append(pt_ref)

                # subleading jet pt cut
                if len(reco_jets.to_array)>1:
                    if not (reco_jets.subleading_jet.pt < max(12, pt_ref*Config.SECOND_JET_REF_PT_FRACTION_MAX)): continue
                self.event_counts["subleading_jet_cut"] += 1
                # is leading jet truth matched?
                is_leading_jet_truth_matched = reco_jets.is_leading_jet_truth_matched(truth_jets)

                if not is_leading_jet_truth_matched: continue
                self.event_counts["truth_matched"] += 1

                if Config.DEBUG:
                    if is_leading_jet_truth_matched:
                        print("[Info]:: leading jet truth matched")
                    #is_subleading_jet_truth_matched = reco_jets.is_subleading_jet_truth_matched(truth_jets)
                    #if is_subleading_jet_truth_matched:
                    #    print("[Info]:: subleading jet truth matched")

                # clusters corresponding to the leading jet
                cluster_leading_jet = Cluster(
                    energy_truth=awkward_to_numpy(jet_cluster_truthE[reco_jets.leading_jet_index]),
                    energy_em=awkward_to_numpy(jet_cluster_rawE[reco_jets.leading_jet_index]),
                    energy_lcw=awkward_to_numpy(jet_cluster_calE[reco_jets.leading_jet_index]),
                    energy_ml=awkward_to_numpy(jet_cluster_MLE[reco_jets.leading_jet_index]),
                    eta=awkward_to_numpy(jet_cluster_eta[reco_jets.leading_jet_index]),
                    phi=awkward_to_numpy(jet_cluster_phi[reco_jets.leading_jet_index]),
                )

                # fill variables for later histograms
                leading_jet_pt.append(reco_jets.leading_jet.pt)
                truth_jet_pt.append(reco_jets.get_truth_jet_matched_to_leading_jet(truth_jets).pt)
                truth_jet_energy.append(reco_jets.get_truth_jet_matched_to_leading_jet(truth_jets).energy)
                z_pt.append(z_transverse_vector(leptons.pt, leptons.phi)[2])

                # jet pt
                cluster_pt_em = calculate_pt_from_energy(cluster_leading_jet.jet_energy_em, reco_jets.leading_jet.eta)
                leading_jet_from_cluster_pt_em.append(cluster_pt_em)
                cluster_pt_lcw = calculate_pt_from_energy(cluster_leading_jet.jet_energy_lcw, reco_jets.leading_jet.eta)
                leading_jet_from_cluster_pt_lcw.append(cluster_pt_lcw)
                cluster_pt_ml = calculate_pt_from_energy(cluster_leading_jet.jet_energy_ml, reco_jets.leading_jet.eta)
                leading_jet_from_cluster_pt_ml.append(cluster_pt_ml)
                cluster_pt_truth = calculate_pt_from_energy(cluster_leading_jet.jet_energy_truth, reco_jets.leading_jet.eta)
                leading_jet_from_cluster_pt_truth.append(cluster_pt_truth)

                pt_ref_from_cluster_pt_em.append(
                    project_jet_on_z_axis(cluster_pt_em, reco_jets.leading_jet.phi,
                                               leptons.pt, leptons.phi)) # the phi should be same as reco jet phi
                pt_ref_from_cluster_pt_lcw.append(
                    project_jet_on_z_axis(cluster_pt_lcw, reco_jets.leading_jet.phi,
                                               leptons.pt, leptons.phi)) # the phi should be same as reco jet phi
                pt_ref_from_cluster_pt_ml.append(
                    project_jet_on_z_axis(cluster_pt_ml, reco_jets.leading_jet.phi,
                                          leptons.pt, leptons.phi))  # the phi should be same as reco jet phi
                pt_ref_from_cluster_pt_truth.append(
                    project_jet_on_z_axis(cluster_pt_truth, reco_jets.leading_jet.phi,
                                          leptons.pt, leptons.phi))  # the phi should be same as reco jet phi
                # jet energy
                leading_jet_from_cluster_energy_em.append(cluster_leading_jet.jet_energy_em)
                leading_jet_from_cluster_energy_lcw.append(cluster_leading_jet.jet_energy_lcw)
                leading_jet_from_cluster_energy_ml.append(cluster_leading_jet.jet_energy_ml)
                leading_jet_from_cluster_energy_truth.append(cluster_leading_jet.jet_energy_truth)


        # make histograms
        x_bin_edges = get_bins(0, 500, 100)
        y_bin_edges = get_bins(0, 500, 100)
        hist_pt_z_vs_leading_jet = make_numpy_hists_2d("hist_pt_z_vs_leading_jet",
                                                       "", x_bin_edges, z_pt, y_bin_edges, leading_jet_pt)
        hist_pt_z_vs_truth_jet = make_numpy_hists_2d("hist_pt_z_vs_truth_jet",
                                                     "", x_bin_edges, z_pt, y_bin_edges, truth_jet_pt)
        hist_pt_truth_jet_vs_leading_jet_from_cluster_em = make_numpy_hists_2d(
            "hist_pt_truth_jet_vs_leading_jet_from_cluster_em", "", x_bin_edges, truth_jet_pt, y_bin_edges,
            leading_jet_from_cluster_pt_em)

        hist_pt_truth_jet_vs_leading_jet_from_cluster_lcw = make_numpy_hists_2d(
            "hist_pt_truth_jet_vs_leading_jet_from_cluster_lcw", "", x_bin_edges, truth_jet_pt, y_bin_edges,
            leading_jet_from_cluster_pt_lcw)

        hist_pt_truth_jet_vs_leading_jet_from_cluster_ml = make_numpy_hists_2d(
            "hist_pt_truth_jet_vs_leading_jet_from_cluster_ml", "", x_bin_edges, truth_jet_pt, y_bin_edges,
            leading_jet_from_cluster_pt_ml)

        self.histograms["hist_pt_z_vs_leading_jet"] = hist_pt_z_vs_leading_jet
        self.histograms["hist_pt_z_vs_truth_jet"] = hist_pt_z_vs_truth_jet
        self.histograms["hist_pt_truth_jet_vs_leading_jet_from_cluster_em"] = hist_pt_truth_jet_vs_leading_jet_from_cluster_em
        self.histograms["hist_pt_truth_jet_vs_leading_jet_from_cluster_lcw"] = hist_pt_truth_jet_vs_leading_jet_from_cluster_lcw
        self.histograms["hist_pt_truth_jet_vs_leading_jet_from_cluster_ml"] = hist_pt_truth_jet_vs_leading_jet_from_cluster_ml

        # z_pt vs jet pt
        hist_pt_z_vs_leading_jet_from_cluster_em = make_numpy_hists_2d(
            "hist_pt_z_vs_leading_jet_from_cluster_em","", x_bin_edges, z_pt, y_bin_edges, leading_jet_from_cluster_pt_em)
        hist_pt_z_vs_leading_jet_from_cluster_lcw = make_numpy_hists_2d(
            "hist_pt_z_vs_leading_jet_from_cluster_lcw","", x_bin_edges, z_pt, y_bin_edges, leading_jet_from_cluster_pt_lcw)
        hist_pt_z_vs_leading_jet_from_cluster_ml = make_numpy_hists_2d(
            "hist_pt_z_vs_leading_jet_from_cluster_ml","", x_bin_edges, z_pt, y_bin_edges, leading_jet_from_cluster_pt_ml)
        hist_pt_z_vs_leading_jet_from_cluster_truth= make_numpy_hists_2d(
            "hist_pt_z_vs_leading_jet_from_cluster_truth","", x_bin_edges, z_pt, y_bin_edges, leading_jet_from_cluster_pt_truth)
        self.histograms["hist_pt_z_vs_leading_jet_from_cluster_em"] = hist_pt_z_vs_leading_jet_from_cluster_em
        self.histograms["hist_pt_z_vs_leading_jet_from_cluster_lcw"] = hist_pt_z_vs_leading_jet_from_cluster_lcw
        self.histograms["hist_pt_z_vs_leading_jet_from_cluster_ml"] = hist_pt_z_vs_leading_jet_from_cluster_ml
        self.histograms["hist_pt_z_vs_leading_jet_from_cluster_truth"] = hist_pt_z_vs_leading_jet_from_cluster_truth

        # truth jet pT bins calculate the IQR
        truth_jet_pt_bins = get_bins_log(10, 500, 25)

        # define response first
        # response = abs (leading jet projected on pT (Z)/|pT(Z)|^2)
        response_em = np.array(pt_ref_from_cluster_pt_em)/np.array(z_pt) # projection already divide once by z_pt
        response_lcw= np.array(pt_ref_from_cluster_pt_lcw)/np.array(z_pt)
        response_ml= np.array(pt_ref_from_cluster_pt_ml)/np.array(z_pt)
        response_truth = np.array(pt_ref_from_cluster_pt_truth)/np.array(z_pt)

        # get binning between 0 to 1
        response_binning = get_bins(0, 1, 20)
        # save response histograms
        hist_response_em = make_numpy_hist("hist_response_em", "", response_binning, response_em)
        hist_response_lcw = make_numpy_hist("hist_response_lcw", "", response_binning, response_lcw)
        hist_response_ml = make_numpy_hist("hist_response_ml", "", response_binning, response_ml)
        hist_response_truth = make_numpy_hist("hist_response_truth", "", response_binning, response_truth)

        self.histograms["hist_response_em"] = hist_response_em
        self.histograms["hist_response_lcw"] = hist_response_lcw
        self.histograms["hist_response_ml"] = hist_response_ml
        self.histograms["hist_response_truth"] = hist_response_truth

        # store jet energy at different scales for x-bins
        median_em, sigma_em = calculate_sigma_iqr_in_x_bins(truth_jet_pt, response_em, truth_jet_pt_bins)
        median_lcw, sigma_lcw = calculate_sigma_iqr_in_x_bins(truth_jet_pt, response_lcw, truth_jet_pt_bins)
        median_ml, sigma_ml = calculate_sigma_iqr_in_x_bins(truth_jet_pt, response_ml, truth_jet_pt_bins)
        median_truth, sigma_truth = calculate_sigma_iqr_in_x_bins(truth_jet_pt, response_truth, truth_jet_pt_bins)
        # make histograms
        hist_median_em = make_hist_from_bin_contents("hist_mean_em", "", truth_jet_pt_bins, median_em)
        hist_median_lcw = make_hist_from_bin_contents("hist_mean_lcw", "", truth_jet_pt_bins, median_lcw)
        hist_median_ml = make_hist_from_bin_contents("hist_mean_ml", "", truth_jet_pt_bins, median_ml)
        hist_median_truth = make_hist_from_bin_contents("hist_mean_truth", "", truth_jet_pt_bins, median_truth)
        hist_sigma_em = make_hist_from_bin_contents("hist_sigma_em", "", truth_jet_pt_bins, sigma_em/2*median_em)
        hist_sigma_lcw = make_hist_from_bin_contents("hist_sigma_lcw", "", truth_jet_pt_bins, sigma_lcw/2*median_lcw)
        hist_sigma_ml = make_hist_from_bin_contents("hist_sigma_ml", "", truth_jet_pt_bins, sigma_ml/2*median_ml)
        hist_sigma_truth = make_hist_from_bin_contents("hist_sigma_truth", "", truth_jet_pt_bins, sigma_truth/2*median_truth)

        self.histograms["hist_mean_em"] = hist_median_em
        self.histograms["hist_mean_lcw"] = hist_median_lcw
        self.histograms["hist_mean_ml"] = hist_median_ml
        self.histograms["hist_mean_truth"] = hist_median_truth

        self.histograms["hist_sigma_em"] = hist_sigma_em
        self.histograms["hist_sigma_lcw"] = hist_sigma_lcw
        self.histograms["hist_sigma_ml"] = hist_sigma_ml
        self.histograms["hist_sigma_truth"] = hist_sigma_truth

        ## compare performance of different energy scales
        performance_em = compare_energy_scales(sigma_em, sigma_truth)
        performance_lcw = compare_energy_scales(sigma_lcw, sigma_truth)
        performance_ml = compare_energy_scales(sigma_ml, sigma_truth)

        # save them in histograms
        hist_performance_em = make_hist_from_bin_contents("hist_performance_em", "",
                                                          truth_jet_pt_bins, performance_em)
        hist_performance_lcw = make_hist_from_bin_contents("hist_performance_lcw", "",
                                                           truth_jet_pt_bins, performance_lcw)
        hist_performance_ml = make_hist_from_bin_contents("hist_performance_ml", "",
                                                          truth_jet_pt_bins, performance_ml)

        self.histograms["hist_performance_em"] = hist_performance_em
        self.histograms["hist_performance_lcw"] = hist_performance_lcw
        self.histograms["hist_performance_ml"] = hist_performance_ml


    def write_histogram(self):
        root_file = ROOT.TFile(self.output_path, "RECREATE")
        for key, value in self.histograms.items():
            value.Write()
        root_file.Close()

    def print_event_summary(self):
        print("\n" + "="*60)
        print("EVENT SELECTION SUMMARY")
        print("="*60)
        for cut, count in self.event_counts.items():
            if cut == "total":
                print(f"{cut:30s}: {count:10d}")
            else:
                prev_cut = list(self.event_counts.keys())[list(self.event_counts.keys()).index(cut)-1]
                prev_count = self.event_counts[prev_cut]
                eff = (count/prev_count*100) if prev_count > 0 else 0
                print(f"{cut:30s}: {count:10d} ({eff:5.1f}%)")
        print("="*60 + "\n")

# ============================================================================
# CLI AND MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze jet performance in Z+jets events"
    )
    parser.add_argument("--input",  action="append", help="Input ntuple files")
    parser.add_argument("--input-dir", help="Input directory containing ntuple files")
    parser.add_argument("--tree", required=True, help="Tree name in ntuple")
    parser.add_argument("--nEvents", type=int, default=-1, help="Max events to process")
    parser.add_argument("--output", default="jet_performance.root", help="Output ROOT file")
    return parser.parse_args()

def main():
    ROOT.gROOT.SetBatch(True)
    input_list = []

    args = parse_args()
    if args.input_dir:
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(".root"):
                    input_list.append(os.path.join(root, file))
    if args.input_dir:
        analysis = ZJetsAnalysis(input_list, args.tree, args.output, args.nEvents)
    else:
        analysis = ZJetsAnalysis(args.input, args.tree, args.output, args.nEvents)
    analysis.run()
    analysis.write_histogram()
    analysis.print_event_summary()

if __name__ == "__main__":
    main()
