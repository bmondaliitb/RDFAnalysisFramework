#!/usr/bin/env python3

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.core import NtupleProcessor, awkward_to_numpy
from rdf_analysis.stats import make_hist_from_bin_contents, make_numpy_hist


class Config:
    """All analysis configuration in one place."""

    DEBUG = False

    # Z boson
    Z_MASS = 91.1876
    Z_WINDOW = 20.0

    # Units
    GEV = 0.001

    # The stored recoil used to build the reference pTmiss.
    MET_INDEX = 8

    # Forward jet selection
    FORWARD_ETA_MIN = 2.5
    FORWARD_ETA_MAX = 4.9

    # Keep the same cluster units used in jet_performace_in_Zjets.py.
    CLUSTER_ENERGY_SCALES = (
        ("em", "EM", "jet_cluster_rawE", 1.0),
        ("lcw", "LCW", "jet_cluster_calE", 1.0),
        ("ml", "ML", "jet_cluster_MLE", 1.0),
        ("truth", "truth", "jet_cluster_truthE", GEV),
    )

    # Histogram binning
    PTZ_BIN_EDGES = (
        0, 5, 10, 15, 20, 25, 30, 40,
        50, 60, 70, 80, 90, 100, 150, 250,
    )
    PZ_BINS = 100
    PZ_MIN = -100.0
    PZ_MAX = 100.0
    MET_BINS = 100
    MET_MIN = 0.0
    MET_MAX = 250.0
    CLOSURE_BINS = 100
    CLOSURE_MIN = -1.0e-6
    CLOSURE_MAX = 1.0e-6
    PROGRESS_INTERVAL = 10000


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class TransverseVector:
    """Two-dimensional transverse vector."""

    x: float
    y: float

    @property
    def pt(self) -> float:
        return float(np.hypot(self.x, self.y))

    def dot(self, other: "TransverseVector") -> float:
        return float(self.x * other.x + self.y * other.y)

    def unit(self) -> "TransverseVector":
        if self.pt == 0.0:
            return TransverseVector(np.nan, np.nan)
        return TransverseVector(self.x / self.pt, self.y / self.pt)

    def __add__(self, other: "TransverseVector") -> "TransverseVector":
        return TransverseVector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "TransverseVector") -> "TransverseVector":
        return TransverseVector(self.x - other.x, self.y - other.y)

    def __neg__(self) -> "TransverseVector":
        return TransverseVector(-self.x, -self.y)

    @classmethod
    def from_pt_phi(cls, pt: float, phi: float) -> "TransverseVector":
        return cls(
            float(pt * np.cos(phi)),
            float(pt * np.sin(phi)),
        )


@dataclass
class Lepton:
    """Collection of leptons used to reconstruct the Z boson."""

    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray

    @property
    def count(self) -> int:
        return len(self.pt)

    @property
    def z_vector(self) -> TransverseVector:
        pt = np.asarray(self.pt[:2], dtype=np.float64)
        phi = np.asarray(self.phi[:2], dtype=np.float64)
        return TransverseVector(
            float(np.sum(pt * np.cos(phi))),
            float(np.sum(pt * np.sin(phi))),
        )

    @property
    def dilepton_mass(self) -> float:
        if self.count < 2:
            return np.nan

        deta = self.eta[0] - self.eta[1]
        dphi = ROOT.TVector2.Phi_mpi_pi(self.phi[0] - self.phi[1])
        mass_squared = 2.0 * self.pt[0] * self.pt[1] * (np.cosh(deta) - np.cos(dphi))
        return float(np.sqrt(max(mass_squared, 0.0)))


@dataclass
class Cluster:
    """Clusters belonging to one reconstructed jet."""

    eta: np.ndarray
    phi: np.ndarray
    energy: Dict[str, np.ndarray]

    def has_consistent_size(self) -> bool:
        expected_size = len(self.eta)
        return len(self.phi) == expected_size and all(
            len(values) == expected_size for values in self.energy.values()
        )

    def transverse_momentum(self, scale: str) -> TransverseVector:
        energy = self.energy[scale]
        valid = np.isfinite(energy) & np.isfinite(self.eta) & np.isfinite(self.phi)
        if not np.any(valid):
            return TransverseVector(0.0, 0.0)

        cluster_pt = energy[valid] / np.cosh(self.eta[valid])
        return TransverseVector(
            float(np.sum(cluster_pt * np.cos(self.phi[valid]))),
            float(np.sum(cluster_pt * np.sin(self.phi[valid]))),
        )


@dataclass
class Jet:
    """Collection of reconstructed jets and their constituent clusters."""

    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    cluster_eta: object
    cluster_phi: object
    cluster_energy: Dict[str, object]

    @property
    def count(self) -> int:
        return len(self.pt)

    def has_consistent_size(self) -> bool:
        return (
            len(self.eta) == self.count
            and len(self.phi) == self.count
            and len(self.cluster_eta) == self.count
            and len(self.cluster_phi) == self.count
            and all(len(values) == self.count for values in self.cluster_energy.values())
        )

    def forward_indices(self, eta_min: float, eta_max: float) -> np.ndarray:
        abs_eta = np.abs(self.eta)
        return np.flatnonzero(
            np.isfinite(abs_eta)
            & (abs_eta >= eta_min)
            & (abs_eta < eta_max)
        )

    def vector(self, index: int) -> TransverseVector:
        return TransverseVector.from_pt_phi(self.pt[index], self.phi[index])

    def clusters(self, index: int) -> Cluster:
        return Cluster(
            eta=awkward_to_numpy(self.cluster_eta[index]),
            phi=awkward_to_numpy(self.cluster_phi[index]),
            energy={
                scale: awkward_to_numpy(values[index])
                for scale, values in self.cluster_energy.items()
            },
        )


@dataclass(frozen=True)
class MissingTransverseMomentum:
    """Missing transverse momentum vector."""

    vector: TransverseVector

    @property
    def pt(self) -> float:
        return self.vector.pt

    def projection_on(self, vector: TransverseVector) -> float:
        unit_vector = vector.unit()
        if not np.isfinite(unit_vector.x):
            return np.nan
        return self.vector.dot(unit_vector)

    @classmethod
    def from_recoil(
        cls,
        recoil: TransverseVector,
        z_vector: TransverseVector,
    ) -> "MissingTransverseMomentum":
        return cls(-(recoil + z_vector))


@dataclass
class ObservableValues:
    """Event-level observables saved for one pTmiss definition."""

    ptz: List[float] = field(default_factory=list)
    pz: List[float] = field(default_factory=list)
    met: List[float] = field(default_factory=list)

    def append(self, z_vector: TransverseVector, met: MissingTransverseMomentum) -> None:
        self.ptz.append(z_vector.pt)
        self.pz.append(met.projection_on(z_vector))
        self.met.append(met.pt)

    def convert_to_numpy(self) -> None:
        self.ptz = np.asarray(self.ptz, dtype=np.float64)
        self.pz = np.asarray(self.pz, dtype=np.float64)
        self.met = np.asarray(self.met, dtype=np.float64)


@dataclass
class BinnedPerformance:
    """pTmiss scale, response, and resolution in pTZ bins."""

    entries: np.ndarray
    mean_ptz: np.ndarray
    mean_pz: np.ndarray
    response: np.ndarray
    raw_resolution: np.ndarray
    resolution: np.ndarray


# ============================================================================
# PHYSICS AND HISTOGRAM UTILITIES
# ============================================================================


def parse_bin_edges(text: str) -> np.ndarray:
    """Parse a comma-separated, strictly increasing list of bin edges."""
    edges = np.asarray([float(value) for value in text.split(",")], dtype=np.float64)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise argparse.ArgumentTypeError("bin edges must be a comma-separated increasing list")
    return edges


def calculate_binned_performance(
    ptz: np.ndarray,
    pz: np.ndarray,
    ptz_bin_edges: Iterable[float],
) -> BinnedPerformance:
    """
    Calculate the Z-system pTmiss performance.

    The definitions follow arXiv:2402.05858:
      C^Z = 1 + <P^Z>/<pTZ>
      resolution = RMS(P^Z)/C^Z
    """
    ptz = np.asarray(ptz, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64)
    edges = np.asarray(ptz_bin_edges, dtype=np.float64)

    entries = []
    mean_ptz = []
    mean_pz = []
    response = []
    raw_resolution = []
    resolution = []

    finite = np.isfinite(ptz) & np.isfinite(pz)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            in_bin = finite & (ptz >= low) & (ptz <= high)
        else:
            in_bin = finite & (ptz >= low) & (ptz < high)

        bin_entries = int(np.count_nonzero(in_bin))
        entries.append(bin_entries)
        if bin_entries == 0:
            mean_ptz.append(np.nan)
            mean_pz.append(np.nan)
            response.append(np.nan)
            raw_resolution.append(np.nan)
            resolution.append(np.nan)
            continue

        bin_mean_ptz = float(np.mean(ptz[in_bin]))
        bin_mean_pz = float(np.mean(pz[in_bin]))
        bin_response = (
            1.0 + bin_mean_pz / bin_mean_ptz
            if bin_mean_ptz != 0.0
            else np.nan
        )

        # np.std is the same width returned by ROOT's histogram RMS.
        bin_raw_resolution = float(np.std(pz[in_bin]))
        bin_resolution = (
            bin_raw_resolution / bin_response
            if np.isfinite(bin_response) and bin_response != 0.0
            else np.nan
        )

        mean_ptz.append(bin_mean_ptz)
        mean_pz.append(bin_mean_pz)
        response.append(bin_response)
        raw_resolution.append(bin_raw_resolution)
        resolution.append(bin_resolution)

    return BinnedPerformance(
        entries=np.asarray(entries, dtype=np.int64),
        mean_ptz=np.asarray(mean_ptz, dtype=np.float64),
        mean_pz=np.asarray(mean_pz, dtype=np.float64),
        response=np.asarray(response, dtype=np.float64),
        raw_resolution=np.asarray(raw_resolution, dtype=np.float64),
        resolution=np.asarray(resolution, dtype=np.float64),
    )


def make_histogram_2d(
    name: str,
    title: str,
    x_edges: Iterable[float],
    y_edges: Iterable[float],
    x_values: np.ndarray,
    y_values: np.ndarray,
):
    """Make a ROOT TH2D from two arrays."""
    x_edges = np.asarray(x_edges, dtype=np.float64)
    y_edges = np.asarray(y_edges, dtype=np.float64)
    histogram = ROOT.TH2D(
        name,
        title,
        len(x_edges) - 1,
        x_edges,
        len(y_edges) - 1,
        y_edges,
    )
    for x_value, y_value in zip(x_values, y_values):
        if np.isfinite(x_value) and np.isfinite(y_value):
            histogram.Fill(float(x_value), float(y_value))
    return histogram


# ============================================================================
# EVENT PROCESSOR
# ============================================================================


class METCalculationValidation:
    """Validate stored pTmiss and test forward-jet cluster calibrations."""

    COLUMNS = (
        "mu_pt",
        "mu_eta",
        "mu_phi",
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_cluster_eta",
        "jet_cluster_phi",
        "jet_cluster_rawE",
        "jet_cluster_calE",
        "jet_cluster_MLE",
        "jet_cluster_truthE",
        "u_pt.second",
        "u_phi.second",
    )

    VARIANT_LABELS = {
        "stored": "stored p_{T}^{miss}, all Z events",
        "stored_forward": "stored p_{T}^{miss}, forward-jet events",
        "forward_jets_removed": "stored p_{T}^{miss}, forward jets removed",
        "forward_clusters_em": "forward jets replaced by EM clusters",
        "forward_clusters_lcw": "forward jets replaced by LCW clusters",
        "forward_clusters_ml": "forward jets replaced by ML clusters",
        "forward_clusters_truth": "forward jets replaced by truth clusters",
    }

    def __init__(
        self,
        input_files: Union[str, List[str]],
        tree_name: str,
        output_path: str,
        max_events: int = -1,
        eta_min: float = Config.FORWARD_ETA_MIN,
        eta_max: float = Config.FORWARD_ETA_MAX,
        ptz_bin_edges: Iterable[float] = Config.PTZ_BIN_EDGES,
        apply_z_mass_window: bool = True,
    ):
        self.processor = NtupleProcessor(input_files, tree_name, max_events)
        self.output_path = output_path
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.ptz_bin_edges = np.asarray(ptz_bin_edges, dtype=np.float64)
        self.apply_z_mass_window = apply_z_mass_window

        self.values = {
            name: ObservableValues()
            for name in self.VARIANT_LABELS
        }
        self.projection_closure_ptz = []
        self.projection_closure = []
        self.performance: Dict[str, BinnedPerformance] = {}
        self.histograms: Dict[str, object] = {}
        self.bin_histograms: Dict[str, List[object]] = {}
        self.event_counts = {
            "total": 0,
            "two_muons": 0,
            "z_mass_window": 0,
            "valid_stored_recoil": 0,
            "valid_jet_branches": 0,
            "has_forward_jets": 0,
            "valid_forward_clusters": 0,
        }
        self.forward_jet_count = 0

    def run(self) -> None:
        """Execute the full analysis."""
        print("Starting pTmiss calculation validation...")
        for arrays in self.processor.iter_arrays(self.COLUMNS):
            self._process_chunk(arrays)

        self._convert_values_to_numpy()
        self._calculate_performance()
        self._build_histograms()

    def _process_chunk(self, arrays) -> None:
        for event in range(len(arrays["mu_pt"])):
            if self.event_counts["total"] % Config.PROGRESS_INTERVAL == 0:
                print("[Info]:: Processing event {}".format(self.event_counts["total"]))
            self.event_counts["total"] += 1

            leptons = Lepton(
                pt=awkward_to_numpy(arrays["mu_pt"][event]) * Config.GEV,
                eta=awkward_to_numpy(arrays["mu_eta"][event]),
                phi=awkward_to_numpy(arrays["mu_phi"][event]),
            )
            if leptons.count < 2:
                continue
            self.event_counts["two_muons"] += 1

            if self.apply_z_mass_window:
                mass = leptons.dilepton_mass
                if not (
                    (Config.Z_MASS - Config.Z_WINDOW)
                    < mass
                    < (Config.Z_MASS + Config.Z_WINDOW)
                ):
                    continue
            self.event_counts["z_mass_window"] += 1

            z_vector = leptons.z_vector
            if not np.isfinite(z_vector.pt) or z_vector.pt == 0.0:
                continue

            recoil = self._stored_recoil(arrays, event)
            if recoil is None:
                continue
            self.event_counts["valid_stored_recoil"] += 1

            stored_met = MissingTransverseMomentum.from_recoil(recoil, z_vector)
            self.values["stored"].append(z_vector, stored_met)
            self._append_projection_closure(z_vector, recoil, stored_met)

            jets = self._build_jets(arrays, event)
            if not jets.has_consistent_size():
                if Config.DEBUG:
                    print(
                        "[Warning]:: Jet/cluster outer-size mismatch in event {}".format(
                            self.event_counts["total"] - 1
                        )
                    )
                continue
            self.event_counts["valid_jet_branches"] += 1

            forward_indices = jets.forward_indices(self.eta_min, self.eta_max)
            if len(forward_indices) == 0:
                continue
            self.event_counts["has_forward_jets"] += 1
            self.forward_jet_count += len(forward_indices)

            cluster_vectors = self._forward_cluster_vectors(jets, forward_indices)
            if cluster_vectors is None:
                continue
            self.event_counts["valid_forward_clusters"] += 1

            self.values["stored_forward"].append(z_vector, stored_met)

            # MET = -sum(visible). Removing a jet adds its momentum to MET.
            forward_jet_vector = TransverseVector(0.0, 0.0)
            for jet_index in forward_indices:
                forward_jet_vector = forward_jet_vector + jets.vector(jet_index)

            jets_removed_vector = stored_met.vector + forward_jet_vector
            jets_removed_met = MissingTransverseMomentum(jets_removed_vector)
            self.values["forward_jets_removed"].append(z_vector, jets_removed_met)

            # Adding clusters to the visible sum subtracts their momentum from MET.
            for scale, _, _, _ in Config.CLUSTER_ENERGY_SCALES:
                cluster_met = MissingTransverseMomentum(
                    jets_removed_vector - cluster_vectors[scale]
                )
                self.values[f"forward_clusters_{scale}"].append(z_vector, cluster_met)

    def _stored_recoil(self, arrays, event: int) -> Optional[TransverseVector]:
        recoil_pt = awkward_to_numpy(arrays["u_pt.second"][event])
        recoil_phi = awkward_to_numpy(arrays["u_phi.second"][event])
        if len(recoil_pt) <= Config.MET_INDEX or len(recoil_phi) <= Config.MET_INDEX:
            return None

        pt = float(recoil_pt[Config.MET_INDEX]) * Config.GEV
        phi = float(recoil_phi[Config.MET_INDEX])
        if not np.isfinite(pt) or not np.isfinite(phi):
            return None
        return TransverseVector.from_pt_phi(pt, phi)

    def _build_jets(self, arrays, event: int) -> Jet:
        energy = {}
        for scale, _, branch, unit in Config.CLUSTER_ENERGY_SCALES:
            energy[scale] = arrays[branch][event] * unit

        return Jet(
            pt=awkward_to_numpy(arrays["jet_pt"][event]) * Config.GEV,
            eta=awkward_to_numpy(arrays["jet_eta"][event]),
            phi=awkward_to_numpy(arrays["jet_phi"][event]),
            cluster_eta=arrays["jet_cluster_eta"][event],
            cluster_phi=arrays["jet_cluster_phi"][event],
            cluster_energy=energy,
        )

    def _forward_cluster_vectors(
        self,
        jets: Jet,
        forward_indices: np.ndarray,
    ) -> Optional[Dict[str, TransverseVector]]:
        vectors = {
            scale: TransverseVector(0.0, 0.0)
            for scale, _, _, _ in Config.CLUSTER_ENERGY_SCALES
        }

        for jet_index in forward_indices:
            clusters = jets.clusters(jet_index)
            if not clusters.has_consistent_size():
                if Config.DEBUG:
                    print(
                        "[Warning]:: Cluster-size mismatch for forward jet {}".format(
                            jet_index
                        )
                    )
                return None

            for scale in vectors:
                vectors[scale] = vectors[scale] + clusters.transverse_momentum(scale)

        return vectors

    def _append_projection_closure(
        self,
        z_vector: TransverseVector,
        recoil: TransverseVector,
        met: MissingTransverseMomentum,
    ) -> None:
        """
        Validate PZ calculated in two independent ways.

        Since pTmiss = -(uT + pTZ),
          PZ = pTmiss . zhat = -(u_parallel + pTZ).
        """
        pz_from_met = met.projection_on(z_vector)
        pz_from_recoil = -(recoil.dot(z_vector.unit()) + z_vector.pt)
        self.projection_closure_ptz.append(z_vector.pt)
        self.projection_closure.append(pz_from_met - pz_from_recoil)

    def _convert_values_to_numpy(self) -> None:
        for values in self.values.values():
            values.convert_to_numpy()
        self.projection_closure_ptz = np.asarray(
            self.projection_closure_ptz,
            dtype=np.float64,
        )
        self.projection_closure = np.asarray(
            self.projection_closure,
            dtype=np.float64,
        )

    def _calculate_performance(self) -> None:
        for name, values in self.values.items():
            self.performance[name] = calculate_binned_performance(
                values.ptz,
                values.pz,
                self.ptz_bin_edges,
            )

    def _build_histograms(self) -> None:
        for name, values in self.values.items():
            self._build_variant_histograms(name, values)

        self._build_projection_closure_histograms()
        self._build_improvement_histograms()

    def _build_variant_histograms(
        self,
        name: str,
        values: ObservableValues,
    ) -> None:
        label = self.VARIANT_LABELS[name]
        finite = (
            np.isfinite(values.ptz)
            & np.isfinite(values.pz)
            & np.isfinite(values.met)
        )

        self.histograms[f"h_pTZ_{name}"] = make_numpy_hist(
            f"h_pTZ_{name}",
            f"{label};p_{{T}}^{{Z}} [GeV];Events",
            self.ptz_bin_edges,
            values.ptz[finite],
        )
        self.histograms[f"h_pz_{name}"] = make_numpy_hist(
            f"h_pz_{name}",
            f"{label};P^{{Z}} [GeV];Events",
            Config.PZ_BINS,
            Config.PZ_MIN,
            Config.PZ_MAX,
            values.pz[finite],
        )
        self.histograms[f"h_met_{name}"] = make_numpy_hist(
            f"h_met_{name}",
            f"{label};p_{{T}}^{{miss}} [GeV];Events",
            Config.MET_BINS,
            Config.MET_MIN,
            Config.MET_MAX,
            values.met[finite],
        )
        self.histograms[f"h2_pz_vs_pTZ_{name}"] = make_histogram_2d(
            f"h2_pz_vs_pTZ_{name}",
            f"{label};p_{{T}}^{{Z}} [GeV];P^{{Z}} [GeV]",
            self.ptz_bin_edges,
            np.linspace(Config.PZ_MIN, Config.PZ_MAX, Config.PZ_BINS + 1),
            values.ptz[finite],
            values.pz[finite],
        )

        performance = self.performance[name]
        self.histograms[f"h_entries_vs_pTZ_{name}"] = make_hist_from_bin_contents(
            f"h_entries_vs_pTZ_{name}",
            f"{label};p_{{T}}^{{Z}} [GeV];Events",
            self.ptz_bin_edges,
            performance.entries,
        )
        self.histograms[f"h_mean_pz_vs_pTZ_{name}"] = make_hist_from_bin_contents(
            f"h_mean_pz_vs_pTZ_{name}",
            f"{label};p_{{T}}^{{Z}} [GeV];<P^{{Z}}> [GeV]",
            self.ptz_bin_edges,
            performance.mean_pz,
        )
        self.histograms[f"h_response_vs_pTZ_{name}"] = make_hist_from_bin_contents(
            f"h_response_vs_pTZ_{name}",
            f"{label};p_{{T}}^{{Z}} [GeV];C^{{Z}}",
            self.ptz_bin_edges,
            performance.response,
        )
        self.histograms[f"h_raw_resolution_vs_pTZ_{name}"] = make_hist_from_bin_contents(
            f"h_raw_resolution_vs_pTZ_{name}",
            f"{label};p_{{T}}^{{Z}} [GeV];RMS(P^{{Z}}) [GeV]",
            self.ptz_bin_edges,
            performance.raw_resolution,
        )
        self.histograms[f"h_resolution_vs_pTZ_{name}"] = make_hist_from_bin_contents(
            f"h_resolution_vs_pTZ_{name}",
            f"{label};p_{{T}}^{{Z}} [GeV];RMS(P^{{Z}})/C^{{Z}} [GeV]",
            self.ptz_bin_edges,
            performance.resolution,
        )

        self.bin_histograms[name] = []
        for index, (low, high) in enumerate(
            zip(self.ptz_bin_edges[:-1], self.ptz_bin_edges[1:])
        ):
            if index == len(self.ptz_bin_edges) - 2:
                in_bin = finite & (values.ptz >= low) & (values.ptz <= high)
            else:
                in_bin = finite & (values.ptz >= low) & (values.ptz < high)
            suffix = f"ptz_{low:g}_{high:g}".replace(".", "p")
            self.bin_histograms[name].append(
                make_numpy_hist(
                    f"h_pz_{name}_{suffix}",
                    (
                        f"{label}, {low:g} <= p_{{T}}^{{Z}} < {high:g} GeV;"
                        "P^{Z} [GeV];Events"
                    ),
                    Config.PZ_BINS,
                    Config.PZ_MIN,
                    Config.PZ_MAX,
                    values.pz[in_bin],
                )
            )

    def _build_projection_closure_histograms(self) -> None:
        finite = (
            np.isfinite(self.projection_closure_ptz)
            & np.isfinite(self.projection_closure)
        )
        self.histograms["h_projection_closure"] = make_numpy_hist(
            "h_projection_closure",
            (
                "P^{Z} validation;"
                "P^{Z}_{from pTmiss} - P^{Z}_{from recoil} [GeV];Events"
            ),
            Config.CLOSURE_BINS,
            Config.CLOSURE_MIN,
            Config.CLOSURE_MAX,
            self.projection_closure[finite],
        )

        closure_performance = calculate_binned_performance(
            self.projection_closure_ptz[finite],
            self.projection_closure[finite],
            self.ptz_bin_edges,
        )
        self.histograms["h_mean_projection_closure_vs_pTZ"] = (
            make_hist_from_bin_contents(
                "h_mean_projection_closure_vs_pTZ",
                (
                    "P^{Z} validation;p_{T}^{Z} [GeV];"
                    "<P^{Z}_{from pTmiss} - P^{Z}_{from recoil}> [GeV]"
                ),
                self.ptz_bin_edges,
                closure_performance.mean_pz,
            )
        )

    def _build_improvement_histograms(self) -> None:
        ml = self.performance["forward_clusters_ml"]
        for reference_name in ("em", "lcw"):
            reference = self.performance[f"forward_clusters_{reference_name}"]
            response_bias_improvement = (
                np.abs(reference.response - 1.0)
                - np.abs(ml.response - 1.0)
            )
            resolution_improvement = reference.resolution - ml.resolution

            self.histograms[
                f"h_ml_response_bias_improvement_vs_{reference_name}_vs_pTZ"
            ] = make_hist_from_bin_contents(
                f"h_ml_response_bias_improvement_vs_{reference_name}_vs_pTZ",
                (
                    f"ML improvement over {reference_name.upper()};"
                    "p_{T}^{Z} [GeV];"
                    "|C^{Z}_{reference}-1|-|C^{Z}_{ML}-1|"
                ),
                self.ptz_bin_edges,
                response_bias_improvement,
            )
            self.histograms[
                f"h_ml_resolution_improvement_vs_{reference_name}_vs_pTZ"
            ] = make_hist_from_bin_contents(
                f"h_ml_resolution_improvement_vs_{reference_name}_vs_pTZ",
                (
                    f"ML improvement over {reference_name.upper()};"
                    "p_{T}^{Z} [GeV];"
                    "resolution_{reference}-resolution_{ML} [GeV]"
                ),
                self.ptz_bin_edges,
                resolution_improvement,
            )

    def write_histograms(self) -> None:
        """Write all histograms to the configured ROOT file."""
        output_file = ROOT.TFile.Open(self.output_path, "RECREATE")
        if not output_file or output_file.IsZombie():
            raise OSError("Unable to open output ROOT file: {}".format(self.output_path))

        for histogram in self.histograms.values():
            histogram.Write()

        bin_directory = output_file.mkdir("pz_in_ptz_bins")
        for name, histograms in self.bin_histograms.items():
            directory = bin_directory.mkdir(name)
            directory.cd()
            for histogram in histograms:
                histogram.Write()
            output_file.cd()

        output_file.Close()
        print("Histograms saved to {}".format(self.output_path))

    def print_event_summary(self) -> None:
        """Print event selection and the ML comparison in each pTZ bin."""
        print("\n" + "=" * 72)
        print("PTMISS CALCULATION VALIDATION SUMMARY")
        print("=" * 72)
        for cut, count in self.event_counts.items():
            print(f"{cut:32s}: {count:10d}")
        print(f"{'forward jets replaced':32s}: {self.forward_jet_count:10d}")
        print(
            "Forward jet range: {:.2f} <= |eta| < {:.2f}".format(
                self.eta_min,
                self.eta_max,
            )
        )

        if len(self.projection_closure) > 0:
            print(
                "Max |PZ projection closure|: {:.6e} GeV".format(
                    float(np.nanmax(np.abs(self.projection_closure)))
                )
            )

        print("\nML comparison in forward-jet events")
        print(
            "{:>13s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
                "pTZ [GeV]",
                "C_EM",
                "C_LCW",
                "C_ML",
                "Res_LCW",
                "Res_ML",
            )
        )
        em = self.performance["forward_clusters_em"]
        lcw = self.performance["forward_clusters_lcw"]
        ml = self.performance["forward_clusters_ml"]
        for index, (low, high) in enumerate(
            zip(self.ptz_bin_edges[:-1], self.ptz_bin_edges[1:])
        ):
            if ml.entries[index] == 0:
                continue
            print(
                "{:5g}-{:5g} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(
                    low,
                    high,
                    em.response[index],
                    lcw.response[index],
                    ml.response[index],
                    lcw.resolution[index],
                    ml.resolution[index],
                )
            )
        print("=" * 72 + "\n")


# ============================================================================
# CLI AND MAIN
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate stored pTmiss in pTZ bins and replace forward jets "
            "with EM, LCW, ML, or truth clusters."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        help="Input ntuple file; repeat for multiple files.",
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing ntuple files.",
    )
    parser.add_argument(
        "--tree",
        required=True,
        help="Tree name in the ntuple.",
    )
    parser.add_argument(
        "--nEvents",
        type=int,
        default=-1,
        help="Maximum number of events to process.",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=Config.FORWARD_ETA_MIN,
        help="Minimum forward-jet absolute eta.",
    )
    parser.add_argument(
        "--eta-max",
        type=float,
        default=Config.FORWARD_ETA_MAX,
        help="Maximum forward-jet absolute eta.",
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
        help="Do not apply the nominal Z-mass window.",
    )
    parser.add_argument(
        "--output",
        default="validate_met_calculation.root",
        help="Output ROOT file.",
    )
    return parser.parse_args()


def input_files_from_args(args) -> List[str]:
    input_files = []
    if args.input:
        input_files.extend(args.input)
    if args.input_dir:
        for root, _, files in os.walk(args.input_dir):
            for file_name in files:
                if file_name.endswith(".root"):
                    input_files.append(os.path.join(root, file_name))

    input_files = sorted(set(input_files))
    if not input_files:
        raise ValueError("Provide at least one --input or --input-dir.")
    return input_files


def main() -> None:
    ROOT.gROOT.SetBatch(True)
    args = parse_args()

    if args.eta_min < 0.0 or args.eta_min >= args.eta_max:
        raise ValueError("Require 0 <= eta-min < eta-max.")

    analysis = METCalculationValidation(
        input_files=input_files_from_args(args),
        tree_name=args.tree,
        output_path=args.output,
        max_events=args.nEvents,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        ptz_bin_edges=args.ptz_bins,
        apply_z_mass_window=not args.no_z_window,
    )
    analysis.run()
    analysis.write_histograms()
    analysis.print_event_summary()


if __name__ == "__main__":
    main()
