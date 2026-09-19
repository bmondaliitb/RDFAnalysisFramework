"""Shared object-oriented implementation for W/Z MET resolution studies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
import ROOT

from rdf_analysis.analyses.met_inputs_common import (BinnedPerformance, METConfig, ObservableValues, TransverseVector, build_clusters, build_met_inputs, build_truth_met, calculate_binned_performance,)
from rdf_analysis.core import NtupleProcessor
from rdf_analysis.stats.fits import calculate_sigma_iqr_in_x_bins, calculate_binned_mean
from rdf_analysis.stats.histograms import (make_hist_from_bin_contents, make_numpy_hist, make_numpy_hists_2d,)


class METResolutionConfig:
    """Shared selection and histogram settings."""

    # Forward objects use the half-open interval η_min ≤ |η| < η_max.
    FORWARD_ETA_MIN = 2.5
    FORWARD_ETA_MAX = 4.9
    CLUSTER_RADIUS = 0.4
    APPLY_Z_MASS_WINDOW = False
    PROGRESS_INTERVAL = 10000

    # Strict pT threshold in GeV for forward-jet removal and cluster matching.
    FORWARD_JET_PT_MIN = 30.0
    FORWARD_JET_PT_BINS = 150
    FORWARD_JET_PT_RANGE = (-50.0, 700.0)
    FORWARD_CLUSTER_ENERGY_BINS = 250
    FORWARD_CLUSTER_ENERGY_RANGE = (-10.0, 240.0)

    PTZ_BIN_EDGES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 160, 200, 260,)
    AVGMU_BIN_EDGES = tuple(range(0, 85, 5))
    NPRIMVTX_BIN_EDGES = tuple(range(0, 51, 2))

    MET_BINS = 120
    MET_MIN = 0.0
    MET_MAX = 300.0
    PZ_BINS = 120
    PZ_MIN = -120.0
    PZ_MAX = 120.0
    BOSON_ABS_ETA_BINS = 100
    BOSON_ABS_ETA_MIN = 0.0
    BOSON_ABS_ETA_MAX = 4.9
    PHI_BINS = 64
    PHI_MIN = -np.pi
    PHI_MAX = np.pi

    CLUSTER_ENERGY_BINS = 250
    CLUSTER_ENERGY_MIN = -10.0
    CLUSTER_ENERGY_MAX = 240.0


@dataclass(frozen=True)
class BosonCandidate:
    """Selected boson direction and pseudorapidity."""
    vector: TransverseVector
    eta: float

class METResolutionStudyBase(ABC):
    """Shared event loop, MET scenarios, metrics, and ROOT output."""

    CONFIG = METResolutionConfig
    BOSON_SYMBOL = "V"

    BASE_VARIANT_LABELS = {
        "current": "current MET",
        "truth_met": "truth MET (Int + IntMuons)",
    }

    def __init__(self, input_files: Union[str, List[str]], tree_name: str, output_path: str, max_events: int = -1,) -> None:
        config = self.CONFIG
        self.processor = NtupleProcessor(input_files, tree_name, max_events)
        self.output_path = output_path

        self.eta_min = config.FORWARD_ETA_MIN
        self.eta_max = config.FORWARD_ETA_MAX
        self.cluster_radius = config.CLUSTER_RADIUS
        self.ptz_bin_edges = np.asarray(config.PTZ_BIN_EDGES, dtype=np.float64)
        self.apply_z_mass_window = config.APPLY_Z_MASS_WINDOW

        self.boson_abs_eta_min = config.BOSON_ABS_ETA_MIN
        self.boson_abs_eta_max = config.BOSON_ABS_ETA_MAX
        if (self.boson_abs_eta_min < 0.0 or self.boson_abs_eta_min >= self.boson_abs_eta_max):
            raise ValueError("require 0 <= BOSON_ABS_ETA_MIN < BOSON_ABS_ETA_MAX")

        self.truth_pt_branch = METConfig.TRUTH_MET_PT_BRANCH
        self.truth_phi_branch = METConfig.TRUTH_MET_PHI_BRANCH

        self.variant_labels = self._build_variant_labels()
        self.jet_inclusive_values = ObservableValues()
        self.boson_eta_values = []

        self.values = {name: ObservableValues() for name in self.variant_labels}
        self.met_phi_values = {name: [] for name in self.variant_labels}

        self.pileup_values = {"avgMu": [], "nPrimVtx": []}
        self.met_residuals = {name: {"x": [], "y": []} for name in self.variant_labels}

        self.performance: Dict[str, BinnedPerformance] = {}
        self.histograms = {}
        self.event_counts = {
            "total": 0,
            "valid_met_inputs": 0,
            **self.selection_event_counts(),
            "boson_abs_eta_range": 0,
            "valid_truth_met": 0,
            "valid_truth_direction": 0,
            "has_forward_met_input_jets": 0,
            "valid_global_clusters": 0,
        }

    def _build_variant_labels(self) -> Dict[str, str]:
        labels = dict(self.BASE_VARIANT_LABELS)
        for scale, (scale_label, _) in METConfig.CLUSTER_SCALES.items():
            labels[f"keep_jets_forward_clusters_{scale}"] = ("forward jets kept + forward clusters outside jets with " f"pT > {self.CONFIG.FORWARD_JET_PT_MIN:g} GeV, {scale_label} scale")
            labels[f"replace_forward_jets_with_all_forward_clusters_{scale}"] = (f"forward jets with pT > {self.CONFIG.FORWARD_JET_PT_MIN:g} GeV " "removed + all forward clusters, " f"{scale_label} scale")
        return labels

    @property
    def columns(self) -> List[str]:
        branches = (
            list(METConfig.MET_INPUT_BRANCHES.values())
            + [self.truth_pt_branch, self.truth_phi_branch]
            + self.boson_columns()
            + list(self.pileup_values)
            + list(METConfig.LEPTON_BRANCHES.values())
            + list(METConfig.CLUSTER_DIRECTION_BRANCHES.values())
            + [branch for _, branch in METConfig.CLUSTER_SCALES.values()]
        )
        return sorted(set(branches))

    @abstractmethod
    def boson_columns(self) -> List[str]:
        """Return process-specific boson branches."""

    @abstractmethod
    def selection_event_counts(self) -> Dict[str, int]:
        """Return process-specific cut-flow counters."""

    @abstractmethod
    def select_boson(self, arrays: Mapping[str, object], event: int,) -> Optional[BosonCandidate]:
        """Select and build the process-specific boson candidate."""

    def accepts_boson_eta(self, eta: float) -> bool:
        """Return whether eta passes the configured absolute acceptance."""
        # This is an acceptance cut on the selected boson, not on individual
        # leptons or jets; the W/Z subclass supplies the boson η.
        abs_eta = abs(eta)
        return bool(np.isfinite(abs_eta) and self.boson_abs_eta_min <= abs_eta < self.boson_abs_eta_max)

    def run(self) -> None:
        """Process input events and retain observables."""
        print(f"Starting forward-jet {self.BOSON_SYMBOL} MET resolution study...")
        for index, input_file in enumerate(self.processor.input_file, start=1):
            remaining_events = (self.processor.n_events_to_process - self.event_counts["total"])
            if remaining_events <= 0:
                break
            print("[Info] Processing input file {}/{}: {}".format(index, len(self.processor.input_file), input_file,), flush=True,)
            file_processor = NtupleProcessor(input_file, self.processor.tree_name, remaining_events, step_size=self.processor.step_size,)
            for arrays in file_processor.iter_arrays(self.columns):
                self._process_chunk(arrays)
        self._convert_to_numpy()

    def _convert_to_numpy(self) -> None:
        for values in self.values.values():
            values.convert_to_numpy()
        self.jet_inclusive_values.convert_to_numpy()
        self.boson_eta_values = np.asarray(self.boson_eta_values, dtype=np.float64)
        self.met_phi_values = {name: np.asarray(values, dtype=np.float64) for name, values in self.met_phi_values.items()}
        self.pileup_values = {name: np.asarray(values, dtype=np.float64) for name, values in self.pileup_values.items()}
        for residuals in self.met_residuals.values():
            for component in residuals:
                residuals[component] = np.asarray(residuals[component], dtype=np.float64,)

    def _process_chunk(self, arrays: Mapping[str, object]) -> None:
        first_branch = METConfig.LEPTON_BRANCHES["pt"] # just to get the length of the chunk
        for event in range(len(arrays[first_branch])):
            if self.event_counts["total"] % self.CONFIG.PROGRESS_INTERVAL == 0:
                print(f"[Info] Processing event {self.event_counts['total']}")
            self.event_counts["total"] += 1

            # Apply cuts in this order: input integrity → channel/boson
            # selection → boson acceptance → truth-MET validity → clusters.
            met_inputs = build_met_inputs(arrays, event)
            if not met_inputs.has_consistent_size():
                continue
            self.event_counts["valid_met_inputs"] += 1

            boson = self.select_boson(arrays, event)
            if boson is None:
                continue
            boson_abs_eta = abs(boson.eta)
            if not self.accepts_boson_eta(boson.eta):
                continue
            self.event_counts["boson_abs_eta_range"] += 1
            # current_met is reconstructed from the stored weighted terms;
            # it is the baseline against which all cluster variants are tested.
            current_met = met_inputs.calculate_met()

            truth_met = build_truth_met(arrays, event, pt_branch=self.truth_pt_branch, phi_branch=self.truth_phi_branch,)
            if truth_met is None:
                continue
            self.event_counts["valid_truth_met"] += 1
            truth_met_vector = truth_met.vector
            if truth_met_vector is None:
                continue
            self.event_counts["valid_truth_direction"] += 1
            self.jet_inclusive_values.append(boson.vector, current_met, current_met - truth_met_vector,)

            # Forward jets are diagnostic here: their presence is counted, but
            # no forward-jet requirement is imposed on event selection.
            forward_indices = met_inputs.forward_jet_indices(self.eta_min, self.eta_max,)
            if len(forward_indices) > 0:
                self.event_counts["has_forward_met_input_jets"] += 1

            clusters = build_clusters(arrays, event)
            if not clusters.has_consistent_size():
                continue
            self.event_counts["valid_global_clusters"] += 1
            self.boson_eta_values.append(boson_abs_eta)
            self.pileup_values["avgMu"].append(float(arrays["avgMu"][event]))
            self.pileup_values["nPrimVtx"].append(float(arrays["nPrimVtx"][event]))

            self._append_variant("current", boson.vector, current_met, truth_met_vector,)
            self._append_variant("truth_met", boson.vector, truth_met_vector, truth_met_vector,)

            forward_jet_pt = met_inputs.jets.pt[forward_indices]
            selected_forward_indices = forward_indices[np.isfinite(forward_jet_pt) & (forward_jet_pt > self.CONFIG.FORWARD_JET_PT_MIN)]
            # Variant A keeps jets and adds forward clusters outside the
            # matching radius of forward jets above the configured pT cut.
            away_mask = clusters.away_from_jets(met_inputs.jets.eta[selected_forward_indices], met_inputs.jets.phi[selected_forward_indices], self.cluster_radius,)
            forward_cluster_mask = clusters.in_abs_eta_range(self.eta_min, self.eta_max,)
            forward_nonoverlap_vectors = clusters.visible_vectors(forward_cluster_mask & away_mask)
            all_forward_cluster_vectors = clusters.visible_vectors(forward_cluster_mask)
            # Variant B removes forward jets above the configured pT cut, then
            # adds all forward clusters: MET' = current MET + jets − clusters.
            forward_jet_vector = met_inputs.jets.visible_vector(selected_forward_indices)

            for scale in METConfig.CLUSTER_SCALES:
                self._append_variant(f"keep_jets_forward_clusters_{scale}", boson.vector, current_met - forward_nonoverlap_vectors[scale], truth_met_vector,)
                self._append_variant(f"replace_forward_jets_with_all_forward_clusters_{scale}", boson.vector, (current_met + forward_jet_vector - all_forward_cluster_vectors[scale]), truth_met_vector,)

    def _append_variant(self, name: str, boson: TransverseVector, met: TransverseVector, truth_met: TransverseVector,) -> None:
        residual = met - truth_met
        self.values[name].append(boson, met, residual)
        self.met_phi_values[name].append(met.phi)
        self.met_residuals[name]["x"].append(residual.x)
        self.met_residuals[name]["y"].append(residual.y)

    def _calculate_performance(self) -> None:
        """Measure truth-subtracted response and robust resolution."""
        self.performance = {}
        for name, values in self.values.items():
            moments = calculate_binned_performance(values.ptz, values.pz, self.ptz_bin_edges,)
            median_pz, iqr_sigma_pz = calculate_sigma_iqr_in_x_bins(values.ptz, values.pz, self.ptz_bin_edges,)
            iqr_sigma_pz = iqr_sigma_pz/2.0
            mean_ptz = calculate_binned_mean(values.ptz, values.ptz, self.ptz_bin_edges,)
            response = np.full_like(median_pz, np.nan)
            valid = (np.isfinite(median_pz) & np.isfinite(mean_ptz) & (mean_ptz != 0.0))
            response[valid] = 1.0 + median_pz[valid] / mean_ptz[valid]
            self.performance[name] = BinnedPerformance(entries=moments.entries, mean_pz=median_pz, response=response, raw_resolution=iqr_sigma_pz, resolution=iqr_sigma_pz.copy(),)

    def _build_histograms(self) -> None:
        config = self.CONFIG
        symbol = self.BOSON_SYMBOL
        inclusive = calculate_binned_performance(self.jet_inclusive_values.ptz, self.jet_inclusive_values.pz, self.ptz_bin_edges,)
        name = "h_mean_pz_vs_pTZ_jet_inclusive"
        self.histograms[name] = make_hist_from_bin_contents(name, (f"jet-inclusive current MET;p_{{T}}^{{{symbol}}} [GeV];" "<P_{||}^{reco-truth}> [GeV]"), self.ptz_bin_edges, inclusive.mean_pz,)
        finite_eta = np.isfinite(self.boson_eta_values)
        self.histograms["h_boson_eta"] = make_numpy_hist("h_boson_eta", f"Selected events;|#eta_{{{symbol}}}|;Events", config.BOSON_ABS_ETA_BINS, config.BOSON_ABS_ETA_MIN, config.BOSON_ABS_ETA_MAX, self.boson_eta_values[finite_eta],)
        for variant, values in self.values.items():
            self._build_variant_histograms(variant, values)
            self._build_pileup_resolution_histograms(variant)

    def _build_pileup_resolution_histograms(self, name: str) -> None:
        binnings = {
            "avgMu": (self.CONFIG.AVGMU_BIN_EDGES, "<#mu>"),
            "nPrimVtx": (self.CONFIG.NPRIMVTX_BIN_EDGES, "N_{PV}"),
        }
        for variable, (edges, axis_title) in binnings.items():
            for component in ("x", "y"):
                # Keep the legacy key so existing plotting jobs still find it.
                histogram_name = (f"h_rms_met_p{component}_residual_vs_{variable}_{name}")
                _, iqr_sigma = calculate_sigma_iqr_in_x_bins(self.pileup_values[variable], self.met_residuals[name][component], edges,)
                iqr_sigma = iqr_sigma/2.0
                self.histograms[histogram_name] = make_hist_from_bin_contents(histogram_name, (f"{self.variant_labels[name]};{axis_title};#sigma_{{IQR}}(p_{{{component}}}^{{miss}}-p_{{{component}}}^{{miss,true}}) [GeV]"), edges, iqr_sigma,)
                if variable == "nPrimVtx" and component == "x":
                    values = self.values[name]
                    median_pz, _ = calculate_sigma_iqr_in_x_bins(self.pileup_values[variable], values.pz, edges,)
                    mean_ptz = calculate_binned_mean(self.pileup_values[variable], values.ptz, edges,)
                    # Calculate C^Z in the same vertex bins as the width.
                    response = np.full_like(median_pz, np.nan)
                    valid = (np.isfinite(median_pz) & np.isfinite(mean_ptz) & (mean_ptz != 0.0))
                    response[valid] = 1.0 + median_pz[valid] / mean_ptz[valid]
                    histogram_name = f"h_response_vs_nPrimVtx_{name}"
                    self.histograms[histogram_name] = make_hist_from_bin_contents(histogram_name, f"{self.variant_labels[name]};{axis_title};C^{{Z}}", edges, response,)
                    corrected_resolution = np.full_like(iqr_sigma, np.nan)
                    np.divide(iqr_sigma, response, out=corrected_resolution, where=(np.isfinite(iqr_sigma) & np.isfinite(response) & (response != 0.0)),)
                    histogram_name = (f"h_iqr_met_px_residual_over_cz_vs_nPrimVtx_{name}")
                    self.histograms[histogram_name] = make_hist_from_bin_contents(histogram_name, (f"{self.variant_labels[name]};{axis_title};" "#sigma_{IQR}(p_{x}^{miss}-p_{x}^{miss,true})/C^{Z} [GeV]"), edges, corrected_resolution,)

    def _build_variant_histograms(self, name: str, values: ObservableValues,) -> None:
        config = self.CONFIG
        symbol = self.BOSON_SYMBOL
        label = self.variant_labels[name]
        finite = (np.isfinite(values.ptz) & np.isfinite(values.pz) & np.isfinite(values.met))
        self.histograms[f"h_met_{name}"] = make_numpy_hist(f"h_met_{name}", f"{label};p_{{T}}^{{miss}} [GeV];Events", config.MET_BINS, config.MET_MIN, config.MET_MAX, values.met[finite],)
        finite_phi = np.isfinite(self.met_phi_values[name])
        self.histograms[f"h_met_phi_{name}"] = make_numpy_hist(f"h_met_phi_{name}", f"{label};#phi(E_{{T}}^{{miss}});Events", config.PHI_BINS, config.PHI_MIN, config.PHI_MAX, self.met_phi_values[name][finite_phi],)
        self.histograms[f"h_pz_{name}"] = make_numpy_hist(f"h_pz_{name}", f"{label};P_{{||}}^{{reco-truth}} [GeV];Events", config.PZ_BINS, config.PZ_MIN, config.PZ_MAX, values.pz[finite],)
        self.histograms[f"h2_pz_vs_pTZ_{name}"] = make_numpy_hists_2d(f"h2_pz_vs_pTZ_{name}", (f"{label};p_{{T}}^{{{symbol}}} [GeV];" "P_{||}^{reco-truth} [GeV]"), x_bin_edges=self.ptz_bin_edges, y_bin_edges=np.linspace(config.PZ_MIN, config.PZ_MAX, config.PZ_BINS + 1,), x_values=values.ptz[finite], y_values=values.pz[finite],)

        performance = self.performance[name]
        # C^Z is the existing per-bin response; sigma_IQR = (Q84 - Q16) / 2.
        _, iqr_sigma_px = calculate_sigma_iqr_in_x_bins(values.ptz, self.met_residuals[name]["x"], self.ptz_bin_edges,)
        iqr_sigma_px = iqr_sigma_px/2.0
        corrected_px_resolution = np.full_like(iqr_sigma_px, np.nan)
        valid_response = (np.isfinite(iqr_sigma_px) & np.isfinite(performance.response) & (performance.response != 0.0))
        np.divide(iqr_sigma_px, performance.response, out=corrected_px_resolution, where=valid_response,)
        quantities = {
            "iqr_met_px_residual_over_cz": ("#sigma_{IQR}(p_{x}^{miss}-p_{x}^{miss,true})/C^{Z} [GeV]", corrected_px_resolution,),
            "entries": ("Events", performance.entries),
            "mean_pz": ("median(P_{||}^{reco-truth}) [GeV]", performance.mean_pz,),
            "response": ("R_{MET}^{reco-truth}", performance.response),
            "raw_resolution": ("#sigma_{IQR}(P_{||}^{reco-truth}) [GeV]", performance.raw_resolution,),
            "resolution": ("#sigma_{IQR}(P_{||}^{reco-truth}) [GeV]", performance.resolution,),
        }
        for quantity, (axis_title, contents) in quantities.items():
            histogram_name = f"h_{quantity}_vs_pTZ_{name}"
            self.histograms[histogram_name] = make_hist_from_bin_contents(histogram_name, f"{label};p_{{T}}^{{{symbol}}} [GeV];{axis_title}", self.ptz_bin_edges, contents,)

    def write_histograms(self) -> None:
        """Build and write all study histograms."""
        output = ROOT.TFile.Open(self.output_path, "RECREATE")
        if not output or output.IsZombie():
            raise OSError(f"Could not create {self.output_path}")
        self.histograms = {}
        self._calculate_performance()
        self._build_histograms()
        output.cd()
        for histogram in self.histograms.values():
            histogram.Write()
        output.Close()
        print(f"Histograms saved to {self.output_path}")
