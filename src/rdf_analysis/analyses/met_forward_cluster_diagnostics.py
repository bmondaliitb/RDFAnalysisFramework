"""Forward-cluster diagnostic observables for MET resolution studies."""

from typing import Dict

import numpy as np

from rdf_analysis.analyses.met_inputs_common import ClusterCollection, METConfig
from rdf_analysis.stats.histograms import make_numpy_hist


class ForwardClusterDiagnostics:
    """Collect per-cluster energy spectra inside and outside forward jets."""

    REGIONS = {
        "inside_jets": "inside forward jets",
        "outside_jets": "outside forward jets",
    }

    def __init__(
        self,
        energy_bins: int,
        energy_min: float,
        energy_max: float,
    ) -> None:
        self.energy_bins = energy_bins
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.histograms = self._create_histograms()

    def _create_histograms(self) -> Dict[str, object]:
        histograms = {}
        for region, region_label in self.REGIONS.items():
            for scale, (scale_label, _) in METConfig.CLUSTER_SCALES.items():
                name = f"h_forward_cluster_energy_{region}_{scale}"
                histogram = make_numpy_hist(
                    name,
                    (
                        f"Forward clusters {region_label}, {scale_label} scale;"
                        "E_{cluster} [GeV];Clusters"
                    ),
                    self.energy_bins,
                    self.energy_min,
                    self.energy_max,
                    [],
                )
                histogram.SetDirectory(0)
                histograms[name] = histogram
        return histograms

    def record(
        self,
        clusters: ClusterCollection,
        forward_mask: np.ndarray,
        away_from_jets_mask: np.ndarray,
    ) -> None:
        """Record forward-cluster energies split by jet-cone membership."""
        masks = {
            "inside_jets": forward_mask & ~away_from_jets_mask,
            "outside_jets": forward_mask & away_from_jets_mask,
        }
        for region, mask in masks.items():
            for scale, energy in clusters.energy.items():
                selected = np.asarray(energy, dtype=np.float64)[mask]
                histogram = self.histograms[
                    f"h_forward_cluster_energy_{region}_{scale}"
                ]
                for value in selected[np.isfinite(selected)]:
                    histogram.Fill(float(value))

    def build_histograms(self) -> Dict[str, object]:
        """Build one individual-cluster energy histogram per region and scale."""
        return dict(self.histograms)
