"""Shared physics objects for MET-input validation and resolution studies."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from rdf_analysis.core import awkward_to_numpy


class METConfig:
    """Branch names and units shared by both MET analyses."""

    # All vector components and pT values are converted from the ntuple's MeV
    # convention to GeV at the point where the corresponding object is built.
    GEV = 0.001

    MET_INPUT_BRANCHES = {
        "electron_pt": "met_input_el_pt",
        "electron_phi": "met_input_el_phi",
        "electron_weight": "met_input_el_weight",
        "muon_pt": "met_input_mu_pt",
        "muon_phi": "met_input_mu_phi",
        "muon_weight": "met_input_mu_weight",
        "jet_pt": "met_input_jet_pt",
        "jet_eta": "met_input_jet_eta",
        "jet_phi": "met_input_jet_phi",
        "jet_weight": "met_input_jet_weight",
        "soft_mpx": "met_softtrk_mpx",
        "soft_mpy": "met_softtrk_mpy",
    }

    MET_MAKER_BRANCHES = {
        "mpx": "met_maker_mpx",
        "mpy": "met_maker_mpy",
    }

    TRUTH_MET_PT_BRANCH = "tu_pt.second"
    TRUTH_MET_PHI_BRANCH = "tu_phi.second"
    TRUTH_PARTICLE_BRANCHES = {
        "pt": "fMETTruthPt",
        "phi": "fMETTruthPhi",
    }
    # The flattened std::map values follow the maps' lexicographic key order.
    TRUTH_MET_COMPONENT_INDICES = {
        "Int": 0,
        "IntMuons": 1,
        "IntOut": 2,
        "NonInt": 3,
    }
    # Select the truth-MET component used as the resolution reference.  The
    # flattened vector follows TRUTH_MET_COMPONENT_INDICES above; changing this
    # tuple changes the physics reference without changing event processing.
    #TRUTH_MET_REFERENCE_COMPONENTS = ("Int", "IntMuons")
    TRUTH_MET_REFERENCE_COMPONENTS = ("NonInt",)

    LEPTON_BRANCHES = {
        "pt": "mu_pt",
        "eta": "mu_eta",
        "phi": "mu_phi",
    }

    CLUSTER_DIRECTION_BRANCHES = {
        "eta": "fClusterEta",
        "phi": "fClusterPhi",
    }

    CLUSTER_SCALES = {
        "em": ("EM", "fCluster_rawE"),
        "lcw": ("LCW", "fCluster_calE"),
        "ml": ("ML", "fCluster_MLE"),
        "ml_forward": ("ML forward", "cluster_e_ML_forward"),
        "truth": ("truth", "fCluster_truthE"),
    }

    """
    The units are really messed up; all cluster energies are in MeV; but ML_forward is in GeV
    """
    CLUSTER_SCALE_FACTORS = {
        "em": GEV,
        "lcw": GEV,
        "ml": GEV,
        "ml_forward": 1.0,
        "truth": GEV
    }


@dataclass(frozen=True)
class TransverseVector:
    """Two-dimensional momentum vector in GeV."""

    x: float
    y: float

    @property
    def pt(self) -> float:
        return float(np.hypot(self.x, self.y))

    @property
    def phi(self) -> float:
        return float(np.arctan2(self.y, self.x))

    def dot(self, other: "TransverseVector") -> float:
        return float(self.x * other.x + self.y * other.y)

    def unit(self) -> "TransverseVector":
        # A zero/invalid vector has no direction; callers should reject the
        # resulting NaNs before using it for a projection.
        if not np.isfinite(self.pt) or self.pt == 0.0:
            return TransverseVector(np.nan, np.nan)
        return TransverseVector(self.x / self.pt, self.y / self.pt)

    def __add__(self, other: "TransverseVector") -> "TransverseVector":
        return TransverseVector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "TransverseVector") -> "TransverseVector":
        return TransverseVector(self.x - other.x, self.y - other.y)

    def __neg__(self) -> "TransverseVector":
        return TransverseVector(-self.x, -self.y)

    @classmethod
    def zero(cls) -> "TransverseVector":
        return cls(0.0, 0.0)


@dataclass(frozen=True)
class TruthMET:
    """Truth-MET magnitude and, when available, its azimuth."""

    pt: float
    phi: float = np.nan

    @property
    def has_direction(self) -> bool:
        return bool(np.isfinite(self.phi))

    @property
    def vector(self) -> Optional[TransverseVector]:
        if not self.has_direction:
            return None
        return TransverseVector(
            self.pt * float(np.cos(self.phi)),
            self.pt * float(np.sin(self.phi)),
        )


@dataclass
class METInputTerm:
    """One weighted hard-object term in the MET visible-momentum sum."""

    pt: np.ndarray
    phi: np.ndarray
    weight: np.ndarray
    eta: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.pt = np.asarray(self.pt, dtype=np.float64)
        self.phi = np.asarray(self.phi, dtype=np.float64)
        self.weight = np.asarray(self.weight, dtype=np.float64)
        if self.eta is not None:
            self.eta = np.asarray(self.eta, dtype=np.float64)

    @property
    def count(self) -> int:
        return len(self.pt)

    def has_consistent_size(self) -> bool:
        sizes = [len(self.pt), len(self.phi), len(self.weight)]
        if self.eta is not None:
            sizes.append(len(self.eta))
        return len(set(sizes)) == 1

    def visible_vector(
        self,
        indices: Optional[Sequence[int]] = None,
    ) -> TransverseVector:
        """Return sum_i weight_i * pT_i for this term."""
        selected = (
            np.arange(self.count, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        pt = self.pt[selected]
        phi = self.phi[selected]
        weight = self.weight[selected]
        valid = np.isfinite(pt) & np.isfinite(phi) & np.isfinite(weight)
        if not np.any(valid):
            return TransverseVector.zero()

        weighted_pt = pt[valid] * weight[valid]
        return TransverseVector(
            float(np.sum(weighted_pt * np.cos(phi[valid]))),
            float(np.sum(weighted_pt * np.sin(phi[valid]))),
        )


@dataclass
class METInputs:
    """All stored terms needed to independently reconstruct MET."""

    electrons: METInputTerm
    muons: METInputTerm
    jets: METInputTerm
    soft_term: TransverseVector

    def has_consistent_size(self ) -> bool:
        sizes_are_consistent = (
            self.electrons.has_consistent_size()
            and self.muons.has_consistent_size()
            and self.jets.has_consistent_size()
            and np.isfinite(self.soft_term.x)
            and np.isfinite(self.soft_term.y)
        )
        return sizes_are_consistent

    def calculate_met(self) -> TransverseVector:
        """
        Calculate MET from visible hard terms and the stored soft MET term.

        met_softtrk_mpx/mpy already carry the missing-momentum sign:
          MET = -(electrons + muons + jets) + soft MET.
        """
        # Each stored hard-object term is visible momentum; MET uses its
        # negative, while the soft branch already contains the MET sign.
        hard_term = (
            self.electrons.visible_vector()
            + self.muons.visible_vector()
            + self.jets.visible_vector()
        )
        return -hard_term + self.soft_term

    def forward_jet_indices(
        self,
        eta_min: float,
        eta_max: float,
    ) -> np.ndarray:
        abs_eta = np.abs(self.jets.eta)
        # Keep the original jet indices so the same subset can be summed with
        # its stored weights later (e.g. when removing forward jets).
        return np.flatnonzero(
            np.isfinite(abs_eta)
            & (abs_eta >= eta_min)
            & (abs_eta < eta_max)
        )



@dataclass
class LeptonCollection:
    """Muon collection used to reconstruct the reference Z boson."""

    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray

    @property
    def count(self) -> int:
        return len(self.pt)

    @property
    def z_vector(self) -> TransverseVector:
        # The Z candidate is reconstructed from the first two muons, matching
        # the selection convention in met_resolution_study_zmumu.py.
        pt = self.pt[:2]
        phi = self.phi[:2]
        return TransverseVector(
            float(np.sum(pt * np.cos(phi))),
            float(np.sum(pt * np.sin(phi))),
        )

    @property
    def z_eta(self) -> float:
        """Pseudorapidity of the dimuon momentum, using massless muons."""
        if self.count < 2:
            return np.nan
        z_pt = self.z_vector.pt
        z_pz = float(np.sum(self.pt[:2] * np.sinh(self.eta[:2])))
        if not np.isfinite(z_pt) or not np.isfinite(z_pz) or z_pt == 0.0:
            return np.nan
        return float(np.arcsinh(z_pz / z_pt))

    @property
    def dilepton_mass(self) -> float:
        # Use the massless-lepton invariant-mass expression; Δφ is wrapped to
        # avoid a discontinuity at ±π.
        if self.count < 2:
            return np.nan
        delta_eta = self.eta[0] - self.eta[1]
        delta_phi = wrap_delta_phi(self.phi[0] - self.phi[1])
        mass_squared = 2.0 * self.pt[0] * self.pt[1] * (
            np.cosh(delta_eta) - np.cos(delta_phi)
        )
        return float(np.sqrt(max(mass_squared, 0.0)))


@dataclass
class ClusterCollection:
    """Global topo-clusters with alternative energy calibrations."""

    eta: np.ndarray
    phi: np.ndarray
    energy: Dict[str, np.ndarray]

    @property
    def count(self) -> int:
        return len(self.eta)

    def has_consistent_size(self) -> bool:
        return (
            len(self.phi) == self.count
            and all(len(values) == self.count for values in self.energy.values())
        )

    def away_from_jets( self, jet_eta: np.ndarray, jet_phi: np.ndarray, radius: float,) -> np.ndarray:
        """Return True for clusters at least `radius` away from every jet."""

        if self.count == 0:
            return np.zeros(0, dtype=bool)

        if len(jet_eta) == 0:
            return np.ones(self.count, dtype=bool)

        cluster_eta = self.eta[:, None]
        cluster_phi = self.phi[:, None]

        jets_eta = jet_eta[None, :]
        jets_phi = jet_phi[None, :]

        eta_difference = cluster_eta - jets_eta

        raw_phi_difference = cluster_phi - jets_phi
        phi_difference = np.arctan2(
            np.sin(raw_phi_difference),
            np.cos(raw_phi_difference),
        )

        distance_squared = (
                eta_difference ** 2
                + phi_difference ** 2
        )

        # A cluster is retained only when its closest forward jet is outside
        # the overlap cone; this mask is later combined with the η acceptance.
        closest_jet_distance_squared = np.min(
            distance_squared,
            axis=1,
        )

        return closest_jet_distance_squared >= radius ** 2

    def in_abs_eta_range(
        self,
        eta_min: float,
        eta_max: float,
    ) -> np.ndarray:
        """Mask finite clusters in eta_min <= |eta| < eta_max."""
        abs_eta = np.abs(self.eta)
        return (
            np.isfinite(abs_eta)
            & (abs_eta >= eta_min)
            & (abs_eta < eta_max)
        )

    def visible_vectors(self, mask: np.ndarray) -> Dict[str, TransverseVector]:
        """Sum cluster pT vectors for every configured energy scale."""
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != self.count:
            raise ValueError("Cluster mask has the wrong size")

        vectors = {}
        # Convert cluster energy to transverse momentum with pT = E/cosh(η),
        # then sum each calibration scale independently.
        for scale, energy in self.energy.items():
            valid = (
                mask
                & np.isfinite(self.eta)
                & np.isfinite(self.phi)
                & np.isfinite(energy)
            )
            if not np.any(valid):
                vectors[scale] = TransverseVector.zero()
                continue

            cluster_pt = energy[valid] / np.cosh(self.eta[valid])
            vectors[scale] = TransverseVector(
                float(np.sum(cluster_pt * np.cos(self.phi[valid]))),
                float(np.sum(cluster_pt * np.sin(self.phi[valid]))),
            )
        return vectors


@dataclass
class ObservableValues:
    """Event-level values retained for one MET scenario."""

    ptz: List[float] = field(default_factory=list)
    pz: List[float] = field(default_factory=list)
    met: List[float] = field(default_factory=list)

    def append(
        self,
        z_vector: TransverseVector,
        met: TransverseVector,
        projection_met: Optional[TransverseVector] = None,
    ) -> None:
        """Store MET kinematics and a possibly truth-subtracted projection."""
        self.ptz.append(z_vector.pt)
        projected = met if projection_met is None else projection_met
        self.pz.append(projected.dot(z_vector.unit()))
        self.met.append(met.pt)


    def convert_to_numpy(self) -> None:
        for name in (
            "ptz",
            "pz",
            "met",
        ):
            setattr(
                self,
                name,
                np.asarray(getattr(self, name), dtype=np.float64),
            )


@dataclass
class BinnedPerformance:
    """Paper-style Z response and resolution in pTZ bins."""

    entries: np.ndarray
    mean_pz: np.ndarray
    response: np.ndarray
    raw_resolution: np.ndarray
    resolution: np.ndarray


def wrap_delta_phi(delta_phi: float) -> float:
    """Wrap an angular difference into [-pi, pi]."""
    return float(np.arctan2(np.sin(delta_phi), np.cos(delta_phi)))


def calculate_binned_performance(
    ptz: np.ndarray,
    pz: np.ndarray,
    bin_edges: Iterable[float],
) -> BinnedPerformance:
    """Calculate Z-balance performance versus pTZ."""
    ptz = np.asarray(ptz, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    finite = np.isfinite(ptz) & np.isfinite(pz)

    entries = []
    mean_pz = []
    response = []
    raw_resolution = []
    resolution = []

    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        upper = ptz <= high if index == len(edges) - 2 else ptz < high
        selected = finite & (ptz >= low) & upper
        count = int(np.count_nonzero(selected))
        entries.append(count)
        if count == 0:
            mean_pz.append(np.nan)
            response.append(np.nan)
            raw_resolution.append(np.nan)
            resolution.append(np.nan)
            continue

        bin_mean_ptz = float(np.mean(ptz[selected]))
        bin_mean_pz = float(np.mean(pz[selected]))
        bin_response = (
            1.0 + bin_mean_pz / bin_mean_ptz
            if bin_mean_ptz != 0.0
            else np.nan
        )
        bin_raw_resolution = float(np.std(pz[selected]))
        bin_resolution = (
            bin_raw_resolution / bin_response
            if np.isfinite(bin_response) and bin_response != 0.0
            else np.nan
        )

        mean_pz.append(bin_mean_pz)
        response.append(bin_response)
        raw_resolution.append(bin_raw_resolution)
        resolution.append(bin_resolution)

    return BinnedPerformance(
        entries=np.asarray(entries, dtype=np.int64),
        mean_pz=np.asarray(mean_pz, dtype=np.float64),
        response=np.asarray(response, dtype=np.float64),
        raw_resolution=np.asarray(raw_resolution, dtype=np.float64),
        resolution=np.asarray(resolution, dtype=np.float64),
    )


def build_met_inputs(
    arrays: Mapping[str, object],
    event: int,
    include_jet_eta: bool = True,
) -> METInputs:
    """Build one event's MET input object from an awkward-array chunk."""
    branch = METConfig.MET_INPUT_BRANCHES
    return METInputs(
        electrons=METInputTerm(
            pt=awkward_to_numpy(arrays[branch["electron_pt"]][event])
            * METConfig.GEV,
            phi=awkward_to_numpy(arrays[branch["electron_phi"]][event]),
            weight=awkward_to_numpy(arrays[branch["electron_weight"]][event]),
        ),
        muons=METInputTerm(
            pt=awkward_to_numpy(arrays[branch["muon_pt"]][event])
            * METConfig.GEV,
            phi=awkward_to_numpy(arrays[branch["muon_phi"]][event]),
            weight=awkward_to_numpy(arrays[branch["muon_weight"]][event]),
        ),
        jets=METInputTerm(
            pt=awkward_to_numpy(arrays[branch["jet_pt"]][event])
            * METConfig.GEV,
            eta=(
                awkward_to_numpy(arrays[branch["jet_eta"]][event])
                if include_jet_eta
                else None
            ),
            phi=awkward_to_numpy(arrays[branch["jet_phi"]][event]),
            weight=awkward_to_numpy(arrays[branch["jet_weight"]][event]),
        ),
        soft_term=TransverseVector(
            float(arrays[branch["soft_mpx"]][event]) * METConfig.GEV,
            float(arrays[branch["soft_mpy"]][event]) * METConfig.GEV,
        ),
    )


def build_maker_met(
    arrays: Mapping[str, object],
    event: int,
) -> TransverseVector:
    """Build the stored METMaker vector in GeV."""
    branch = METConfig.MET_MAKER_BRANCHES
    return TransverseVector(
        float(arrays[branch["mpx"]][event]) * METConfig.GEV,
        float(arrays[branch["mpy"]][event]) * METConfig.GEV,
    )


def build_truth_met(
    arrays: Mapping[str, object],
    event: int,
    pt_branch: str = METConfig.TRUTH_MET_PT_BRANCH,
    phi_branch: str = METConfig.TRUTH_MET_PHI_BRANCH,
) -> Optional[TruthMET]:
    """Build the Int + IntMuons truth-MET reference vector."""
    pt_values = awkward_to_numpy(arrays[pt_branch][event])
    phi_values = awkward_to_numpy(arrays[phi_branch][event])

    reference = TransverseVector.zero()
    for component in METConfig.TRUTH_MET_REFERENCE_COMPONENTS:
        index = METConfig.TRUTH_MET_COMPONENT_INDICES[component]
        pt = float(pt_values[index]) * METConfig.GEV
        phi = float(phi_values[index])
        if not np.isfinite(pt) or not np.isfinite(phi):
            return None
        reference += TransverseVector(
            pt * float(np.cos(phi)),
            pt * float(np.sin(phi)),
        )

    return TruthMET(pt=reference.pt, phi=reference.phi)

def build_leptons(
    arrays: Mapping[str, object],
    event: int,
) -> LeptonCollection:
    """Build the muon collection in GeV."""
    branch = METConfig.LEPTON_BRANCHES
    return LeptonCollection(
        pt=awkward_to_numpy(arrays[branch["pt"]][event]) * METConfig.GEV,
        eta=awkward_to_numpy(arrays[branch["eta"]][event]),
        phi=awkward_to_numpy(arrays[branch["phi"]][event]),
    )


def build_clusters(
    arrays: Mapping[str, object],
    event: int,
) -> ClusterCollection:
    """Build the global cluster collection, converting energies to GeV."""
    direction = METConfig.CLUSTER_DIRECTION_BRANCHES
    return ClusterCollection(
        eta=awkward_to_numpy(arrays[direction["eta"]][event]),
        phi=awkward_to_numpy(arrays[direction["phi"]][event]),
        energy={
            scale: awkward_to_numpy(arrays[branch][event]) * METConfig.CLUSTER_SCALE_FACTORS[scale] # applying correct unit (unnecessary complication)
            for scale, (_, branch) in METConfig.CLUSTER_SCALES.items()
        },
    )
