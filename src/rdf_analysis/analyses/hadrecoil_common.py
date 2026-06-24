import awkward as ak
import numpy as np

from rdf_analysis.core import NtupleProcessor
from rdf_analysis.physics import hadrecoil_from_truth_particles, hadronic_recoil_from_clusters_python


def _safe_denominator(values):
    return np.maximum(values, 1e-9)


def _at(values, index):
    return values[:, index]


HADRECOIL_RAW_BRANCHES = {
    "mu_pt",
    "mu_eta",
    "mu_phi",
    "tu_pt/tu_pt.second",
    "tu_phi/tu_phi.second",
    "u_pt/u_pt.second",
    "u_phi/u_phi.second",
    "fCluster_truthE",
    "fCluster_rawE",
    "fCluster_calE",
    "fCluster_MLE",
    "fClusterEta",
    "fClusterPhi",
    "fMETTruthPt",
    "fMETTruthEta",
    "fMETTruthPhi",
    "fMETTruthE",
}

HADRECOIL_COLUMN_RAW_DEPENDENCIES = {
    "mu_pt0": {"mu_pt"},
    "z_x": {"mu_pt", "mu_phi"},
    "z_y": {"mu_pt", "mu_phi"},
    "z_r": {"mu_pt", "mu_phi"},
    "qT": {"mu_pt", "mu_phi"},
    "q_x": {"mu_pt", "mu_phi"},
    "q_y": {"mu_pt", "mu_phi"},
    "qhat_x": {"mu_pt", "mu_phi"},
    "qhat_y": {"mu_pt", "mu_phi"},
    "qhat_perp_x": {"mu_pt", "mu_phi"},
    "qhat_perp_y": {"mu_pt", "mu_phi"},
    "lep_pt": {"mu_pt"},
    "lep_eta": {"mu_eta"},
    "lep_phi": {"mu_phi"},
    "clus_e_truth": {"fCluster_truthE"},
    "clus_e_em": {"fCluster_rawE"},
    "clus_e_lcw": {"fCluster_calE"},
    "clus_e_ml": {"fCluster_MLE"},
    "clus_eta": {"fClusterEta"},
    "clus_phi": {"fClusterPhi"},
    "met_truth_particle_pt": {"fMETTruthPt"},
    "met_truth_particle_eta": {"fMETTruthEta"},
    "met_truth_particle_phi": {"fMETTruthPhi"},
    "met_truth_particle_e": {"fMETTruthE"},
}

HADRECOIL_SCALE_SPECS = {
    "truth": ("tu_pt/tu_pt.second", "tu_phi/tu_phi.second", 0),
    "em": ("u_pt/u_pt.second", "u_phi/u_phi.second", 9),
    "lcw": ("u_pt/u_pt.second", "u_phi/u_phi.second", 10),
    "ml": ("u_pt/u_pt.second", "u_phi/u_phi.second", 11),
}

for scale, (pt_branch, phi_branch, _) in HADRECOIL_SCALE_SPECS.items():
    HADRECOIL_COLUMN_RAW_DEPENDENCIES[f"u_pt_{scale}"] = {pt_branch}
    for prefix in (
        "u_x",
        "u_y",
        "u_mag",
    ):
        HADRECOIL_COLUMN_RAW_DEPENDENCIES[f"{prefix}_{scale}"] = {pt_branch, phi_branch}
    for prefix in (
        "u_perp",
        "u_par",
        "u_par_plus_qT",
    ):
        HADRECOIL_COLUMN_RAW_DEPENDENCIES[f"{prefix}_{scale}"] = {
            "mu_pt",
            "mu_phi",
            pt_branch,
            phi_branch,
        }
    HADRECOIL_COLUMN_RAW_DEPENDENCIES[f"u_perp_{scale}_alt"] = {
        "mu_pt",
        "mu_phi",
        pt_branch,
        phi_branch,
    }


def _hadrecoil_raw_columns(columns):
    raw_columns = {"mu_pt"}
    for column in columns:
        if column in HADRECOIL_RAW_BRANCHES:
            raw_columns.add(column)
        elif column in HADRECOIL_COLUMN_RAW_DEPENDENCIES:
            raw_columns.update(HADRECOIL_COLUMN_RAW_DEPENDENCIES[column])
        else:
            raw_columns.add(column)

    return sorted(raw_columns)


def build_hadrecoil_arrays(raw_arrays, columns):
    requested = set(columns)
    arrays = {column: raw_arrays[column] for column in requested if column in raw_arrays}

    def add(column, values):
        if column in requested:
            arrays[column] = values

    if "mu_pt" in raw_arrays:
        add("mu_pt0", _at(raw_arrays["mu_pt"], 0) / 1000.0)
        add("lep_pt", raw_arrays["mu_pt"] / 1000.0)
    if "mu_eta" in raw_arrays:
        add("lep_eta", raw_arrays["mu_eta"])
    if "mu_phi" in raw_arrays:
        add("lep_phi", raw_arrays["mu_phi"])

    z_x = z_y = z_r = qhat_perp_x = qhat_perp_y = None
    if "mu_pt" in raw_arrays and "mu_phi" in raw_arrays:
        z_x = _at(raw_arrays["mu_pt"], 0) / 1000.0 * np.cos(_at(raw_arrays["mu_phi"], 0))
        z_x = z_x + _at(raw_arrays["mu_pt"], 1) / 1000.0 * np.cos(_at(raw_arrays["mu_phi"], 1))
        z_y = _at(raw_arrays["mu_pt"], 0) / 1000.0 * np.sin(_at(raw_arrays["mu_phi"], 0))
        z_y = z_y + _at(raw_arrays["mu_pt"], 1) / 1000.0 * np.sin(_at(raw_arrays["mu_phi"], 1))
        z_r = np.sqrt(z_x * z_x + z_y * z_y)
        qhat_x = z_x / _safe_denominator(z_r)
        qhat_y = z_y / _safe_denominator(z_r)
        qhat_perp_x = -qhat_y
        qhat_perp_y = qhat_x

        add("z_x", z_x)
        add("z_y", z_y)
        add("z_r", z_r)
        add("qT", z_r)
        add("q_x", z_x)
        add("q_y", z_y)
        add("qhat_x", qhat_x)
        add("qhat_y", qhat_y)
        add("qhat_perp_x", qhat_perp_x)
        add("qhat_perp_y", qhat_perp_y)

    for scale, (pt_branch, phi_branch, index) in HADRECOIL_SCALE_SPECS.items():
        if pt_branch not in raw_arrays:
            continue

        u_pt = _at(raw_arrays[pt_branch], index) / 1000.0
        add(f"u_pt_{scale}", u_pt)

        if phi_branch not in raw_arrays:
            continue

        u_x = u_pt * np.cos(_at(raw_arrays[phi_branch], index))
        u_y = u_pt * np.sin(_at(raw_arrays[phi_branch], index))
        add(f"u_x_{scale}", u_x)
        add(f"u_y_{scale}", u_y)
        add(f"u_mag_{scale}", np.sqrt(u_x * u_x + u_y * u_y))

        if z_x is None or z_y is None or z_r is None:
            continue

        u_perp = (z_x * u_y - z_y * u_x) / _safe_denominator(z_r)
        u_par = (u_x * z_x + u_y * z_y) / _safe_denominator(z_r)
        add(f"u_perp_{scale}", u_perp)
        add(f"u_par_{scale}", u_par)
        add(f"u_par_plus_qT_{scale}", u_par + z_r)

        if qhat_perp_x is not None and qhat_perp_y is not None:
            add(f"u_perp_{scale}_alt", u_x * qhat_perp_x + u_y * qhat_perp_y)

    if "fCluster_truthE" in raw_arrays:
        add("clus_e_truth", raw_arrays["fCluster_truthE"] / 1000.0)
    if "fCluster_rawE" in raw_arrays:
        add("clus_e_em", raw_arrays["fCluster_rawE"] / 1000.0)
    if "fCluster_calE" in raw_arrays:
        add("clus_e_lcw", raw_arrays["fCluster_calE"] / 1000.0)
    if "fCluster_MLE" in raw_arrays:
        add("clus_e_ml", raw_arrays["fCluster_MLE"] / 1000.0)
    if "fClusterEta" in raw_arrays:
        add("clus_eta", raw_arrays["fClusterEta"])
    if "fClusterPhi" in raw_arrays:
        add("clus_phi", raw_arrays["fClusterPhi"])

    if "fMETTruthPt" in raw_arrays:
        add("met_truth_particle_pt", raw_arrays["fMETTruthPt"] / 1000.0)
    if "fMETTruthEta" in raw_arrays:
        add("met_truth_particle_eta", raw_arrays["fMETTruthEta"])
    if "fMETTruthPhi" in raw_arrays:
        add("met_truth_particle_phi", raw_arrays["fMETTruthPhi"])
    if "fMETTruthE" in raw_arrays:
        add("met_truth_particle_e", raw_arrays["fMETTruthE"] / 1000.0)

    missing_columns = [column for column in columns if column not in arrays]
    if missing_columns:
        raise KeyError("Unable to build hadrecoil columns: {}".format(", ".join(missing_columns)))

    return {column: arrays[column] for column in columns}


class NtupleProcessor_hadrecoil(NtupleProcessor):
    def _required_branches(self, columns):
        branches = set(super()._required_branches(columns))
        branches.add("mu_pt")
        return sorted(branches)

    def _select_events(self, raw_arrays):
        keep = ak.num(raw_arrays["mu_pt"]) > 1
        return {name: values[keep] for name, values in raw_arrays.items()}

    def materialize(self, columns):
        columns = list(columns)
        raw_columns = _hadrecoil_raw_columns(columns)
        raw_arrays = super().materialize(raw_columns)
        return build_hadrecoil_arrays(raw_arrays, columns)


def build_threshold_scan_observables(lep_pt, lep_eta, lep_phi, clus_e, clus_eta, clus_phi, thresholds):
    out = {}

    clus_e = np.asarray(clus_e)
    clus_eta = np.asarray(clus_eta)
    clus_phi = np.asarray(clus_phi)

    # Compute absolute momentum from energy and eta: p = E / cosh(eta) * cosh(eta) = E (for massless)
    # Actually: pT = E / cosh(eta), p = pT * cosh(eta) = E
    # So absolute momentum equals energy for massless approximation
    # But let's be more explicit: p = sqrt(pT^2 + pz^2) where pT = E/cosh(eta), pz = E*sinh(eta)/cosh(eta)
    # p = E/cosh(eta) * sqrt(1 + sinh^2(eta)) = E/cosh(eta) * cosh(eta) = E
    # So p = E for massless particles. But for clarity we compute it explicitly.
    clus_pt = clus_e / np.cosh(clus_eta)
    clus_p = clus_pt * np.cosh(clus_eta)  # This equals E for massless

    for thr in thresholds:
        # Apply cut on absolute momentum
        mask = clus_p > thr
        clus_e_sel = clus_e[mask]
        clus_eta_sel = clus_eta[mask]
        clus_phi_sel = clus_phi[mask]

        hadrecoil = hadronic_recoil_from_clusters_python(
            np.asarray(lep_pt),
            np.asarray(lep_eta),
            np.asarray(lep_phi),
            clus_e_sel,
            clus_eta_sel,
            clus_phi_sel,
        )

        out[thr] = {
            "u_x": hadrecoil["u_x"],
            "u_y": hadrecoil["u_y"],
            "u_pt": hadrecoil["u_pt"],
        }

    return out


def build_truth_particle_threshold_scan_observables(truth_particle_pt, truth_particle_eta, truth_particle_phi, truth_particle_e, thresholds):
    out = {}

    truth_particle_pt = np.asarray(truth_particle_pt)
    truth_particle_eta = np.asarray(truth_particle_eta)
    truth_particle_phi = np.asarray(truth_particle_phi)
    truth_particle_e = np.asarray(truth_particle_e)

    for thr in thresholds:
        hadrecoil = hadrecoil_from_truth_particles(
            truth_particle_pt,
            truth_particle_eta,
            truth_particle_phi,
            truth_particle_e,
            threshold=thr,
        )

        out[thr] = {
            "u_x": hadrecoil["u_x"],
            "u_y": hadrecoil["u_y"],
            "u_pt": hadrecoil["u_pt"],
        }

    return out
