import awkward as ak
import numpy as np
import uproot

from rdf_analysis.physics import hadrecoil_from_truth_particles, hadronic_recoil_from_clusters_python


def read_input_file(input_file, tree_name, n_events=10000):
    """Read the hadrecoil inputs and keep events with at least two muons."""
    branches = [
        "mu_pt", "mu_eta", "mu_phi",
        "tu_pt.second", "tu_phi.second",
        "u_pt.second", "u_phi.second",
        "fCluster_truthE", "fCluster_rawE", "fCluster_calE", "fCluster_MLE",
        "fClusterEta", "fClusterPhi",
        "fMETTruthPt", "fMETTruthEta", "fMETTruthPhi", "fMETTruthE",
    ]
    if n_events is None or n_events < 0:
        n_events = None

    with uproot.open(input_file) as input_root:
        tree = input_root[tree_name]
        data = tree.arrays(branches, entry_stop=n_events, library="ak", how=dict)

    # Apply the event limit before the two-muon selection, as in the other studies.
    keep = ak.num(data["mu_pt"]) >= 2
    for branch in branches:
        data[branch] = data[branch][keep]
    return data


def calculate_recoil(data):
    """Calculate the dimuon momentum and recoil components in GeV."""
    mu_pt = data["mu_pt"] / 1000.0
    mu_phi = data["mu_phi"]

    # Add the transverse momenta of the two muons.
    mu1_px = mu_pt[:, 0] * np.cos(mu_phi[:, 0])
    mu1_py = mu_pt[:, 0] * np.sin(mu_phi[:, 0])
    mu2_px = mu_pt[:, 1] * np.cos(mu_phi[:, 1])
    mu2_py = mu_pt[:, 1] * np.sin(mu_phi[:, 1])
    q_x = mu1_px + mu2_px
    q_y = mu1_py + mu2_py
    qT = np.sqrt(q_x**2 + q_y**2)
    qT_safe = np.maximum(qT, 1e-9)

    recoil = dict(data)
    recoil["mu_pt0"] = mu_pt[:, 0]
    recoil["lep_pt"] = mu_pt
    recoil["lep_phi"] = mu_phi
    recoil["q_x"] = q_x
    recoil["q_y"] = q_y
    recoil["qT"] = qT
    recoil["z_x"] = q_x
    recoil["z_y"] = q_y
    recoil["z_r"] = qT
    recoil["qhat_x"] = q_x / qT_safe
    recoil["qhat_y"] = q_y / qT_safe
    recoil["qhat_perp_x"] = -q_y / qT_safe
    recoil["qhat_perp_y"] = q_x / qT_safe

    # Truth recoil (slot 0).
    u_pt = data["tu_pt.second"][:, 0] / 1000.0
    u_phi = data["tu_phi.second"][:, 0]
    u_x = u_pt * np.cos(u_phi)
    u_y = u_pt * np.sin(u_phi)
    u_par = (u_x * q_x + u_y * q_y) / qT_safe
    u_perp = (q_x * u_y - q_y * u_x) / qT_safe

    recoil["u_pt_truth"] = u_pt
    recoil["u_x_truth"] = u_x
    recoil["u_y_truth"] = u_y
    recoil["u_mag_truth"] = np.sqrt(u_x**2 + u_y**2)
    recoil["u_par_truth"] = u_par
    recoil["u_perp_truth"] = u_perp
    recoil["u_par_plus_qT_truth"] = u_par + qT
    recoil["u_perp_truth_alt"] = u_x * (-q_y / qT_safe) + u_y * (q_x / qT_safe)

    # EM recoil (slot 9).
    u_pt = data["u_pt.second"][:, 9] / 1000.0
    u_phi = data["u_phi.second"][:, 9]
    u_x = u_pt * np.cos(u_phi)
    u_y = u_pt * np.sin(u_phi)
    u_par = (u_x * q_x + u_y * q_y) / qT_safe
    u_perp = (q_x * u_y - q_y * u_x) / qT_safe

    recoil["u_pt_em"] = u_pt
    recoil["u_x_em"] = u_x
    recoil["u_y_em"] = u_y
    recoil["u_mag_em"] = np.sqrt(u_x**2 + u_y**2)
    recoil["u_par_em"] = u_par
    recoil["u_perp_em"] = u_perp
    recoil["u_par_plus_qT_em"] = u_par + qT
    recoil["u_perp_em_alt"] = u_x * (-q_y / qT_safe) + u_y * (q_x / qT_safe)

    # LCW recoil (slot 10).
    u_pt = data["u_pt.second"][:, 10] / 1000.0
    u_phi = data["u_phi.second"][:, 10]
    u_x = u_pt * np.cos(u_phi)
    u_y = u_pt * np.sin(u_phi)
    u_par = (u_x * q_x + u_y * q_y) / qT_safe
    u_perp = (q_x * u_y - q_y * u_x) / qT_safe

    recoil["u_pt_lcw"] = u_pt
    recoil["u_x_lcw"] = u_x
    recoil["u_y_lcw"] = u_y
    recoil["u_mag_lcw"] = np.sqrt(u_x**2 + u_y**2)
    recoil["u_par_lcw"] = u_par
    recoil["u_perp_lcw"] = u_perp
    recoil["u_par_plus_qT_lcw"] = u_par + qT
    recoil["u_perp_lcw_alt"] = u_x * (-q_y / qT_safe) + u_y * (q_x / qT_safe)

    # ML recoil (slot 11).
    u_pt = data["u_pt.second"][:, 11] / 1000.0
    u_phi = data["u_phi.second"][:, 11]
    u_x = u_pt * np.cos(u_phi)
    u_y = u_pt * np.sin(u_phi)
    u_par = (u_x * q_x + u_y * q_y) / qT_safe
    u_perp = (q_x * u_y - q_y * u_x) / qT_safe

    recoil["u_pt_ml"] = u_pt
    recoil["u_x_ml"] = u_x
    recoil["u_y_ml"] = u_y
    recoil["u_mag_ml"] = np.sqrt(u_x**2 + u_y**2)
    recoil["u_par_ml"] = u_par
    recoil["u_perp_ml"] = u_perp
    recoil["u_par_plus_qT_ml"] = u_par + qT
    recoil["u_perp_ml_alt"] = u_x * (-q_y / qT_safe) + u_y * (q_x / qT_safe)

    # Additional inputs for the cluster and truth-particle studies.
    recoil["lep_eta"] = data["mu_eta"]
    recoil["clus_e_truth"] = data["fCluster_truthE"] / 1000.0
    recoil["clus_e_em"] = data["fCluster_rawE"] / 1000.0
    recoil["clus_e_lcw"] = data["fCluster_calE"] / 1000.0
    recoil["clus_e_ml"] = data["fCluster_MLE"] / 1000.0
    recoil["clus_eta"] = data["fClusterEta"]
    recoil["clus_phi"] = data["fClusterPhi"]
    recoil["met_truth_particle_pt"] = data["fMETTruthPt"] / 1000.0
    recoil["met_truth_particle_eta"] = data["fMETTruthEta"]
    recoil["met_truth_particle_phi"] = data["fMETTruthPhi"]
    recoil["met_truth_particle_e"] = data["fMETTruthE"] / 1000.0

    return recoil


def build_threshold_scan_observables(lep_pt, lep_eta, lep_phi, clus_e, clus_eta, clus_phi, thresholds):
    out = {}

    clus_e = np.asarray(clus_e)
    clus_eta = np.asarray(clus_eta)
    clus_phi = np.asarray(clus_phi)

    # For massless clusters, pT = E / cosh(eta) and |p| = pT * cosh(eta).
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
