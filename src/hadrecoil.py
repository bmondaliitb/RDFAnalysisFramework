import numpy as np

def delta_phi(phi1, phi2):
    """Return phi1 - phi2 folded into [-pi, pi)."""
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    dphi = delta_phi(phi1, phi2)
    deta = eta1 - eta2
    return np.sqrt(deta * deta + dphi * dphi)


def rotate_phi(phi, angle):
    """Rotate phi by angle and wrap back to [-pi, pi)."""
    return (phi + angle + np.pi) % (2.0 * np.pi) - np.pi


def cluster_pt_from_e_eta(e, eta):
    """Massless-cluster approximation used for E/eta/phi inputs."""
    return e / np.cosh(eta)


def _sum_clusters(
    cl_e,
    cl_eta,
    cl_phi,
    threshold=0.0,
    threshold_on_pt=True,
    abs_eta_min=0.0,
    abs_eta_max=10.0,
):
    """
    Sum all accepted clusters into a hadronic-vector sum.
    Returns dict with pt/eta/phi/px/py/e for the summed vector.
    """
    px_sum = 0.0
    py_sum = 0.0
    pz_sum = 0.0
    e_sum = 0.0
    pt_list = []

    for e, eta, phi in zip(cl_e, cl_eta, cl_phi):
        if e < 0:
            continue

        abseta = abs(eta)
        if abseta < abs_eta_min or abseta > abs_eta_max:
            continue

        pt = cluster_pt_from_e_eta(e, eta)
        pt_list.append(pt)
        test_val = pt if threshold_on_pt else e
        if test_val < threshold:
            continue

        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)

        px_sum += px
        py_sum += py
        pz_sum += pz
        e_sum += e

    pt_sum = np.hypot(px_sum, py_sum)
    phi_sum = np.arctan2(py_sum, px_sum)
    eta_sum = np.arcsinh(pz_sum / pt_sum) if pt_sum > 0 else 0.0

    return {
        "px": px_sum,
        "py": py_sum,
        "pz": pz_sum,
        "e": e_sum,
        "pt": pt_sum,
        "eta": eta_sum,
        "phi": phi_sum,
    }


def _hadrecoil_uncorrected(
    lep_eta,
    lep_phi,
    cl_e,
    cl_eta,
    cl_phi,
    threshold=0.0,
    threshold_on_pt=True,
    abs_eta_min=0.0,
    abs_eta_max=10.0,
    dr_cone=0.2,
):
    """
    Mimics hadrecoil_PFO():
    1) sum all accepted clusters
    2) subtract clusters inside lepton cones
    """
    summed = _sum_clusters(
        cl_e=cl_e,
        cl_eta=cl_eta,
        cl_phi=cl_phi,
        threshold=threshold,
        threshold_on_pt=threshold_on_pt,
        abs_eta_min=abs_eta_min,
        abs_eta_max=abs_eta_max,
    )

    px_sum = summed["px"]
    py_sum = summed["py"]
    pz_sum = summed["pz"]
    e_sum = summed["e"]

    for e, eta, phi in zip(cl_e, cl_eta, cl_phi):
        if e < 0:
            continue

        abseta = abs(eta)
        if abseta < abs_eta_min or abseta > abs_eta_max:
            continue

        pt = cluster_pt_from_e_eta(e, eta)
        test_val = pt if threshold_on_pt else e
        if test_val < threshold:
            continue

        remove = False
        for l_eta, l_phi in zip(lep_eta, lep_phi):
            if delta_r(eta, phi, l_eta, l_phi) < dr_cone:
                remove = True
                break

        if not remove:
            continue

        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)

        px_sum -= px
        py_sum -= py
        pz_sum -= pz
        e_sum -= e

    pt_sum = np.hypot(px_sum, py_sum)
    phi_sum = np.arctan2(py_sum, px_sum)
    eta_sum = np.arcsinh(pz_sum / pt_sum) if pt_sum > 0 else 0.0

    return {
        "px": px_sum,
        "py": py_sum,
        "pz": pz_sum,
        "e": e_sum,
        "pt": pt_sum,
        "eta": eta_sum,
        "phi": phi_sum,
    }


def _get_random_cone_phis(hr_eta, hr_phi, lep_pt, lep_eta, lep_phi, min_dist_cone):
    """
    Mimics getRnd():
    - seed = floor(last lepton pt * 1e3)
    - for each lepton pick a random phi
      avoiding all leptons and the HR direction
    """
    if len(lep_pt) == 0:
        return []

    seed = int(np.floor(float(lep_pt[-1]) * 1e3))
    rng = np.random.default_rng(seed)

    rnd_phis = []
    for l_eta, l_phi in zip(lep_eta, lep_phi):
        while True:
            rnd_phi = rng.uniform(-np.pi, np.pi)

            is_next_to_hr = delta_r(hr_eta, hr_phi, l_eta, rnd_phi) < min_dist_cone
            is_next_to_part = False

            for other_eta, other_phi in zip(lep_eta, lep_phi):
                if delta_r(l_eta, rnd_phi, other_eta, other_phi) < min_dist_cone:
                    is_next_to_part = True
                    break

            if not is_next_to_hr and not is_next_to_part:
                rnd_phis.append(rnd_phi)
                break

    return rnd_phis


def hadronic_recoil_from_clusters_python(
    lep_pt,
    lep_eta,
    lep_phi,
    cl_e,
    cl_eta,
    cl_phi,
    threshold=0.0,
    threshold_on_pt=True,
    abs_eta_min=0.0,
    abs_eta_max=10.0,
    dr_cone=0.2,
    min_dist_cone=0.4,
    return_components=True,
):
    """
    Python implementation of the C++ hadrecoilCorr_PFO-like logic for clusters.

    Inputs
    ------
    lep_pt, lep_eta, lep_phi : array-like
        Lepton kinematics for the event.
    cl_e, cl_eta, cl_phi : array-like
        Cluster kinematics for the event. cl_e should be in GeV.
    threshold : float
        Cluster threshold value.
    threshold_on_pt : bool
        If True threshold on cluster pT (= E/cosh(eta)), else on E.
    abs_eta_min, abs_eta_max : float
        Acceptance on |eta|.
    dr_cone : float
        Lepton-cone radius used for removal and random-cone collection.
    min_dist_cone : float
        Minimum distance of random cone from leptons and recoil direction.
    return_components : bool
        If True return full dict; else return only (u_x, u_y).

    Returns
    -------
    dict or tuple
        Corrected recoil vector and bookkeeping.
    """
    lep_pt = np.asarray(lep_pt, dtype=float)
    lep_eta = np.asarray(lep_eta, dtype=float)
    lep_phi = np.asarray(lep_phi, dtype=float)

    cl_e = np.asarray(cl_e, dtype=float)
    cl_eta = np.asarray(cl_eta, dtype=float)
    cl_phi = np.asarray(cl_phi, dtype=float)

    # Step 1: uncorrected hadronic sum after lepton-cone removal
    hr = _hadrecoil_uncorrected(
        lep_eta=lep_eta,
        lep_phi=lep_phi,
        cl_e=cl_e,
        cl_eta=cl_eta,
        cl_phi=cl_phi,
        threshold=threshold,
        threshold_on_pt=threshold_on_pt,
        abs_eta_min=abs_eta_min,
        abs_eta_max=abs_eta_max,
        dr_cone=dr_cone,
    )

    # Step 2: random cone positions
    rnd_phis = _get_random_cone_phis(
        hr_eta=hr["eta"],
        hr_phi=hr["phi"],
        lep_pt=lep_pt,
        lep_eta=lep_eta,
        lep_phi=lep_phi,
        min_dist_cone=min_dist_cone,
    )

    # Step 3: UE correction by rotating random-cone content onto lepton phi
    ue_corr_per_lepton = []
    px = hr["px"]
    py = hr["py"]
    pz = hr["pz"]
    e_sum = hr["e"]

    for l_eta, l_phi, rnd_phi in zip(lep_eta, lep_phi, rnd_phis):
        ue_px = 0.0
        ue_py = 0.0
        ue_pz = 0.0
        ue_e = 0.0

        for e, eta, phi in zip(cl_e, cl_eta, cl_phi):
            if e < 0:
                continue

            abseta = abs(eta)
            if abseta < abs_eta_min or abseta > abs_eta_max:
                continue

            pt = cluster_pt_from_e_eta(e, eta)
            test_val = pt if threshold_on_pt else e
            if test_val < threshold:
                continue

            if delta_r(eta, phi, l_eta, rnd_phi) >= dr_cone:
                continue

            angle = delta_phi(l_phi, rnd_phi)
            phi_rot = rotate_phi(phi, angle)

            px_i = pt * np.cos(phi_rot)
            py_i = pt * np.sin(phi_rot)
            pz_i = pt * np.sinh(eta)

            px += px_i
            py += py_i
            pz += pz_i
            e_sum += e

            ue_px += px_i
            ue_py += py_i
            ue_pz += pz_i
            ue_e += e

        ue_pt = np.hypot(ue_px, ue_py)
        ue_phi = np.arctan2(ue_py, ue_px) if ue_pt > 0 else 0.0
        ue_eta = np.arcsinh(ue_pz / ue_pt) if ue_pt > 0 else 0.0

        ue_corr_per_lepton.append(
            {
                "px": ue_px,
                "py": ue_py,
                "pz": ue_pz,
                "e": ue_e,
                "pt": ue_pt,
                "eta": ue_eta,
                "phi": ue_phi,
                "rnd_phi": rnd_phi,
            }
        )

    # Final corrected hadronic vector sum
    hr_pt = np.hypot(px, py)
    hr_phi = np.arctan2(py, px) if hr_pt > 0 else 0.0
    hr_eta = np.arcsinh(pz / hr_pt) if hr_pt > 0 else 0.0

    # Recoil convention:
    # C++ hadrecoilCorr_PFO returns the hadronic-vector sum.
    # If you want recoil u = -sum pT, flip the sign here.
    u_x = -px
    u_y = -py
    u_pt = np.hypot(u_x, u_y)
    u_phi = np.arctan2(u_y, u_x) if u_pt > 0 else 0.0

    result = {
        "hr_px": px,
        "hr_py": py,
        "hr_pt": hr_pt,
        "hr_eta": hr_eta,
        "hr_phi": hr_phi,
        "u_x": u_x,
        "u_y": u_y,
        "u_pt": u_pt,
        "u_phi": u_phi,
        "random_cone_phis": rnd_phis,
        "ue_corr_per_lepton": ue_corr_per_lepton,
    }

    if return_components:
        return result
    return u_x, u_y


def hadrecoil_from_truth_particles(
    event_truth_pt, event_truth_eta, event_truth_phi, return_components=True
):
    """
    Recompute truth MET components with TruthType::Int convention (weight = -1),
    and expose recoil as u = -MET.
    """
    pt = np.asarray(event_truth_pt, dtype=float)
    eta = np.asarray(event_truth_eta, dtype=float)  # kept for interface consistency
    phi = np.asarray(event_truth_phi, dtype=float)

    if not (pt.shape == eta.shape == phi.shape):
        raise ValueError("event_truth_pt, event_truth_eta, event_truth_phi must match in shape")

    met_px = -np.sum(pt * np.cos(phi))
    met_py = -np.sum(pt * np.sin(phi))
    met_sumet = np.sum(np.abs(pt))

    met_pt = np.hypot(met_px, met_py)
    met_phi = np.arctan2(met_py, met_px) if met_pt > 0 else 0.0

    u_x = -met_px
    u_y = -met_py
    u_pt = np.hypot(u_x, u_y)
    u_phi = np.arctan2(u_y, u_x) if u_pt > 0 else 0.0

    result = {
        "met_px": met_px,
        "met_py": met_py,
        "met_pt": met_pt,
        "met_phi": met_phi,
        "met_sumet": met_sumet,
        "u_x": u_x,
        "u_y": u_y,
        "u_pt": u_pt,
        "u_phi": u_phi,
        "n_truth_particles": int(pt.size),
    }

    return result if return_components else (u_x, u_y)


def compute_sum_et_from_energy(energy, eta=None):
    """
    Compute sum_et from particle/cluster energy.
    sum_et is scalar sum of particle/cluster transverse momentum.

    If eta is provided:
        et_i = E_i / cosh(eta_i)   (massless approximation)
    If eta is not provided:
        et_i = E_i

    Parameters
    ----------
    energy : array-like
        Per-particle energies.
    eta : array-like or None
        Per-particle pseudorapidities.

    Returns
    -------
    float
        Scalar sum_et.
    """
    energy = np.asarray(energy, dtype=float)

    if eta is None:
        et = np.abs(energy)
    else:
        eta = np.asarray(eta, dtype=float)
        if energy.shape != eta.shape:
            raise ValueError("energy and eta must match in shape")
        et = np.abs(energy / np.cosh(eta))

    return float(np.sum(et))

def compute_sum_et(pt):
    pt = np.asarray(pt, dtype=float)
    return float(np.sum(np.abs(pt)))

