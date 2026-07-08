#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.core import NtupleProcessor, awkward_to_numpy

Z_MASS = 91.1876
Z_WINDOW = 20.0
JET_PT_MIN = 10.0
JET_Z_DPHI_MIN = 2.8
SECOND_JET_ABS_PT_MAX = 12.0
SECOND_JET_REF_PT_FRACTION_MAX = 0.3
TRUTH_MATCH_DR = 0.4
GEV = 0.001
JET_ENERGY_HIST_MAX = 4000.0

JET_CLUSTER_ENERGY_SCALES = (
    ("cluster", "cluster E", "jet_clusterE", 1.0),
    ("truth", "cluster truth E", "jet_cluster_truthE", GEV),
    ("raw", "cluster raw E", "jet_cluster_rawE", 1.0),
    ("cal", "cluster calibrated E", "jet_cluster_calE", 1.0),
    ("ml", "cluster ML E", "jet_cluster_MLE", 1.0),
)

LEADING_CLUSTER_PT_SCALES = (
    ("raw", "cluster raw E"),
    ("cal", "cluster calibrated E"),
    ("truth", "cluster truth E"),
    ("ml", "cluster ML E"),
)


ROOT.gROOT.SetBatch(True)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot truth-matched jet pT vs reco jet pT in Z+jets events.")
    parser.add_argument("--input", required=True, action="append")
    parser.add_argument("--tree", required=True)
    parser.add_argument("--nEvents", type=int, default=-1)
    parser.add_argument("--output", default="jet_performace_in_Zjets.root")
    return parser.parse_args()


def to_numpy(values):
    return awkward_to_numpy(values)


def delta_phi(phi1, phi2):
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def dilepton_mass(lep_pt, lep_eta, lep_phi):
    # Massless two-lepton invariant mass, using the first two selected muons.
    deta = lep_eta[0] - lep_eta[1]
    dphi = delta_phi(lep_phi[0], lep_phi[1])
    mass2 = 2.0 * lep_pt[0] * lep_pt[1] * (np.cosh(deta) - np.cos(dphi))
    return np.sqrt(max(mass2, 0.0))


def z_transverse_vector(lep_pt, lep_phi):
    # Build the Z transverse momentum vector from the two leptons.
    lep_pt = np.asarray(lep_pt[:2], dtype=float)
    lep_phi = np.asarray(lep_phi[:2], dtype=float)

    z_x = np.sum(lep_pt * np.cos(lep_phi))
    z_y = np.sum(lep_pt * np.sin(lep_phi))
    z_pt = np.hypot(z_x, z_y)
    return z_x, z_y, z_pt


def z_phi(lep_pt, lep_phi):
    z_x, z_y, z_pt = z_transverse_vector(lep_pt, lep_phi)
    if z_pt == 0.0:
        return np.nan
    return np.arctan2(z_y, z_x)


def delta_phi_to_z(jet_phi, lep_pt, lep_phi):
    event_z_phi = z_phi(lep_pt, lep_phi)
    if not np.isfinite(event_z_phi):
        return np.nan
    return abs(delta_phi(jet_phi, event_z_phi))


def project_jet_on_z_axis(jet_pt, jet_phi, lep_pt, lep_phi):
    # Absolute projection of the jet pT vector onto the unit pT(Z) direction.
    z_x, z_y, z_pt = z_transverse_vector(lep_pt, lep_phi)
    if z_pt == 0.0:
        return np.nan

    jet_x = jet_pt * np.cos(jet_phi)
    jet_y = jet_pt * np.sin(jet_phi)
    return abs((jet_x * z_x + jet_y * z_y) / z_pt)


def selected_jet_indices(jet_pt):
    # Keep reco jets above threshold and sort them from leading to subleading.
    selected = np.where(np.isfinite(jet_pt) & (jet_pt > JET_PT_MIN))[0]
    return selected[np.argsort(jet_pt[selected])[::-1]]


def passes_second_jet_veto(jets, jet_pt, leading_pt_ref):
    if len(jets) < 2:
        return True

    second_jet = int(jets[1])
    max_second_jet_pt = max(SECOND_JET_ABS_PT_MAX, SECOND_JET_REF_PT_FRACTION_MAX * leading_pt_ref)
    return float(jet_pt[second_jet]) < max_second_jet_pt


def matched_truth_jet_index(jet_eta, jet_phi, tjet_eta, tjet_phi):
    # Match the reco jet to the closest truth jet in deltaR.
    if len(tjet_eta) == 0:
        return None

    deta = tjet_eta - jet_eta
    dphi = delta_phi(tjet_phi, jet_phi)
    dr = np.hypot(deta, dphi)
    match = int(np.argmin(dr))

    if dr[match] > TRUTH_MATCH_DR:
        return None
    return match


def matched_truth_jet(jet_eta, jet_phi, tjet_pt, tjet_eta, tjet_phi):
    match = matched_truth_jet_index(jet_eta, jet_phi, tjet_eta, tjet_phi)
    if match is None:
        return np.nan, np.nan
    return float(tjet_pt[match]), float(tjet_phi[match])


def make_jet_hist(name, title, y_title):
    return ROOT.TH2D(
        name,
        f"{title};Truth jet p_{{T}} [GeV];{y_title} [GeV]",
        100,
        0.0,
        200.0,
        100,
        0.0,
        200.0,
    )


def make_cluster_energy_hist(jet_label, scale_key, scale_title):
    return ROOT.TH2D(
        f"h2_{jet_label}_{scale_key}_cluster_sum_energy_vs_truth_jet_energy",
        (
            f"{jet_label.title()} reco jet summed {scale_title} vs truth matched jet energy;"
            "Truth jet energy [GeV];"
            f"Summed {scale_title} [GeV]"
        ),
        100,
        0.0,
        JET_ENERGY_HIST_MAX,
        100,
        0.0,
        JET_ENERGY_HIST_MAX,
    )


def make_leading_cluster_pt_hist(scale_key, scale_title):
    return ROOT.TH2D(
        f"h2_leading_{scale_key}_cluster_sum_pt_vs_truth_jet_pt",
        (
            f"Leading truth-matched reco jet p_{{T}} from summed {scale_title} clusters vs truth jet p_{{T}};"
            "Truth jet p_{T} [GeV];"
            f"Cluster-summed reco jet p_{{T}} from {scale_title} [GeV]"
        ),
        100,
        0.0,
        200.0,
        100,
        0.0,
        200.0,
    )


def make_jet_response_profile(name, title):
    return ROOT.TProfile(
        name,
        f"{title};Reco jet p_{{T}} [GeV];#LTp_{{T}}^{{jet}}/p_{{T}}^{{ref}}#GT",
        40,
        0.0,
        200.0,
    )


def make_z_pt_vs_leading_jet_pt_hist():
    return ROOT.TH2D(
        "h2_z_pt_vs_leading_reco_jet_pt",
        "Z p_{T} vs leading reco jet p_{T};Z p_{T} [GeV];Leading reco jet p_{T} [GeV]",
        100,
        0.0,
        200.0,
        100,
        0.0,
        200.0,
    )


def make_z_pt_vs_truth_matched_leading_jet_pt_hist():
    return ROOT.TH2D(
        "h2_z_pt_vs_truth_matched_leading_jet_pt",
        "Z p_{T} vs truth matched leading jet p_{T};Z p_{T} [GeV];Truth matched leading jet p_{T} [GeV]",
        100,
        0.0,
        200.0,
        100,
        0.0,
        200.0,
    )


def make_z_pt_vs_truth_matched_leading_reco_jet_pt_hist():
    return ROOT.TH2D(
        "h2_z_pt_vs_truth_matched_leading_reco_jet_pt",
        "Z p_{T} vs truth matched leading reco jet p_{T};Z p_{T} [GeV];Truth matched leading reco jet p_{T} [GeV]",
        100,
        0.0,
        200.0,
        100,
        0.0,
        200.0,
    )


def make_z_pt_vs_truth_matched_truth_jet_pt_hist():
    return ROOT.TH2D(
        "h2_z_pt_vs_truth_matched_truth_jet_pt",
        "Z p_{T} vs truth matched truth jet p_{T};Z p_{T} [GeV];Truth matched truth jet p_{T} [GeV]",
        100,
        0.0,
        200.0,
        100,
        0.0,
        200.0,
    )


def define_histograms():
    histograms = {
        "mll": ROOT.TH1D("h_mll", "Dilepton mass;m_{ll} [GeV];Events", 80, 50.0, 130.0),
        "z_pt_vs_leading_jet_pt": make_z_pt_vs_leading_jet_pt_hist(),
        "z_pt_vs_truth_matched_leading_jet_pt": make_z_pt_vs_truth_matched_leading_jet_pt_hist(),
        "z_pt_vs_truth_matched_leading_reco_jet_pt": make_z_pt_vs_truth_matched_leading_reco_jet_pt_hist(),
        "z_pt_vs_truth_matched_truth_jet_pt": make_z_pt_vs_truth_matched_truth_jet_pt_hist(),
        "leading_raw": make_jet_hist(
            "h2_leading_reco_jet_pt_vs_truth_jet_pt",
            "Leading reco jet p_{T} vs truth matched jet p_{T}",
            "Leading reco jet p_{T}",
        ),
        "subleading_raw": make_jet_hist(
            "h2_subleading_reco_jet_pt_vs_truth_jet_pt",
            "Subleading reco jet p_{T} vs truth matched jet p_{T}",
            "Subleading reco jet p_{T}",
        ),
        "leading_ref": make_jet_hist(
            "h2_leading_reco_jet_pt_ref_vs_truth_jet_pt_ref",
            "Leading reco jet p_{T}^{ref} vs truth matched jet p_{T}^{ref}",
            "Leading reco jet p_{T}^{ref}",
        ),
        "subleading_ref": make_jet_hist(
            "h2_subleading_reco_jet_pt_ref_vs_truth_jet_pt_ref",
            "Subleading reco jet p_{T}^{ref} vs truth matched jet p_{T}^{ref}",
            "Subleading reco jet p_{T}^{ref}",
        ),
        "jet_pt_over_ref": make_jet_response_profile(
            "hprof_reco_jet_pt_over_ref_vs_reco_jet_pt",
            "Average p_{T}^{jet}/p_{T}^{ref} vs reco jet p_{T}",
        ),
        "leading_pt_over_ref": make_jet_response_profile(
            "hprof_leading_reco_jet_pt_over_ref_vs_reco_jet_pt",
            "Average leading p_{T}^{jet}/p_{T}^{ref} vs reco jet p_{T}",
        ),
        "subleading_pt_over_ref": make_jet_response_profile(
            "hprof_subleading_reco_jet_pt_over_ref_vs_reco_jet_pt",
            "Average subleading p_{T}^{jet}/p_{T}^{ref} vs reco jet p_{T}",
        ),
    }

    for jet_label in ("leading", "subleading"):
        for scale_key, scale_title, _, _ in JET_CLUSTER_ENERGY_SCALES:
            histograms[f"{jet_label}_{scale_key}_cluster_energy"] = make_cluster_energy_hist(
                jet_label,
                scale_key,
                scale_title,
            )

    for scale_key, scale_title in LEADING_CLUSTER_PT_SCALES:
        histograms[f"leading_{scale_key}_cluster_pt"] = make_leading_cluster_pt_hist(
            scale_key,
            scale_title,
        )

    return histograms


def fill_matched_jet(
    raw_hist,
    ref_hist,
    response_profiles,
    truth_jet,
    jet,
    jet_pt,
    jet_eta,
    jet_phi,
    tjet_pt,
    tjet_phi,
    mu_pt,
    mu_phi,
):
    if truth_jet is None:
        return False

    truth_pt = float(tjet_pt[truth_jet])
    truth_phi = float(tjet_phi[truth_jet])
    reco_pt_ref = project_jet_on_z_axis(jet_pt[jet], jet_phi[jet], mu_pt, mu_phi)
    truth_pt_ref = project_jet_on_z_axis(truth_pt, truth_phi, mu_pt, mu_phi)
    if not np.isfinite(reco_pt_ref) or not np.isfinite(truth_pt_ref):
        return False

    raw_hist.Fill(truth_pt, float(jet_pt[jet]))
    ref_hist.Fill(truth_pt_ref, reco_pt_ref)
    if reco_pt_ref > 0.0:
        response = float(jet_pt[jet]) / reco_pt_ref
        for profile in response_profiles:
            profile.Fill(float(jet_pt[jet]), response)
    return True


def sum_jet_cluster_energy(jet_cluster_energy, jet, unit_scale):
    if jet >= len(jet_cluster_energy):
        return np.nan

    cluster_energy = to_numpy(jet_cluster_energy[jet]) * unit_scale
    if len(cluster_energy) == 0:
        return 0.0
    return float(np.sum(cluster_energy[np.isfinite(cluster_energy)]))


def fill_cluster_energy_histograms(histograms, jet_label, truth_energy, jet, cluster_energy_by_scale):
    if not np.isfinite(truth_energy):
        return

    for scale_key, _, _, unit_scale in JET_CLUSTER_ENERGY_SCALES:
        summed_energy = sum_jet_cluster_energy(cluster_energy_by_scale[scale_key], jet, unit_scale)
        if np.isfinite(summed_energy):
            histograms[f"{jet_label}_{scale_key}_cluster_energy"].Fill(truth_energy, summed_energy)


def sum_jet_cluster_pt(jet_cluster_energy, jet_cluster_eta, jet_cluster_phi, jet, unit_scale):
    if jet >= len(jet_cluster_energy) or jet >= len(jet_cluster_eta) or jet >= len(jet_cluster_phi):
        return np.nan

    cluster_energy = to_numpy(jet_cluster_energy[jet]) * unit_scale
    cluster_eta = to_numpy(jet_cluster_eta[jet])
    cluster_phi = to_numpy(jet_cluster_phi[jet])
    if not (len(cluster_energy) == len(cluster_eta) == len(cluster_phi)):
        return np.nan
    if len(cluster_energy) == 0:
        return 0.0

    finite = np.isfinite(cluster_energy) & np.isfinite(cluster_eta) & np.isfinite(cluster_phi)
    cluster_pt = cluster_energy[finite] / np.cosh(cluster_eta[finite])
    jet_px = np.sum(cluster_pt * np.cos(cluster_phi[finite]))
    jet_py = np.sum(cluster_pt * np.sin(cluster_phi[finite]))
    return float(np.hypot(jet_px, jet_py))


def fill_leading_cluster_pt_histograms(
    histograms,
    truth_pt,
    jet,
    cluster_energy_by_scale,
    jet_cluster_eta,
    jet_cluster_phi,
):
    if not np.isfinite(truth_pt):
        return

    scale_units = {scale_key: unit_scale for scale_key, _, _, unit_scale in JET_CLUSTER_ENERGY_SCALES}
    for scale_key, _ in LEADING_CLUSTER_PT_SCALES:
        summed_pt = sum_jet_cluster_pt(
            cluster_energy_by_scale[scale_key],
            jet_cluster_eta,
            jet_cluster_phi,
            jet,
            scale_units[scale_key],
        )
        if np.isfinite(summed_pt):
            histograms[f"leading_{scale_key}_cluster_pt"].Fill(truth_pt, summed_pt)


def event_iterator(arrays):
    return zip(
        arrays["mu_pt"],
        arrays["mu_eta"],
        arrays["mu_phi"],
        arrays["jet_pt"],
        arrays["jet_eta"],
        arrays["jet_phi"],
        arrays["tjet_pt"],
        arrays["tjet_eta"],
        arrays["tjet_phi"],
        arrays["tjet_e"],
        arrays["jet_cluster_eta"],
        arrays["jet_cluster_phi"],
        arrays["jet_clusterE"],
        arrays["jet_cluster_truthE"],
        arrays["jet_cluster_rawE"],
        arrays["jet_cluster_calE"],
        arrays["jet_cluster_MLE"],
    )


def make_counters():
    return {
        "events": 0,
        "z": 0,
        "leading_dphi": 0,
        "second_jet_veto": 0,
        "leading": 0,
        "selected_leading": 0,
        "leading_matched": 0,
        "subleading": 0,
        "subleading_matched": 0,
    }


def run_analysis(args, histograms):
    proc = NtupleProcessor(args.input, args.tree, args.nEvents)
    counters = make_counters()
    INPUT_COLUMNS = [
        "mu_pt","mu_eta","mu_phi","jet_pt","jet_eta","jet_phi","tjet_pt","tjet_eta","tjet_phi","tjet_e",
        "jet_cluster_eta","jet_cluster_phi", "jet_clusterE","jet_cluster_truthE","jet_cluster_rawE",
        "jet_cluster_calE","jet_cluster_MLE",
    ]

    for arrays in proc.iter_arrays(INPUT_COLUMNS):
        for (
            mu_pt,
            mu_eta,
            mu_phi,
            jet_pt,
            jet_eta,
            jet_phi,
            tjet_pt,
            tjet_eta,
            tjet_phi,
            tjet_e,
            jet_cluster_eta,
            jet_cluster_phi,
            jet_clusterE,
            jet_cluster_truthE,
            jet_cluster_rawE,
            jet_cluster_calE,
            jet_cluster_MLE,
        ) in event_iterator(arrays):
            if counters["events"] % 10000 == 0:
                print(f"Processing event {counters['events']}...")
            counters["events"] += 1

            # Leptons and jets are stored in MeV; convert pT branches to GeV.
            mu_pt = to_numpy(mu_pt) * GEV
            mu_eta = to_numpy(mu_eta)
            mu_phi = to_numpy(mu_phi)
            if len(mu_pt) < 2:
                continue

            # Z -> mumu selection.
            mll = dilepton_mass(mu_pt[:2], mu_eta[:2], mu_phi[:2])
            histograms["mll"].Fill(mll)
            if abs(mll - Z_MASS) > Z_WINDOW:
                continue
            counters["z"] += 1

            jet_pt = to_numpy(jet_pt) * GEV
            jet_eta = to_numpy(jet_eta)
            jet_phi = to_numpy(jet_phi)
            jets = selected_jet_indices(jet_pt)
            if len(jets) == 0:
                continue

            tjet_pt = to_numpy(tjet_pt) * GEV
            tjet_eta = to_numpy(tjet_eta)
            tjet_phi = to_numpy(tjet_phi)
            tjet_e = to_numpy(tjet_e) * GEV
            cluster_energy_by_scale = {
                "cluster": jet_clusterE,
                "truth": jet_cluster_truthE,
                "raw": jet_cluster_rawE,
                "cal": jet_cluster_calE,
                "ml": jet_cluster_MLE,
            }

            # Leading reco jet.
            leading_jet = int(jets[0])
            counters["leading"] += 1
            leading_delta_phi_z = delta_phi_to_z(jet_phi[leading_jet], mu_pt, mu_phi)
            if not np.isfinite(leading_delta_phi_z) or leading_delta_phi_z <= JET_Z_DPHI_MIN:
                continue
            counters["leading_dphi"] += 1

            leading_pt_ref = project_jet_on_z_axis(jet_pt[leading_jet], jet_phi[leading_jet], mu_pt, mu_phi)
            if not np.isfinite(leading_pt_ref):
                continue
            if not passes_second_jet_veto(jets, jet_pt, leading_pt_ref):
                continue
            counters["second_jet_veto"] += 1

            counters["selected_leading"] += 1
            _, _, z_pt = z_transverse_vector(mu_pt, mu_phi)
            histograms["z_pt_vs_leading_jet_pt"].Fill(z_pt, float(jet_pt[leading_jet]))

            leading_truth_jet = matched_truth_jet_index(
                jet_eta[leading_jet],
                jet_phi[leading_jet],
                tjet_eta,
                tjet_phi,
            )
            if leading_truth_jet is not None:
                histograms["z_pt_vs_truth_matched_leading_reco_jet_pt"].Fill(
                    z_pt,
                    float(jet_pt[leading_jet]),
                )
                histograms["z_pt_vs_truth_matched_leading_jet_pt"].Fill(
                    z_pt,
                    float(tjet_pt[leading_truth_jet]),
                )
                histograms["z_pt_vs_truth_matched_truth_jet_pt"].Fill(
                    z_pt,
                    float(tjet_pt[leading_truth_jet]),
                )
                fill_leading_cluster_pt_histograms(
                    histograms,
                    float(tjet_pt[leading_truth_jet]),
                    leading_jet,
                    cluster_energy_by_scale,
                    jet_cluster_eta,
                    jet_cluster_phi,
                )
                fill_cluster_energy_histograms(
                    histograms,
                    "leading",
                    float(tjet_e[leading_truth_jet]),
                    leading_jet,
                    cluster_energy_by_scale,
                )

            if fill_matched_jet(
                histograms["leading_raw"],
                histograms["leading_ref"],
                [histograms["jet_pt_over_ref"], histograms["leading_pt_over_ref"]],
                leading_truth_jet,
                leading_jet,
                jet_pt,
                jet_eta,
                jet_phi,
                tjet_pt,
                tjet_phi,
                mu_pt,
                mu_phi,
            ):
                counters["leading_matched"] += 1

            # Subleading reco jet, if present.
            if len(jets) > 1:
                subleading_jet = int(jets[1])
                counters["subleading"] += 1
                subleading_truth_jet = matched_truth_jet_index(
                    jet_eta[subleading_jet],
                    jet_phi[subleading_jet],
                    tjet_eta,
                    tjet_phi,
                )
                if subleading_truth_jet is not None:
                    fill_cluster_energy_histograms(
                        histograms,
                        "subleading",
                        float(tjet_e[subleading_truth_jet]),
                        subleading_jet,
                        cluster_energy_by_scale,
                    )

                if fill_matched_jet(
                    histograms["subleading_raw"],
                    histograms["subleading_ref"],
                    [histograms["jet_pt_over_ref"], histograms["subleading_pt_over_ref"]],
                    subleading_truth_jet,
                    subleading_jet,
                    jet_pt,
                    jet_eta,
                    jet_phi,
                    tjet_pt,
                    tjet_phi,
                    mu_pt,
                    mu_phi,
                ):
                    counters["subleading_matched"] += 1

    return counters


def draw_histogram(output, hist, suffix):
    # Store one canvas per 2D histogram for quick inspection.
    canvas = ROOT.TCanvas(f"c_{hist.GetName()}", "", 900, 800)
    canvas.SetRightMargin(0.15)
    hist.Draw("COLZ")
    canvas.Write()
    canvas.SaveAs(str(Path(output).with_suffix(f".{suffix}.png")))


def draw_profile(output, hist, suffix):
    canvas = ROOT.TCanvas(f"c_{hist.GetName()}", "", 900, 700)
    canvas.SetGrid()
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.8)
    hist.SetLineWidth(2)
    hist.SetMinimum(0.0)
    hist.Draw("E1")
    canvas.Write()
    canvas.SaveAs(str(Path(output).with_suffix(f".{suffix}.png")))


def write_histograms(output, histograms):
    out_file = ROOT.TFile.Open(output, "RECREATE")
    for hist in histograms.values():
        hist.Write()

    draw_histogram(output, histograms["leading_raw"], "leading_raw")
    draw_histogram(output, histograms["subleading_raw"], "subleading_raw")
    draw_histogram(output, histograms["leading_ref"], "leading_ref")
    draw_histogram(output, histograms["subleading_ref"], "subleading_ref")
    draw_histogram(output, histograms["z_pt_vs_leading_jet_pt"], "z_pt_vs_leading_jet_pt")
    draw_histogram(
        output,
        histograms["z_pt_vs_truth_matched_leading_jet_pt"],
        "z_pt_vs_truth_matched_leading_jet_pt",
    )
    draw_histogram(
        output,
        histograms["z_pt_vs_truth_matched_leading_reco_jet_pt"],
        "z_pt_vs_truth_matched_leading_reco_jet_pt",
    )
    draw_histogram(
        output,
        histograms["z_pt_vs_truth_matched_truth_jet_pt"],
        "z_pt_vs_truth_matched_truth_jet_pt",
    )
    for jet_label in ("leading", "subleading"):
        for scale_key, _, _, _ in JET_CLUSTER_ENERGY_SCALES:
            draw_histogram(
                output,
                histograms[f"{jet_label}_{scale_key}_cluster_energy"],
                f"{jet_label}_{scale_key}_cluster_energy",
            )
    for scale_key, _ in LEADING_CLUSTER_PT_SCALES:
        draw_histogram(
            output,
            histograms[f"leading_{scale_key}_cluster_pt"],
            f"leading_{scale_key}_cluster_pt",
        )
    draw_profile(output, histograms["jet_pt_over_ref"], "jet_pt_over_ref")
    draw_profile(output, histograms["leading_pt_over_ref"], "leading_pt_over_ref")
    draw_profile(output, histograms["subleading_pt_over_ref"], "subleading_pt_over_ref")
    out_file.Close()


def print_summary(counters, output):
    print(f"events processed: {counters['events']}")
    print(f"events in |mll - {Z_MASS}| < {Z_WINDOW} GeV: {counters['z']}")
    print(f"events with DeltaPhi(leading jet, Z) > {JET_Z_DPHI_MIN}: {counters['leading_dphi']}")
    print(
        "events passing second-jet veto "
        f"pT2 < max({SECOND_JET_ABS_PT_MAX} GeV, "
        f"{SECOND_JET_REF_PT_FRACTION_MAX} * pTref): {counters['second_jet_veto']}"
    )
    print(f"events with leading reco jet pT > {JET_PT_MIN} GeV: {counters['leading']}")
    print(f"selected events filled for leading reco jet: {counters['selected_leading']}")
    print(f"truth jets matched to leading reco jet: {counters['leading_matched']}")
    print(
        "selected leading reco jets without truth match: "
        f"{counters['selected_leading'] - counters['leading_matched']}"
    )
    print(f"selected events with subleading reco jet pT > {JET_PT_MIN} GeV: {counters['subleading']}")
    print(f"truth jets matched to subleading reco jet: {counters['subleading_matched']}")
    print(f"subleading reco jets without truth match: {counters['subleading'] - counters['subleading_matched']}")
    print(f"wrote {output}")


def main():
    args = parse_args()
    histograms = define_histograms()
    counters = run_analysis(args, histograms)
    write_histograms(args.output, histograms)
    print_summary(counters, args.output)


if __name__ == "__main__":
    main()
