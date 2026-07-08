#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import ROOT

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.hadrecoil_common import NtupleProcessor_hadrecoil
from rdf_analysis.stats import make_hist_from_bin_contents, make_numpy_hist

MET_INDEX = 8
DEFAULT_PTZ_BIN_EDGES = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 250]


def parse_bin_edges(text):
    edges = np.asarray([float(value) for value in text.split(",")], dtype=float)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise argparse.ArgumentTypeError("bin edges must be a comma-separated increasing list")
    return edges


def z_transverse_vector(lep_pt, lep_phi):
    lep_pt = np.asarray(lep_pt[:2], dtype=float)
    lep_phi = np.asarray(lep_phi[:2], dtype=float)

    z_x = np.sum(lep_pt * np.cos(lep_phi))
    z_y = np.sum(lep_pt * np.sin(lep_phi))
    z_pt = np.hypot(z_x, z_y)
    return z_x, z_y, z_pt


def project_met_on_z_axis(lep_pt, lep_phi, met_x, met_y):
    """Compute pTmiss projected onto the Z-boson transverse direction."""
    z_x, z_y, z_pt = z_transverse_vector(lep_pt, lep_phi)
    if z_pt == 0.0:
        return np.nan

    return (met_x * z_x + met_y * z_y) / z_pt


def met_from_recoil_and_z(lep_pt, lep_phi, recoil_pt, recoil_phi):
    z_x, z_y, z_pt = z_transverse_vector(lep_pt, lep_phi)
    recoil_x = recoil_pt * np.cos(recoil_phi)
    recoil_y = recoil_pt * np.sin(recoil_phi)

    # pTmiss = -(uT + pT(ll))
    met_x = -(recoil_x + z_x)
    met_y = -(recoil_y + z_y)
    return met_x, met_y, z_pt


def mean_in_ptz_bins(ptz, values, ptz_bin_edges):
    means = []
    for low, high in zip(ptz_bin_edges[:-1], ptz_bin_edges[1:]):
        in_bin = (low <= ptz) & (ptz < high) & np.isfinite(values)
        means.append(np.mean(values[in_bin]) if np.any(in_bin) else np.nan)
    return np.asarray(means, dtype=float)


def save_pz_histograms(output_file, ptz, pz_ref, pz_ml_forward, pz_diff, ptz_bin_edges, pz_bins, pz_min, pz_max):
    out_file = ROOT.TFile.Open(output_file, "RECREATE")
    if not out_file or out_file.IsZombie():
        raise OSError(f"Unable to open output ROOT file: {output_file}")

    ptz = np.asarray(ptz, dtype=float)
    pz_ref = np.asarray(pz_ref, dtype=float)
    pz_ml_forward = np.asarray(pz_ml_forward, dtype=float)
    pz_diff = np.asarray(pz_diff, dtype=float)
    finite = np.isfinite(ptz) & np.isfinite(pz_ref) & np.isfinite(pz_ml_forward) & np.isfinite(pz_diff)

    make_numpy_hist("h_pTZ", "p_{T}^{Z};p_{T}^{Z} [GeV];Events", ptz_bin_edges, ptz[finite]).Write()
    make_hist_from_bin_contents(
        "h_mean_pz_ref_vs_pTZ",
        "<P^{Z}_{ref}> vs p_{T}^{Z};p_{T}^{Z} [GeV];<P^{Z}_{ref}> [GeV]",
        ptz_bin_edges,
        mean_in_ptz_bins(ptz[finite], pz_ref[finite], ptz_bin_edges),
    ).Write()
    make_hist_from_bin_contents(
        "h_mean_pz_ml_forward_vs_pTZ",
        "<P^{Z}_{ML-forward}> vs p_{T}^{Z};p_{T}^{Z} [GeV];<P^{Z}_{ML-forward}> [GeV]",
        ptz_bin_edges,
        mean_in_ptz_bins(ptz[finite], pz_ml_forward[finite], ptz_bin_edges),
    ).Write()
    make_hist_from_bin_contents(
        "h_mean_pz_diff_vs_pTZ",
        "<P^{Z}_{ML-forward} - P^{Z}_{ref}> vs p_{T}^{Z};p_{T}^{Z} [GeV];<#Delta P^{Z}> [GeV]",
        ptz_bin_edges,
        mean_in_ptz_bins(ptz[finite], pz_diff[finite], ptz_bin_edges),
    ).Write()

    h2_ref = ROOT.TH2D(
        "h2_pz_ref_vs_pTZ",
        "Reference P^{Z} vs p_{T}^{Z};p_{T}^{Z} [GeV];P^{Z}_{ref} [GeV]",
        len(ptz_bin_edges) - 1,
        np.asarray(ptz_bin_edges, dtype=np.float64),
        pz_bins,
        pz_min,
        pz_max,
    )
    h2_ml = ROOT.TH2D(
        "h2_pz_ml_forward_vs_pTZ",
        "ML-forward P^{Z} vs p_{T}^{Z};p_{T}^{Z} [GeV];P^{Z}_{ML-forward} [GeV]",
        len(ptz_bin_edges) - 1,
        np.asarray(ptz_bin_edges, dtype=np.float64),
        pz_bins,
        pz_min,
        pz_max,
    )
    h2_diff = ROOT.TH2D(
        "h2_pz_diff_vs_pTZ",
        "P^{Z}_{ML-forward} - P^{Z}_{ref} vs p_{T}^{Z};p_{T}^{Z} [GeV];#Delta P^{Z} [GeV]",
        len(ptz_bin_edges) - 1,
        np.asarray(ptz_bin_edges, dtype=np.float64),
        pz_bins,
        pz_min,
        pz_max,
    )

    for x, ref, ml, diff in zip(ptz[finite], pz_ref[finite], pz_ml_forward[finite], pz_diff[finite]):
        h2_ref.Fill(x, ref)
        h2_ml.Fill(x, ml)
        h2_diff.Fill(x, diff)

    h2_ref.Write()
    h2_ml.Write()
    h2_diff.Write()

    hist_dir = out_file.mkdir("pz_by_pTZ")
    hist_dir.cd()
    for i, (low, high) in enumerate(zip(ptz_bin_edges[:-1], ptz_bin_edges[1:])):
        in_bin = finite & (low <= ptz) & (ptz < high)
        suffix = f"ptz_{low:g}_{high:g}".replace(".", "p")
        title_suffix = f"{low:g} <= p_{{T}}^{{Z}} < {high:g} GeV"

        make_numpy_hist(
            f"h_pz_ref_{suffix}",
            f"Reference P^{{Z}} for {title_suffix};P^{{Z}} [GeV];Events",
            pz_bins,
            pz_min,
            pz_max,
            pz_ref[in_bin],
        ).Write()
        make_numpy_hist(
            f"h_pz_ml_forward_{suffix}",
            f"ML-forward P^{{Z}} for {title_suffix};P^{{Z}} [GeV];Events",
            pz_bins,
            pz_min,
            pz_max,
            pz_ml_forward[in_bin],
        ).Write()
        make_numpy_hist(
            f"h_pz_diff_{suffix}",
            f"P^{{Z}} difference for {title_suffix};#Delta P^{{Z}} [GeV];Events",
            pz_bins,
            pz_min,
            pz_max,
            pz_diff[in_bin],
        ).Write()

        print(f"pTZ bin {i}: {title_suffix}, events={np.count_nonzero(in_bin)}")

    out_file.Close()
    print(f"histograms saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Replace forward EM clusters by ML clusters in stored MET.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--nEvents", type=int, default=10000)
    parser.add_argument("--eta-min", type=float, default=2.5)
    parser.add_argument("--eta-max", type=float, default=4.9)
    parser.add_argument("--output", default="validate_met_pz_histograms.root")
    parser.add_argument("--ptz-bins", type=parse_bin_edges, default=np.asarray(DEFAULT_PTZ_BIN_EDGES, dtype=float))
    parser.add_argument("--pz-bins", type=int, default=80)
    parser.add_argument("--pz-min", type=float, default=-100.0)
    parser.add_argument("--pz-max", type=float, default=100.0)
    args = parser.parse_args()

    cols = [
        "lep_pt",
        "lep_phi",
        "u_pt.second",
        "u_phi.second",
        "clus_e_em",
        "clus_e_ml",
        "clus_eta",
        "clus_phi",
    ]
    arrays = NtupleProcessor_hadrecoil(args.input, args.tree, args.nEvents).to_numpy(cols)

    met_ref = []
    met_ml_forward = []
    ptz = []
    pz_ref = []
    pz_ml_forward = []

    events = zip(
        arrays["lep_pt"],
        arrays["lep_phi"],
        arrays["u_pt.second"],
        arrays["u_phi.second"],
        arrays["clus_e_em"],
        arrays["clus_e_ml"],
        arrays["clus_eta"],
        arrays["clus_phi"],
    )

    for event, (lep_pt, lep_phi, u_pt, u_phi, em_e, ml_e, clus_eta, clus_phi) in enumerate(events):
        if event % 1000 == 0:
            print(f"Processing event {event}...")

        recoil_pt = u_pt[MET_INDEX] / 1000.0
        recoil_phi = u_phi[MET_INDEX]
        met_x, met_y, event_ptz = met_from_recoil_and_z(lep_pt, lep_phi, recoil_pt, recoil_phi)
        met_pt = np.hypot(met_x, met_y)

        new_x = met_x
        new_y = met_y
        n_forward = 0

        #for e_em, e_ml, eta, phi in zip(em_e, ml_e, clus_eta, clus_phi):
        #    if args.eta_min <= abs(eta) < args.eta_max:
        #        em_pt = e_em / np.cosh(eta)
        #        ml_pt = e_ml / np.cosh(eta)

        #        # MET = -sum(visible). Replacing EM with ML gives MET' = MET + EM - ML.
        #        new_x += (em_pt - ml_pt) * np.cos(phi)
        #        new_y += (em_pt - ml_pt) * np.sin(phi)
        #        n_forward += 1

        new_pt = np.hypot(new_x, new_y)
        event_pz_ref = project_met_on_z_axis(lep_pt, lep_phi, met_x, met_y)
        event_pz_ml_forward = project_met_on_z_axis(lep_pt, lep_phi, new_x, new_y)
        met_ref.append(met_pt)
        met_ml_forward.append(new_pt)
        ptz.append(event_ptz)
        pz_ref.append(event_pz_ref)
        pz_ml_forward.append(event_pz_ml_forward)

        #print(
        #    f"event {event}: n_forward={n_forward}, "
        #    f"ref_met={met_pt:.6f}, ml_forward_met={new_pt:.6f}, diff={new_pt - met_pt:.6e}, "
        #    f"pTZ={event_ptz:.6f}, "
        #    f"pz_ref={event_pz_ref:.6f}, pz_ml_forward={event_pz_ml_forward:.6f}, "
        #    f"pz_diff={event_pz_ml_forward - event_pz_ref:.6e}"
        #)

    met_ref = np.asarray(met_ref)
    met_ml_forward = np.asarray(met_ml_forward)
    diff = met_ml_forward - met_ref
    ptz = np.asarray(ptz)
    pz_ref = np.asarray(pz_ref)
    pz_ml_forward = np.asarray(pz_ml_forward)
    pz_diff = pz_ml_forward - pz_ref

    print(f"events: {len(diff)}")
    if len(diff) == 0:
        return

    print(f"forward eta: {args.eta_min} <= |eta| < {args.eta_max}")
    print(f"mean reference MET: {np.mean(met_ref):.6f} GeV")
    print(f"mean ML-forward MET: {np.mean(met_ml_forward):.6f} GeV")
    print(f"mean change: {np.mean(diff):.6e} GeV")
    print(f"std change: {np.std(diff):.6e} GeV")
    print(f"max |change|: {np.max(np.abs(diff)):.6e} GeV")
    print(f"mean pz reference: {np.nanmean(pz_ref):.6f} GeV")
    print(f"mean pz ML-forward: {np.nanmean(pz_ml_forward):.6f} GeV")
    print(f"mean pz change: {np.nanmean(pz_diff):.6e} GeV")
    print(f"std pz change: {np.nanstd(pz_diff):.6e} GeV")
    print(f"max |pz change|: {np.nanmax(np.abs(pz_diff)):.6e} GeV")
    save_pz_histograms(
        args.output,
        ptz,
        pz_ref,
        pz_ml_forward,
        pz_diff,
        args.ptz_bins,
        args.pz_bins,
        args.pz_min,
        args.pz_max,
    )


if __name__ == "__main__":
    main()
