#!/usr/bin/env python3
"""
Validate the uT (hadronic recoil transverse momentum) calculation from Python
against the already present "u_pt_em" field in the data.

This script computes uT using hadronic_recoil_from_clusters_python and compares
it with the stored u_pt_em values to ensure consistency.
"""

import argparse
import numpy as np
from pathlib import Path
import ROOT
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdf_analysis.analyses.hadrecoil_common import NtupleProcessorRDF_hadrecoil
from rdf_analysis.physics import hadronic_recoil_from_clusters_python
from rdf_analysis.stats import make_numpy_hist, make_hist_from_bin_contents


def validate_em_recoil_calculation(proc, max_events=None):
    """
    Validate EM recoil calculation by comparing Python computed u_pt_em
    against the stored u_pt_em branch.

    Returns dict with validation results.
    """
    # Extract required data
    df_lep_pt = proc.to_pandas(["lep_pt"])
    df_lep_eta = proc.to_pandas(["lep_eta"])
    df_lep_phi = proc.to_pandas(["lep_phi"])
    df_clus_e_em = proc.to_pandas(["clus_e_em"])
    df_clus_eta = proc.to_pandas(["clus_eta"])
    df_clus_phi = proc.to_pandas(["clus_phi"])

    # Get the stored u_pt_em values from the processor
    df_u_pt_em_stored = proc.to_pandas(["u_pt_em"])
    u_pt_em_stored = df_u_pt_em_stored["u_pt_em"].to_numpy()

    total_events = len(u_pt_em_stored)
    if max_events is not None and max_events > 0:
        total_events = min(max_events, total_events)

    # Arrays to store results
    u_pt_em_computed = []
    diff_em = []
    rel_diff_em = []
    event_list = []

    print(f"[Info]:: Validating EM recoil calculation for {total_events} events")

    for event in range(total_events):
        if event % 10000 == 0 and event > 0:
            print(f"[Info]:: Validated {event}/{total_events} events")

        # Get event data
        lep_pt = np.array(df_lep_pt.iloc[event]["lep_pt"])
        lep_eta = np.array(df_lep_eta.iloc[event]["lep_eta"])
        lep_phi = np.array(df_lep_phi.iloc[event]["lep_phi"])

        clus_e = np.array(df_clus_e_em.iloc[event]["clus_e_em"])
        clus_eta = np.array(df_clus_eta.iloc[event]["clus_eta"])
        clus_phi = np.array(df_clus_phi.iloc[event]["clus_phi"])

        # Compute EM recoil using Python function
        hadrecoil = hadronic_recoil_from_clusters_python(
            lep_pt,
            lep_eta,
            lep_phi,
            clus_e,
            clus_eta,
            clus_phi,
        )

        u_pt_computed = hadrecoil["u_pt"]
        u_pt_stored = u_pt_em_stored[event]

        u_pt_em_computed.append(u_pt_computed)
        diff = u_pt_computed - u_pt_stored
        diff_em.append(diff)

        # Relative difference (avoid division by zero)
        if u_pt_stored != 0:
            rel_diff = diff / u_pt_stored
        else:
            rel_diff = 0.0 if diff == 0.0 else np.inf
        rel_diff_em.append(rel_diff)

        event_list.append(event)

    u_pt_em_computed = np.array(u_pt_em_computed)
    diff_em = np.array(diff_em)
    rel_diff_em = np.array(rel_diff_em)

    # Calculate statistics
    mean_diff = np.mean(diff_em)
    std_diff = np.std(diff_em)
    max_diff = np.max(np.abs(diff_em))
    mean_rel_diff = np.mean(rel_diff_em)

    # Find events with significant differences
    threshold_rel_diff = 0.01  # 1% relative difference
    outlier_events = np.where(np.abs(rel_diff_em) > threshold_rel_diff)[0]

    results = {
        "u_pt_em_computed": u_pt_em_computed,
        "u_pt_em_stored": u_pt_em_stored[:total_events],
        "diff": diff_em,
        "rel_diff": rel_diff_em,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "max_diff": max_diff,
        "mean_rel_diff": mean_rel_diff,
        "outlier_events": outlier_events,
        "n_outliers": len(outlier_events),
    }

    #print(f"diff {diff_em} GeV")
    #print(f"computed {u_pt_em_computed}, \n stored {u_pt_em_stored}")

    return results


def print_validation_summary(results):
    """Print validation summary to console."""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY: EM Recoil Calculation")
    print("="*70)
    print(f"Total events validated: {len(results['u_pt_em_stored'])}")
    print(f"Mean difference (computed - stored): {results['mean_diff']:.6e} GeV")
    print(f"Std dev of difference: {results['std_diff']:.6e} GeV")
    print(f"Max absolute difference: {results['max_diff']:.6e} GeV")
    print(f"Mean relative difference: {results['mean_rel_diff']:.6e}")
    print(f"Number of outliers (|rel_diff| > 1%): {results['n_outliers']}")

    if results['n_outliers'] > 0:
        print(f"\nOutlier events (first 10):")
        for i, event_idx in enumerate(results['outlier_events'][:10]):
            rel_diff = results['rel_diff'][event_idx]
            diff = results['diff'][event_idx]
            stored = results['u_pt_em_stored'][event_idx]
            computed = results['u_pt_em_computed'][event_idx]
            print(f"  Event {event_idx}: computed={computed:.4f}, stored={stored:.4f}, "
                  f"diff={diff:.4e}, rel_diff={rel_diff:.4e}")

    # Check if validation passes (allowing small floating-point differences)
    validation_passed = (results['max_diff'] < 1e-6 and results['mean_rel_diff'] < 1e-10)
    status = "PASSED" if validation_passed else "WARNING/FAILED"
    print(f"\nValidation Status: {status}")
    print("="*70 + "\n")

    return validation_passed


def save_validation_plots(results, output_file):
    """Save validation plots to ROOT file."""
    out_file = ROOT.TFile.Open(output_file, "RECREATE")

    # Plot 1: Difference distribution
    make_numpy_hist(
        "h_diff_u_pt_em",
        "Difference (computed - stored) for u_pt_em; GeV; Events",
        100, -0.5, 0.5,
        results["diff"]
    ).Write()

    # Plot 2: Relative difference distribution
    # Filter out inf/nan values for relative difference
    rel_diff_finite = results["rel_diff"][np.isfinite(results["rel_diff"])]
    make_numpy_hist(
        "h_rel_diff_u_pt_em",
        "Relative difference (computed - stored)/stored for u_pt_em; Relative diff; Events",
        100, -0.01, 0.01,
        rel_diff_finite
    ).Write()

    # Plot 3: Scatter plot stored vs computed
    # Create a 2D histogram
    h2d = ROOT.TH2F(
        "h_scatter_u_pt_em",
        "EM recoil: Computed vs Stored; u_pt_em (stored) [GeV]; u_pt_em (computed) [GeV]",
        50, 0, 100,
        50, 0, 100
    )
    for stored, computed in zip(results["u_pt_em_stored"], results["u_pt_em_computed"]):
        h2d.Fill(stored, computed)
    h2d.Write()

    # Plot 4: Residuals vs stored value
    h_residuals = ROOT.TH2F(
        "h_residuals_u_pt_em",
        "Residuals vs stored value; u_pt_em (stored) [GeV]; Difference [GeV]",
        50, 0, 100,
        100, -0.5, 0.5
    )
    for stored, diff in zip(results["u_pt_em_stored"], results["diff"]):
        h_residuals.Fill(stored, diff)
    h_residuals.Write()

    out_file.Close()
    print(f"[Info]:: Validation plots saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the u_pt_em recoil calculation.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="validation_plots.root", help="Output ROOT file with validation plots.")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to validate.")
    return parser.parse_args()


def main(args):
    """Main validation routine."""
    # Initialize processor
    proc = NtupleProcessorRDF_hadrecoil(args.input, args.tree, args.nEvents)
    proc.build_dataframe()

    # Run validation
    results = validate_em_recoil_calculation(proc, max_events=args.nEvents)

    # Print summary
    validation_passed = print_validation_summary(results)

    # Save plots
    save_validation_plots(results, args.output)

    return results, validation_passed


if __name__ == "__main__":
    args = parse_args()
    results, validation_passed = main(args)
    sys.exit(0 if validation_passed else 1)

