import os

import numpy as np
import ROOT
from ..config import config_fits

def calculate_mean_sigma_in_x_bins(x_variable, y_variable, x_bin_edges):
    x = np.asarray(x_variable, dtype=np.float64)
    y = np.asarray(y_variable, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    mean_values = []
    sigma_values = []

    for i in range(len(x_bin_edges) - 1):
        x_low = x_bin_edges[i]
        x_high = x_bin_edges[i + 1]
        in_bin = (x >= x_low) & (x < x_high)

        if not np.any(in_bin):
            mean_values.append(np.nan)
            sigma_values.append(np.nan)
            continue

        y_bin = y[in_bin]
        mean_values.append(float(np.mean(y_bin)))
        sigma_values.append(float(np.std(y_bin)))

    return np.array(mean_values), np.array(sigma_values)

def calculate_gaussian_fit_params_from_arrays(x, y, x_bin_edges, out_file=None, y_name="array_obs"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    mean_values = []
    sigma_values = []

    # Load fit settings from config (values expressed in numbers of sigma)
    fit_hist_bins = int(config_fits.get("fit_hist_bins", 100))
    fit_hist_low_sigma = float(config_fits.get("fit_hist_low", -3))
    fit_hist_high_sigma = float(config_fits.get("fit_hist_high", 3))
    fit_range_low_sigma = float(config_fits.get("fit_range_low", -2))
    fit_range_high_sigma = float(config_fits.get("fit_range_high", 1))

    out_dir = None
    out_dir_fs = f"outputfiles/fit_results/{y_name}"
    if out_file is not None:
        out_dir = out_file.GetDirectory(f"fit_results/{y_name}")
        os.makedirs(out_dir_fs, exist_ok=True)
        if not out_dir:
            out_dir = out_file.mkdir(f"fit_results/{y_name}")

    for i in range(len(x_bin_edges) - 1):
        x_low = x_bin_edges[i]
        x_high = x_bin_edges[i + 1]
        in_bin = (x >= x_low) & (x < x_high)

        if not np.any(in_bin):
            mean_values.append(np.nan)
            sigma_values.append(np.nan)
            continue

        y_bin = y[in_bin]
        if y_bin.size < 10:
            mean_values.append(np.nan)
            sigma_values.append(np.nan)
            continue

        y_mean = float(np.mean(y_bin))
        y_std = float(np.std(y_bin))

        # Histogram range and fit range are specified in config as multiples of sigma
        y_low = y_mean + fit_hist_low_sigma * y_std
        y_high = y_mean + fit_hist_high_sigma * y_std

        # Protect against invalid ranges (e.g. zero or NaN std)
        if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low >= y_high:
            mean_values.append(np.nan)
            sigma_values.append(np.nan)
            continue

        hname = f"h_fit_input_{y_name}_bin{i:03d}"
        htitle = f"{y_name};{y_name};Events"
        htmp = ROOT.TH1D(hname, htitle, fit_hist_bins, y_low, y_high)
        htmp.SetDirectory(0)
        htmp.Sumw2()

        for val in y_bin:
            htmp.Fill(float(val))

        if  not np.isfinite(y_std):
            mean_values.append(np.nan)
            sigma_values.append(np.nan)
            continue

        # Build fit function using config-provided sigma multipliers
        fit_low = y_mean + fit_range_low_sigma * y_std
        fit_high = y_mean + fit_range_high_sigma * y_std
        if not np.isfinite(fit_low) or not np.isfinite(fit_high) or fit_low >= fit_high:
            mean_values.append(np.nan)
            sigma_values.append(np.nan)
            continue

        fit_name = f"fgaus_{y_name}_bin{i:03d}"
        fit = ROOT.TF1(fit_name, "gaus", fit_low, fit_high)
        fit.SetParameters(htmp.GetMaximum(), y_mean, y_std)

        fit_result = htmp.Fit(fit, "QNR")
        if int(fit_result) != 0:
            mean_values.append(np.nan)
            sigma_values.append(np.nan)
        else:
            mean_values.append(float(fit.GetParameter(1)))
            sigma_values.append(abs(float(fit.GetParameter(2))))

        if out_dir is not None:
            prev_dir = ROOT.gDirectory
            out_dir.cd()

            cname = f"c_fit_overlay_{y_name}_bin{i:03d}"
            ctitle = f"{y_name} fit overlay bin {i} [{x_low:.1f}, {x_high:.1f})"
            c = ROOT.TCanvas(cname, ctitle, 800, 600)

            htmp.SetLineColor(ROOT.kBlack)
            htmp.SetMarkerStyle(20)
            htmp.SetMarkerSize(0.7)
            htmp.Draw("E")

            fit.SetLineColor(ROOT.kRed + 1)
            fit.SetLineWidth(2)
            fit.Draw("SAME")

            leg = ROOT.TLegend(0.60, 0.75, 0.88, 0.88)
            leg.AddEntry(htmp, "Data", "lep")
            leg.AddEntry(fit, "Gaussian fit", "l")
            leg.Draw()

            c.Write("", ROOT.TObject.kOverwrite)
            c.SaveAs(os.path.join(out_dir_fs, f"{cname}.png"))
            if prev_dir:
                prev_dir.cd()

    return np.array(mean_values), np.array(sigma_values)


def calculate_sigma_iqr_in_x_bins(x_variable, y_variable, x_bin_edges, min_entries=10):
    x = np.asarray(x_variable, dtype=np.float64)
    y = np.asarray(y_variable, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    median_values = []
    sigma_iqr68_values = []

    for i in range(len(x_bin_edges) - 1):
        x_low = x_bin_edges[i]
        x_high = x_bin_edges[i + 1]

        if i == len(x_bin_edges) - 2:
            in_bin = (x >= x_low) & (x <= x_high)
        else:
            in_bin = (x >= x_low) & (x < x_high)

        if np.count_nonzero(in_bin) < min_entries:
            median_values.append(np.nan)
            sigma_iqr68_values.append(np.nan)
            continue

        y_bin = y[in_bin]

        q16, q50, q84 = np.quantile(y_bin, [0.16, 0.50, 0.84])

        median_values.append(float(q50))
        sigma_iqr68_values.append(float((q84 - q16) / 2.0))

    return np.array(median_values), np.array(sigma_iqr68_values)
