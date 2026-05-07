import os

import numpy as np
import ROOT


def calculate_gaussian_fit_params_from_arrays(x, y, x_bin_edges, out_file=None, y_name="array_obs"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    mean_values = []
    sigma_values = []

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
        y_low = y_mean - 3 * y_std
        y_high = y_mean + 3 * y_std

        hname = f"h_fit_input_{y_name}_bin{i:03d}"
        htitle = f"{y_name};{y_name};Events"
        htmp = ROOT.TH1D(hname, htitle, 100, y_low, y_high)
        htmp.SetDirectory(0)
        htmp.Sumw2()

        for val in y_bin:
            htmp.Fill(float(val))

        if y_mean <= 0 or not np.isfinite(y_std):
            mean_values.append(np.nan)
            sigma_values.append(np.nan)
            continue

        fit_name = f"fgaus_{y_name}_bin{i:03d}"
        fit = ROOT.TF1(fit_name, "gaus", y_mean - 2.0 * y_std, y_mean + 1.0 * y_std)
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

