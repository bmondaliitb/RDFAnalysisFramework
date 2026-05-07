import numpy as np
import ROOT
from hadrecoil import *

import os

# Get name of the object
def get_name(obj):
    for name, val in locals().items():
        if val is obj:
            return name

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
        # also save the fits as png to a folder
        os.makedirs(out_dir_fs, exist_ok=True)
        if not out_dir:
            out_dir = out_file.mkdir(f"fit_results/{y_name}")

    for i in range(len(x_bin_edges) - 1):
        x_low = x_bin_edges[i]
        x_high = x_bin_edges[i + 1]
        in_bin = (x >= x_low) & (x < x_high) # ToDo: support using of x-variable. this doesn't support masking with x-variable

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
        y_low = y_mean - 3*y_std
        y_high = y_mean + 3*y_std


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
        fit = ROOT.TF1(fit_name, "gaus", y_mean - 2.0 * y_std, y_mean + 1.0 * y_std) # fit till 2sigma
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
            #htmp.Write("", ROOT.TObject.kOverwrite) # not writing the histogram and fit function
            #fit.Write("", ROOT.TObject.kOverwrite)

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


def make_iqr_hist(hist_name, hist_title, x_bin_edges, sigma_iqr_values):
    """
    Create TH1D with user-defined bin edges and fill bin content with sigma_iqr.
    """
    edges = np.asarray(x_bin_edges, dtype=np.float64)
    hist = ROOT.TH1D(hist_name, hist_title, len(edges) - 1, edges)

    for i, val in enumerate(sigma_iqr_values, start=1):
        if np.isfinite(val):
            hist.SetBinContent(i, float(val))
        else:
            hist.SetBinContent(i, 0.0)

    hist.GetXaxis().SetTitle("p_{T}^{#mu} [GeV]")
    hist.GetYaxis().SetTitle("#sigma_{IQR}")
    return hist

def make_param_hist(hist_name, hist_title, x_bin_edges, values, y_title):
    edges = np.asarray(x_bin_edges, dtype=np.float64)
    hist = ROOT.TH1D(hist_name, hist_title, len(edges) - 1, edges)

    for i, val in enumerate(values, start=1):
        hist.SetBinContent(i, float(val) if np.isfinite(val) else 0.0)

    hist.GetXaxis().SetTitle("p_{T}^{#mu} [GeV]")
    hist.GetYaxis().SetTitle(y_title)
    return hist

# make root th1d histogram from numpy column
def make_numpy_hist(hist_name, hist_title, n_bins, min, max, values):
    """
    Create a ROOT TH1D with uniform bins and fill it from raw values.
    """
    hist = ROOT.TH1D(hist_name, hist_title, n_bins, min, max)
    for val in values:
        hist.Fill(val)
    hist.GetXaxis().SetTitle(hist_title)
    hist.GetYaxis().SetTitle("Events")
    return hist


def make_numpy_hist(hist_name, hist_title, bin_edges, values):
    """
    Create a ROOT TH1D with variable-width bins and fill it from raw values.
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    hist = ROOT.TH1D(hist_name, hist_title, len(edges) - 1, edges)
    for val in values:
        hist.Fill(val)
    hist.GetXaxis().SetTitle(hist_title)
    hist.GetYaxis().SetTitle("Events")
    return hist

def make_hist_from_bin_contents(hist_name, hist_title, bin_edges, bin_contents):
    """
    Create a ROOT TH1D with custom bin edges and explicit bin contents.
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    hist = ROOT.TH1D(hist_name, hist_title, len(edges) - 1, edges)

    for bin_idx, value in enumerate(bin_contents, start=1):
        hist.SetBinContent(
            bin_idx,
            float(value) if np.isfinite(value) else 0.0
        )
    hist.GetXaxis().SetTitle(hist_title)
    hist.GetYaxis().SetTitle("Events")
    return hist


def build_threshold_scan_observables(
    lep_pt, lep_eta, lep_phi,
    clus_e, clus_eta, clus_phi,
    thresholds
):
    """
    Rebuild hadronic recoil from clusters for several cluster-energy thresholds.
    Returns a dict:
      {
        thr_value: {
          "u_x": ...,
          "u_y": ...,
          "u_mag": ...
        },
        ...
      }
    }
    """
    out = {}

    for thr in thresholds:
        mask = np.asarray(clus_e) > thr

        clus_e_sel = np.asarray(clus_e)[mask]
        clus_eta_sel = np.asarray(clus_eta)[mask]
        clus_phi_sel = np.asarray(clus_phi)[mask]

        hadrecoil = hadronic_recoil_from_clusters_python(
            np.asarray(lep_pt),
            np.asarray(lep_eta),
            np.asarray(lep_phi),
            clus_e_sel,
            clus_eta_sel,
            clus_phi_sel
        )

        # adapt these keys if your hadrecoil function returns different names
        out[thr] = {
            "u_x": hadrecoil["u_x"],
            "u_y": hadrecoil["u_y"],
            "u_mag": hadrecoil["u_pt"],
        }

    return out


def get_bins_log(min, max, nbins):
    # Log-spaced edges
    edges = np.geomspace(min, max, nbins + 1)
    return np.array(edges)
