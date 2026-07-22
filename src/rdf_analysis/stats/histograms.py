import numpy as np
import ROOT


def make_iqr_hist(hist_name, hist_title, x_bin_edges, sigma_iqr_values, x_title="p_{T}^{#mu} [GeV]", y_title="#sigma_{IQR}"):
    edges = np.asarray(x_bin_edges, dtype=np.float64)
    hist = ROOT.TH1D(hist_name, hist_title, len(edges) - 1, edges)

    for i, val in enumerate(sigma_iqr_values, start=1):
        hist.SetBinContent(i, float(val) if np.isfinite(val) else 0.0)

    hist.GetXaxis().SetTitle(x_title)
    hist.GetYaxis().SetTitle(y_title)
    return hist


def make_param_hist(hist_name, hist_title, x_bin_edges, values, y_title):
    edges = np.asarray(x_bin_edges, dtype=np.float64)
    hist = ROOT.TH1D(hist_name, hist_title, len(edges) - 1, edges)

    for i, val in enumerate(values, start=1):
        hist.SetBinContent(i, float(val) if np.isfinite(val) else 0.0)

    hist.GetXaxis().SetTitle("p_{T}^{#mu} [GeV]")
    hist.GetYaxis().SetTitle(y_title)
    return hist


def make_numpy_hist(hist_name, hist_title, *args):
    """
    Backward-compatible helper for both signatures:
      make_numpy_hist(name, title, n_bins, x_min, x_max, values)
      make_numpy_hist(name, title, bin_edges, values)
    """
    if len(args) == 4:
        n_bins, x_min, x_max, values = args
        hist = ROOT.TH1D(hist_name, hist_title, n_bins, x_min, x_max)
    elif len(args) == 2:
        bin_edges, values = args
        edges = np.asarray(bin_edges, dtype=np.float64)
        hist = ROOT.TH1D(hist_name, hist_title, len(edges) - 1, edges)
    else:
        raise TypeError("make_numpy_hist expects either 4 or 2 positional arguments after hist_title")

    for val in values:
        hist.Fill(val)

    hist.GetXaxis().SetTitle(hist_title)
    hist.GetYaxis().SetTitle("Events")
    return hist


def make_hist_from_bin_contents(hist_name, hist_title, bin_edges, bin_contents):
    edges = np.asarray(bin_edges, dtype=np.float64)
    hist = ROOT.TH1D(hist_name, hist_title, len(edges) - 1, edges)

    for bin_idx, value in enumerate(bin_contents, start=1):
        hist.SetBinContent(bin_idx, float(value) if np.isfinite(value) else 0.0)

    return hist

# get linear binning
def get_bins(x_min, x_max, n_bins):
    return np.linspace(x_min, x_max, n_bins + 1)

def get_bins_log(x_min, x_max, nbins):
    return np.array(np.geomspace(x_min, x_max, nbins + 1))

# make 2d histogram
def make_numpy_hists_2d(hist_name, hist_title, x_bin_edges, x_values, y_bin_edges, y_values):
    x_edges = np.asarray(x_bin_edges, dtype=np.float64)
    y_edges = np.asarray(y_bin_edges, dtype=np.float64)
    hist = ROOT.TH2D(hist_name, hist_title, len(x_edges) - 1, x_edges, len(y_edges) - 1, y_edges)

    for x_val, y_val in zip(x_values, y_values):
        hist.Fill(x_val, y_val)

    hist.GetXaxis().SetTitle("x")
    hist.GetYaxis().SetTitle("y")
    return hist
