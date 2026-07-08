#!/bin/env python

import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    yaml = None

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

try:
    from model import GaussianMixtureDNN
except ImportError:
    GaussianMixtureDNN = None

try:
    from config import base_dir
except ImportError:
    base_dir = os.getcwd()

try:
    from utils_plotting import add_textbox_to_ax
except ImportError:
    def add_textbox_to_ax(ax):
        return

try:
    from dataset import *
    from dataset import filter_clusters_by_eta as filter_eta_window
    from dataset import format_eta_selection_label
except ImportError:
    filter_eta_window = None
    format_eta_selection_label = None

do_lowmu = False
do_highmu = True 

if do_lowmu:
  ENERGY_MIN=0.01
  ENERGY_MAX=100
  
  RESPONSE_MIN=0.01
  RESPONSE_MAX=100
  NBINS=50
  NBINS_IQR = 15


if do_highmu:
  ENERGY_MIN=0.1
  ENERGY_MAX=2000
  
  RESPONSE_MIN=0.1
  RESPONSE_MAX=10000

  NBINS=50
  NBINS_IQR = 15


FEATURE_INDEX = {
    "r_ems": 0,
    "e_ems": 1,
    "y_ems": 2,
    "m_sig": 3,
    "t_ems": 4,
    "m_tim": 5,
    "m_lam": 6,
    "m_cog": 7,
    "m_fem": 8,
    "m_rho": 9,
    "m_lon": 10,
    "m_lat": 11,
    "m_ptd": 12,
    "m_iso": 13,
    "p_npv": 14,
    "p_mmu": 15,
    "cluster_closestDeltaR": 16,
    "cluster_FIRST_TOWER_PHI": 17,
    "cluster_FIRST_TOWER_ETA": 18,
    "cluster_FIRST_TOWER_R": 19,
    "cluster_SECOND_TOWER_PHI": 20,
    "cluster_SECOND_TOWER_ETA": 21,
    "cluster_SECOND_TOWER_R": 22,
    "cluster_TOWER_PTD": 23,
    "e_dep": 24,
    "e_hgm": 25,
    "e_bnn": 26,
    "r_hgm": 27,
    "r_bnn": 28,
    "c_hgm": 29,
    "c_bnn": 30,
    "l_ems": 31,
    "l_had": 32,
    "l_hgm": 33,
    "l_bnn": 34,
}


def apply_save_log(x):
    epsilon = 1e-10
    minimum = np.min(x)
    if minimum <= 0:
        x = x - minimum + epsilon
    else:
        minimum = 0
        epsilon = 0
    return np.log10(x), minimum, epsilon


def resolve_feature_flags(config):
    feature_keys = list(config["data"]["features"].keys())
    has_tower = any(
        key.startswith("cluster_FIRST_TOWER_")
        or key.startswith("cluster_SECOND_TOWER_")
        or key == "cluster_TOWER_PTD"
        for key in feature_keys
    )
    if has_tower:
        return True, False

    if "cluster_closestDeltaR" in feature_keys:
        return False, False

    return False, True


def load_batches(input_path, batch_idxs, mc_release):
    if batch_idxs:
        if mc_release == "mc16":
            filenames = [f"clustercalib_bnn_500k_batch{i:03d}.npy" for i in batch_idxs]
        else:
            filenames = [f"clustercalib_nn_500k_batch{i:03d}.npy" for i in batch_idxs]
        file_paths = [os.path.join(input_path, name) for name in filenames]
    else:
        file_paths = sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.endswith(".npy")
        )

    if not file_paths:
        raise FileNotFoundError("No .npy files found for inference input.")

    data = [np.load(path) for path in file_paths]
    return np.concatenate(data, axis=0)


def apply_preprocessing(data_before, config, scales, use_tower_moments, default_setup):
    data_after = data_before.copy()
    applied_transforms = []

    def record_transform(feature_key, trained_name, transform):
        applied_transforms.append((feature_key, trained_name, transform))

    # log10 preprocessing for target
    x = np.log10(data_after[:, FEATURE_INDEX["r_ems"]])
    print("[Info] r_ems log_10 transformed \n")
    data_after[:, FEATURE_INDEX["r_ems"]] = x
    record_transform("target", "r_ems", "log10")

    if use_tower_moments:
        features = [
            "nPrimVtx",
            "avgMu",
            "cluster_FIRST_TOWER_PHI",
            "cluster_FIRST_TOWER_ETA",
            "cluster_FIRST_TOWER_R",
            "cluster_SECOND_TOWER_PHI",
            "cluster_SECOND_TOWER_ETA",
            "cluster_SECOND_TOWER_R",
            "cluster_TOWER_PTD",
        ]
    elif default_setup:
        features = [
            "clusterEta",
            "cluster_CENTER_MAG",
            "nPrimVtx",
            "avgMu",
            "cluster_LATERAL",
            "cluster_LONGITUDINAL",
            "cluster_PTD",
            "cluster_ISOLATION",
        ]
    else:
        features = [
            "clusterEta",
            "cluster_CENTER_MAG",
            "nPrimVtx",
            "avgMu",
            "cluster_LATERAL",
            "cluster_LONGITUDINAL",
            "cluster_PTD",
            "cluster_ISOLATION",
            "cluster_closestDeltaR",
        ]

    for feature in features:
        trained_name = config["data"]["features"][feature]["trained_name"]
        if trained_name in scales:
            mean, std = scales[trained_name][-2], scales[trained_name][-1]
            idx = FEATURE_INDEX[trained_name]
            data_after[:, idx] = (data_after[:, idx] - mean) / std
            print(f"[Info] {feature}, ")
            record_transform(feature, trained_name, "zscore")

    if use_tower_moments:
        features = ["clusterE"]
    else:
        features = [
            "clusterE",
            "cluster_CENTER_LAMBDA",
            "cluster_FIRST_ENG_DENS",
            "cluster_SECOND_TIME",
            "cluster_SIGNIFICANCE",
        ]

    for feature in features:
        trained_name = config["data"]["features"][feature]["trained_name"]
        if trained_name in scales:
            mean, std = scales[trained_name][-2], scales[trained_name][-1]
            idx = FEATURE_INDEX[trained_name]
            x, _, _ = apply_save_log(data_after[:, idx])
            data_after[:, idx] = (x - mean) / std
            record_transform(feature, trained_name, "log10 + zscore")

    if not use_tower_moments:
        feature = "cluster_time"
        trained_name = config["data"]["features"][feature]["trained_name"]
        if trained_name in scales:
            xmax = scales[trained_name][-1]
            idx = FEATURE_INDEX[trained_name]
            data_after[:, idx] = data_after[:, idx] / xmax
            record_transform(feature, trained_name, "maxabs")

        feature = "cluster_ENG_FRAC_EM_INCL"
        if feature in config["data"]["features"]:
            trained_name = config["data"]["features"][feature]["trained_name"]
            if trained_name in scales:
                xmin, xmax = scales[trained_name][-2], scales[trained_name][-1]
                idx = FEATURE_INDEX[trained_name]
                data_after[:, idx] = (data_after[:, idx] - xmin) / (xmax - xmin)
                record_transform(feature, trained_name, "min-max")

    return data_after, applied_transforms


def print_preprocessing_summary(applied_transforms):
    print("[info] Preprocessing transformations:")
    for feature_name, trained_name, transform in applied_transforms:
        print(f"[info]   {feature_name} ({trained_name}): {transform}")


def select_columns(data, use_tower_moments, default_setup):
    if use_tower_moments:
        cols_to_keep = [0, 1, 14, 15] + list(range(17, 24))
    elif default_setup:
        cols_to_keep = list(range(0, 16))
    else:
        cols_to_keep = list(range(0, 17))

    max_idx = max(cols_to_keep)
    if max_idx >= data.shape[1]:
        raise ValueError("Input data does not contain all required columns.")

    return data[:, cols_to_keep]


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def compute_true_energy(data_before, r_truth):
    if data_before.shape[1] > FEATURE_INDEX["e_dep"]:
        e_dep = data_before[:, FEATURE_INDEX["e_dep"]]
        if np.all(np.isfinite(e_dep)):
            return e_dep
    e_ems = data_before[:, FEATURE_INDEX["e_ems"]]
    return e_ems / np.clip(r_truth, 1e-12, None)


def compute_pred_energy(data_before, r_model):
    e_ems = data_before[:, FEATURE_INDEX["e_ems"]]
    return e_ems / np.clip(r_model, 1e-12, None)


def plot_energy_scatter(output_path, e_truth, e_model):
    """Energy scatter plot: deposited energy vs HGM-predicted energy."""
    figsize = (2.953, 3.2221)
    rect_single = (0.13, 0.12, 0.97, 0.98)
    subs = (0.2, 0.4, 0.6, 0.8, 1.0)
    
    plt.style.use(f'{base_dir}/Isabel-code/nn-topocluster-calibration/plotting-routine/plotting.mplstyle')
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_single)
    
    # Create 2D histogram
    bins_logx = np.logspace(np.log10(ENERGY_MIN), np.log10(ENERGY_MAX), NBINS)
    bins_logy = np.logspace(np.log10(ENERGY_MIN), np.log10(ENERGY_MAX), NBINS)
    ax.hist2d(e_truth, e_model, bins=[bins_logx, bins_logy], norm=mpl.colors.LogNorm())
    
    # Diagonal line
    ax.plot([ENERGY_MIN, ENERGY_MAX], [ENERGY_MIN, ENERGY_MAX], linestyle='dashed', color='black')
    
    ax.set_xlabel('$E_{\\mathrm{truth}} [\\mathrm{GeV}$]')
    ax.set_ylabel('$E_{\\mathrm{HGM}} [\\mathrm{GeV}$]')
    ax.set_xlim((bins_logx[0], bins_logx[-1]))
    ax.set_ylim((bins_logy[0], bins_logy[-1]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    add_textbox_to_ax(ax)
    fig.savefig(os.path.join(output_path, 'scatter_energy_truth_vs_hgm.png'), dpi=200)
    fig.savefig(os.path.join(output_path, 'scatter_energy_truth_vs_hgm.pdf'), dpi=200)
    plt.close(fig)


def plot_response_scatter(output_path, r_truth, r_model):
    """Response scatter plot: target response vs predicted response."""
    figsize = (2.953, 3.2221)
    rect_single = (0.13, 0.12, 0.97, 0.98)
    subs = (0.2, 0.4, 0.6, 0.8, 1.0)
    
    plt.style.use(f'{base_dir}/Isabel-code/nn-topocluster-calibration/plotting-routine/plotting.mplstyle')
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_single)
    
    # Create 2D histogram
    bins_logx = np.logspace(np.log10(RESPONSE_MIN), np.log10(RESPONSE_MAX), NBINS)
    bins_logy = np.logspace(np.log10(RESPONSE_MIN), np.log10(RESPONSE_MAX), NBINS)
    ax.hist2d(r_truth, r_model, bins=[bins_logx, bins_logy], norm=mpl.colors.LogNorm())
    
    # Diagonal line
    ax.plot([RESPONSE_MIN, RESPONSE_MAX], [RESPONSE_MIN, RESPONSE_MAX], linestyle='dashed', color='black')
    
    ax.set_xlabel('Target response [$\\mathcal{R}_\\mathrm{clus}^\\mathrm{EM}$]')
    ax.set_ylabel('HGM-predicted response [$\\mathcal{R}_\\mathrm{clus}^\\mathrm{HGM}$]')
    ax.set_xlim((bins_logx[0], bins_logx[-1]))
    ax.set_ylim((bins_logy[0], bins_logy[-1]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    add_textbox_to_ax(ax)
    fig.savefig(os.path.join(output_path, 'scatter_response_truth_vs_hgm.png'), dpi=200)
    fig.savefig(os.path.join(output_path, 'scatter_response_truth_vs_hgm.pdf'), dpi=200)
    plt.close(fig)


def plot_iqr_vs_energy(output_path, e_x, r_truth, r_model, x_label, out_name):
    """Relative response resolution vs energy using normalized IQR definition.
    
    Computes sigma_rel = Q_68% / (2 * median) where Q_68% is the 68% 
    inter-quantile range of the response ratio r_model/r_truth.
    """
    plt.style.use(f'{base_dir}/Isabel-code/nn-topocluster-calibration/plotting-routine/plotting.mplstyle')

    fig, (ax, ax_ratio) = plt.subplots(
        2, 1,
        figsize=(2.953 * 1.6, 3.2221 * 1.3),
        gridspec_kw={'height_ratios': [4, 1.5], 'hspace': 0.05}
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.12,
        top=0.98
    )
    # Compute response ratio
    mask = np.isfinite(e_x) & (e_x > 0)
    e_x_filtered = e_x[mask]

    if e_x_filtered.size == 0:
        plt.close(fig)
        return
    
    # Create energy bins and compute relative resolution and response
    e_min = ENERGY_MIN#np.nanmin(e_truth_filtered)
    e_max = ENERGY_MAX#np.nanmax(e_truth_filtered)
    energy_bins = np.logspace(np.log10(e_min), np.log10(e_max), NBINS_IQR)
    
    bin_centers = []
    sigma_model = []
    sigma_em = []

    for i in range(len(energy_bins) - 1):
        lo, hi = energy_bins[i], energy_bins[i + 1]
        if i == len(energy_bins) - 2:
            in_bin = (e_x_filtered >= lo) & (e_x_filtered <= hi)
        else:
            in_bin = (e_x_filtered >= lo) & (e_x_filtered < hi)

        if np.count_nonzero(in_bin) < 10:
            continue
        
        bin_centers.append(np.sqrt(lo * hi))
        r_model_bin = r_model[mask][in_bin]
        r_truth_bin = r_truth[mask][in_bin]
        
        # Compute 68% IQR (16th to 84th percentile)
        q84_model = np.nanpercentile(r_model_bin, 84.0)
        q16_model = np.nanpercentile(r_model_bin, 16.0)
        iqr_model_68 = q84_model - q16_model

        q84_em = np.nanpercentile(r_truth_bin, 84.0)
        q16_em = np.nanpercentile(r_truth_bin, 16.0)
        iqr_em_68 = q84_em - q16_em
        # Compute median
        median_model = np.nanmedian(r_model_bin)
        median_em = np.nanmedian(r_truth_bin)

        # Relative resolution: Q_68% / (2 * median)
        sigma_iqr_model = iqr_model_68 / (2.0 * median_model) if median_model > 0 else np.nan
        sigma_model.append(sigma_iqr_model)
        sigma_iqr_em = iqr_em_68 / (2.0 * median_em) if median_em > 0 else np.nan
        sigma_em.append(sigma_iqr_em)

    if len(bin_centers) == 0:
        plt.close(fig)
        return
    
    bin_centers = np.array(bin_centers)
    sigma_model = np.array(sigma_model)
    sigma_em = np.array(sigma_em)

    # Create dual-axis plot
    ax1 = ax
    #ax2 = ax1.twinx()
    
    # Plot relative resolution on primary axis
    line1 = ax1.plot(bin_centers, sigma_model, color=color_model, marker='o', markersize=4,
                     linewidth=1.5, label='HGM')
    
    # Plot true EM response on secondary axis
    line2 = ax1.plot(bin_centers, sigma_em, color=color_truth, marker='s', markersize=4,
                     linewidth=1.5, label='EM')
    
    #ax1.set_xlabel('$E^{clus}_{dep}[\\mathrm{GeV}]$')
    ax1.set_ylabel('Relative resolution $\\sigma$', color=color_model)
    #ax2.set_ylabel('True EM response $\\mathcal{R}^{\\mathrm{truth}}$', color=color_truth)
    
    ax1.tick_params(axis='y', labelcolor=color_model)
    #ax2.tick_params(axis='y', labelcolor=color_truth)
    
    ax1.set_xscale('log')
    ax1.set_xlim((e_min, e_max))
    ax1.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax1.set_ylim(0, 2)

    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=8)

    ax1.tick_params(labelbottom=False)  # hide x labels on top

    ratio = sigma_model / sigma_em

    ax_ratio.plot(bin_centers, ratio,
                  color='black', marker='o', markersize=3,
                  linewidth=1.2)

    ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=1)

    ax_ratio.set_xlabel(x_label)
    ax_ratio.set_ylabel('Ratio (HGM/EM)')

    ax_ratio.set_xscale('log')
    ax_ratio.set_xlim((e_min, e_max))

    ax_ratio.set_ylim(0, 2.0)  # adjust as needed

    add_textbox_to_ax(ax1)
    fig.savefig(os.path.join(output_path, f'{out_name}.png'), dpi=200)
    fig.savefig(os.path.join(output_path, f'{out_name}.pdf'), dpi=200)
    plt.close(fig)

def plot_iqr_vs_energy_eta(output_path, e_x, r_truth, r_model, x_label, out_name):
    """Relative response resolution vs energy using normalized IQR definition.
    
    Computes sigma_rel = Q_68% / (2 * median) where Q_68% is the 68% 
    inter-quantile range of the response ratio r_model/r_truth.
    """
    plt.style.use(f'{base_dir}/Isabel-code/nn-topocluster-calibration/plotting-routine/plotting.mplstyle')

    fig, (ax, ax_ratio) = plt.subplots(
        2, 1,
        figsize=(2.953 * 1.6, 3.2221 * 1.3),
        gridspec_kw={'height_ratios': [4, 1.5], 'hspace': 0.05}
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.12,
        top=0.98
    )
    # Compute response ratio
    mask = np.isfinite(e_x) & (e_x > 0)
    e_x_filtered = e_x[mask]

    if e_x_filtered.size == 0:
        plt.close(fig)
        return
    
    # Create energy bins and compute relative resolution and response
    e_min = 0.0#np.nanmin(e_truth_filtered)
    e_max = 5.0#np.nanmax(e_truth_filtered)
    n_bins = 20
    energy_bins = np.linspace(e_min, e_max, n_bins)
    
    bin_centers = []
    sigma_model = []
    sigma_em = []

    for i in range(len(energy_bins) - 1):
        lo, hi = energy_bins[i], energy_bins[i + 1]
        if i == len(energy_bins) - 2:
            in_bin = (e_x_filtered >= lo) & (e_x_filtered <= hi)
        else:
            in_bin = (e_x_filtered >= lo) & (e_x_filtered < hi)

        if np.count_nonzero(in_bin) < 10:
            continue
        
        bin_centers.append(np.sqrt(lo * hi))
        r_model_bin = r_model[mask][in_bin]
        r_truth_bin = r_truth[mask][in_bin]
        
        # Compute 68% IQR (16th to 84th percentile)
        q84_model = np.nanpercentile(r_model_bin, 84.0)
        q16_model = np.nanpercentile(r_model_bin, 16.0)
        iqr_model_68 = q84_model - q16_model

        q84_em = np.nanpercentile(r_truth_bin, 84.0)
        q16_em = np.nanpercentile(r_truth_bin, 16.0)
        iqr_em_68 = q84_em - q16_em
        # Compute median
        median_model = np.nanmedian(r_model_bin)
        median_em = np.nanmedian(r_truth_bin)

        # Relative resolution: Q_68% / (2 * median)
        sigma_iqr_model = iqr_model_68 / (2.0 * median_model) if median_model > 0 else np.nan
        sigma_model.append(sigma_iqr_model)
        sigma_iqr_em = iqr_em_68 / (2.0 * median_em) if median_em > 0 else np.nan
        sigma_em.append(sigma_iqr_em)

    if len(bin_centers) == 0:
        plt.close(fig)
        return
    
    bin_centers = np.array(bin_centers)
    sigma_model = np.array(sigma_model)
    sigma_em = np.array(sigma_em)

    # Create dual-axis plot
    ax1 = ax
    #ax2 = ax1.twinx()
    
    # Plot relative resolution on primary axis
    line1 = ax1.plot(bin_centers, sigma_model, color=color_model, marker='o', markersize=4,
                     linewidth=1.5, label='HGM')
    
    # Plot true EM response on secondary axis
    line2 = ax1.plot(bin_centers, sigma_em, color=color_truth, marker='s', markersize=4,
                     linewidth=1.5, label='EM')
    
    #ax1.set_xlabel('$E^{clus}_{dep}[\\mathrm{GeV}]$')
    ax1.set_ylabel('Relative resolution $\\sigma$', color=color_model)
    #ax2.set_ylabel('True EM response $\\mathcal{R}^{\\mathrm{truth}}$', color=color_truth)
    
    ax1.tick_params(axis='y', labelcolor=color_model)
    #ax2.tick_params(axis='y', labelcolor=color_truth)
    
    #ax1.set_xscale('log')
    ax1.set_xlim((e_min, e_max))
    ax1.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax1.set_ylim(0, 2)

    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=8)

    ax1.tick_params(labelbottom=False)  # hide x labels on top

    ratio = sigma_model / sigma_em

    ax_ratio.plot(bin_centers, ratio,
                  color='black', marker='o', markersize=3,
                  linewidth=1.2)

    ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=1)

    ax_ratio.set_xlabel(x_label)
    ax_ratio.set_ylabel('Ratio (HGM/EM)')

    #ax_ratio.set_xscale('log')
    ax_ratio.set_xlim((e_min, e_max))

    ax_ratio.set_ylim(0, 2.0)  # adjust as needed

    add_textbox_to_ax(ax1)
    fig.savefig(os.path.join(output_path, f'{out_name}.png'), dpi=200)
    fig.savefig(os.path.join(output_path, f'{out_name}.pdf'), dpi=200)
    plt.close(fig)



def get_feature_info(config, use_tower_moments, default_setup):
    """Get list of features with their names and properties for plotting."""
    features = []
    
    if use_tower_moments:
        feature_list = [
            ("clusterE", "e_ems", r"$E_{clus}^{EM}$ [GeV]", True),
            ("clusterEta", "y_ems", r"Cluster $\eta$", False),
            ("nPrimVtx", "p_npv", r"$N_{PV}$", False),
            ("avgMu", "p_mmu", r"$\langle \mu \rangle$", False),
            ("cluster_FIRST_TOWER_PHI", "cluster_FIRST_TOWER_PHI", r"mean $\phi$(tower)", False),
            ("cluster_FIRST_TOWER_ETA", "cluster_FIRST_TOWER_ETA", r"mean $\eta$(tower)", False),
            ("cluster_FIRST_TOWER_R", "cluster_FIRST_TOWER_R", r"mean R(tower)", False),
            ("cluster_SECOND_TOWER_PHI", "cluster_SECOND_TOWER_PHI", r"var $\phi$(tower)", False),
            ("cluster_SECOND_TOWER_ETA", "cluster_SECOND_TOWER_ETA", r"var $\eta$(tower)", False),
            ("cluster_SECOND_TOWER_R", "cluster_SECOND_TOWER_R", r"R(tower)", False),
            ("cluster_TOWER_PTD", "cluster_TOWER_PTD", r"$p_T^D$ (tower)", False),
        ]
    elif default_setup:
        feature_list = [
            ("clusterE", "e_ems", r"$E_{\mathrm{clus}}^{\mathrm{EM}}\ [\mathrm{GeV}]$", True),
            ("clusterEta", "y_ems", r"Cluster $\eta$", False),
            ("cluster_SIGNIFICANCE", "m_sig", r"Cluster significance", True),
            ("cluster_time", "t_ems", r"Cluster time", False),
            ("cluster_SECOND_TIME", "m_tim", r"Cluster time (second moment)", True),
            ("cluster_CENTER_LAMBDA", "m_lam", r"Depth in calorimeter $\lambda_{\mathrm{clus}}\ [\mathrm{mm}]$", True),
            ("cluster_CENTER_MAG", "m_cog", r"Distance of COG [mm]", False),
            ("cluster_ENG_FRAC_EM_INCL", "m_fem", r"EM energy fraction", False),
            ("cluster_FIRST_ENG_DENS", "m_rho", r"Cluster signal density", True),
            ("cluster_LONGITUDINAL", "m_lon", r"Longitudinal signal dispersion", False),
            ("cluster_LATERAL", "m_lat", r"Lateral signal dispersion", False),
            ("cluster_PTD", "m_ptd", r"Signal compactness $p_{TD}$", False),
            ("cluster_ISOLATION", "m_iso", r"Cluster isolation", False),
            ("nPrimVtx", "p_npv", r"$N_{\mathrm{PV}}$", False),
            ("avgMu", "p_mmu", r"$\langle \mu \rangle$", False),
        ]
    else:
        feature_list = [
            ("clusterE", "e_ems", r"$E_{\mathrm{clus}}^{\mathrm{EM}}\ [\mathrm{GeV}]$", True),
            ("clusterEta", "y_ems", r"Cluster eta", False),
            ("cluster_SIGNIFICANCE", "m_sig", r"Cluster significance", True),
            ("cluster_time", "t_ems", r"Cluster time", False),
            ("cluster_SECOND_TIME", "m_tim", r"Cluster time (second moment)", True),
            ("cluster_CENTER_LAMBDA", "m_lam", r"Depth in calorimeter $\lambda_{\mathrm{clus}}\ [\mathrm{mm}]$", True),
            ("cluster_CENTER_MAG", "m_cog", r"Distance of COG [mm]", False),
            ("cluster_ENG_FRAC_EM_INCL", "m_fem", r"EM energy fraction", False),
            ("cluster_FIRST_ENG_DENS", "m_rho", r"Cluster signal density", True),
            ("cluster_LONGITUDINAL", "m_lon", r"Longitudinal signal dispersion", False),
            ("cluster_LATERAL", "m_lat", r"Lateral signal dispersion", False),
            ("cluster_PTD", "m_ptd", r"Signal compactness $p_{TD}$", False),
            ("cluster_ISOLATION", "m_iso", r"Cluster isolation", False),
            ("nPrimVtx", "p_npv", r"$N_{\mathrm{PV}}$", False),
            ("avgMu", "p_mmu", r"$\langle \mu \rangle$", False),
            ("cluster_closestDeltaR", "cluster_closestDeltaR", "\\Delta R_{closest}", False),
        ]
    
    # Filter to only include features in the config
    available_features = config["data"]["features"].keys()
    filtered_features = []
    for feat_name, trained_name, display_name, is_log in feature_list:
        if feat_name in available_features:
            filtered_features.append((feat_name, trained_name, display_name, is_log))
    
    return filtered_features


def plot_response_vs_features(output_path, data_before, data_after, r_truth, r_model, config, scales, use_tower_moments, default_setup):
    """Plot target response and predicted response vs each feature as 2D histograms."""
    figsize = (2.953, 3.2221)
    rect_single = (0.13, 0.12, 0.97, 0.98)
    subs = (0.2, 0.4, 0.6, 0.8, 1.0)
    num_bins = 50
    
    style_path = f'{base_dir}/Isabel-code/nn-topocluster-calibration/plotting-routine/plotting.mplstyle'
    if os.path.exists(style_path):
        plt.style.use(style_path)
    
    # Get feature information
    features = get_feature_info(config, use_tower_moments, default_setup)
    
    print(f"[info] Generating response vs feature plots for {len(features)} features")
    
    for feat_name, trained_name, display_name, is_log in features:
        if trained_name not in FEATURE_INDEX:
            continue
            
        idx = FEATURE_INDEX[trained_name]
        if idx >= data_before.shape[1]:
            continue
        
        # Get feature values (use raw, unpreprocessed data)
        feature_vals = data_before[:, idx].copy()
        
        # Filter out invalid values
        mask = np.isfinite(feature_vals) & np.isfinite(r_truth) & np.isfinite(r_model)
        feature_vals = feature_vals[mask]
        r_truth_filtered = r_truth[mask]
        r_model_filtered = r_model[mask]
        
        if feature_vals.size < 10:
            continue
        
        # Determine binning based on feature properties
        if is_log:
            # Log scale for features like energy, significance, etc.
            feat_min = np.nanpercentile(feature_vals, 0.1)
            feat_max = np.nanpercentile(feature_vals, 99.9)
            if feat_min <= 0:
                feat_min = np.nanmin(feature_vals[feature_vals > 0]) if np.any(feature_vals > 0) else 1e-10
            bins_feat = np.logspace(np.log10(feat_min), np.log10(feat_max), num_bins)
        else:
            # Linear scale for features like eta, phi, etc.
            feat_min = np.nanpercentile(feature_vals, 0.1)
            feat_max = np.nanpercentile(feature_vals, 99.9)
            bins_feat = np.linspace(feat_min, feat_max, num_bins)
        
        # Response bins (log scale)
        r_min = min(np.nanpercentile(r_truth_filtered, 0.1), np.nanpercentile(r_model_filtered, 0.1))
        r_max = max(np.nanpercentile(r_truth_filtered, 99.9), np.nanpercentile(r_model_filtered, 99.9))
        bins_response = np.logspace(np.log10(r_min), np.log10(r_max), num_bins)
        
        # Plot 1: Target response vs feature
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_single)
        
        ax.hist2d(feature_vals, r_truth_filtered, bins=[bins_feat, bins_response], 
                  norm=mpl.colors.LogNorm(), cmap='viridis')

        print("Display name: ", display_name)
        ax.set_xlabel(display_name)
        ax.set_ylabel('$\\mathcal{R}^{\\mathrm{EM}}_{\\mathrm{clus}}$')
        
        if is_log:
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        ax.set_yscale('log')
        ax.set_xlim((bins_feat[0], bins_feat[-1]))
        ax.set_ylim((bins_response[0], bins_response[-1]))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        # Save with sanitized filename
        safe_name = trained_name.replace('_', '-')
        add_textbox_to_ax(ax)
        fig.savefig(os.path.join(output_path, f'response_truth_vs_{safe_name}.png'), dpi=200)
        fig.savefig(os.path.join(output_path, f'response_truth_vs_{safe_name}.pdf'), dpi=200)
        plt.close(fig)
        
        # Plot 2: Predicted response vs feature
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_single)
        
        ax.hist2d(feature_vals, r_model_filtered, bins=[bins_feat, bins_response], 
                  norm=mpl.colors.LogNorm(), cmap='viridis')
        
        ax.set_xlabel(display_name)
        ax.set_ylabel('$\\mathcal{R}^{\\mathrm{HGM}}_{\\mathrm{clus}}$')
        
        if is_log:
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        ax.set_yscale('log')
        ax.set_xlim((bins_feat[0], bins_feat[-1]))
        ax.set_ylim((bins_response[0], bins_response[-1]))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        add_textbox_to_ax(ax)
        fig.savefig(os.path.join(output_path, f'response_model_vs_{safe_name}.png'), dpi=200)
        fig.savefig(os.path.join(output_path, f'response_model_vs_{safe_name}.pdf'), dpi=200)
        plt.close(fig)
    
    print(f"[info] Completed response vs feature plots")

def plot_energy_comparison(e_truth, e_ems, e_model, output_path):
    # make histogram and ratio histogram comparing e_ems and e_model
    plt.style.use(f'{base_dir}/Isabel-code/nn-topocluster-calibration/plotting-routine/plotting.mplstyle')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.953 * 1.6, 3.2221 * 1.3), gridspec_kw={'height_ratios': [4, 1.5], 'hspace': 0.05})
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.98)
    bins_log = np.logspace(np.log10(ENERGY_MIN), np.log10(ENERGY_MAX), NBINS)
    subs = (0.2, 0.4, 0.6, 0.8, 1.0)
    ax1.hist(e_truth, bins=bins_log, histtype='step', linewidth=1.5, label='Truth energy', color='black')
    ax1.hist(e_ems, bins=bins_log, histtype='step', linewidth=1.5, label='EM energy', color=color_truth)
    ax1.hist(e_model, bins=bins_log, histtype='step', linewidth=1.5, label='HGM energy', color=color_model)
    ax1.set_xscale('log')
    ax1.set_xlim((bins_log[0], bins_log[-1]))
    ax1.set_xlabel('$E_{clus}^{EM}$ [GeV]')
    ax1.set_ylabel('Number of clusters')
    ax1.legend()
    ax1.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # Ratio plot
    #ratio = np.histogram(e_model, bins=bins_log)[0] / np.histogram(e_ems, bins=bins_log)[0]
    ratio = np.histogram(e_ems, bins=bins_log)[0] / np.histogram(e_truth, bins=bins_log)[0]
    ratio_truth = np.histogram(e_model, bins=bins_log)[0] / np.histogram(e_truth, bins=bins_log)[0]
    bin_centers = np.sqrt(bins_log[:-1] * bins_log[1:])
    ax2.plot(bin_centers, ratio, marker='o', linestyle='-', label='EM/Truth', color='black')
    ax2.plot(bin_centers, ratio_truth, marker='s', linestyle='-', label='HGM/Truth', color='red')
    ax2.legend()
    ax2.axhline(1.0, color='gray', linestyle='--')
    ax2.set_xscale('log')
    ax2.set_xlim((bins_log[0], bins_log[-1]))
    ax2.set_xlabel('$E_{clus}$ [GeV]')
    ax2.set_ylabel('Ratio')
    ax2.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
    ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    add_textbox_to_ax(ax1)
    fig.savefig(os.path.join(output_path, 'energy_comparison.png'), dpi=200)
    fig.savefig(os.path.join(output_path, 'energy_comparison.pdf'), dpi=200)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser("Inference with saved preprocessing")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--pt_model", required=True, type=str)
    parser.add_argument("--scales", required=True, type=str)
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--batch_idxs", type=int, nargs="*", default=[])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--make_plots", action="store_true", help="Write comparison plots")
    parser.add_argument("--eta_min", type=float, default=None, help="Lower bound for cluster eta selection")
    parser.add_argument("--eta_max", type=float, default=None, help="Upper bound for cluster eta selection")
    parser.add_argument("--abs_eta", action="store_true", help="Select on absolute eta, i.e. |eta| in [eta_min, eta_max]")
    args = parser.parse_args()

    with open(args.config) as info:
        config = yaml.safe_load(info)

    model_name = config["model"]["model_name"]
    output_path = os.path.join(args.output_path, f"{model_name}_inf")
    os.makedirs(output_path, exist_ok=True)

    sys.stdout = open(os.path.join(output_path, "logfile-std.txt"), mode="w")
    sys.stderr = open(os.path.join(output_path, "logfile-err.txt"), mode="w")

    device = torch.device(args.device)
    print("[info] Starting inference")
    print(f"[info] Model name: {model_name}")
    print(f"[info] Output path: {output_path}")
    print(f"[info] Device: {device}")

    use_tower_moments, default_setup = resolve_feature_flags(config)
    print(f"[info] use_tower_moments={use_tower_moments}, default_setup={default_setup}")

    mc_release = config["model"]["data_args"].get("mc_release", "mc20")
    data_before = load_batches(args.input_path, args.batch_idxs, mc_release)
    print(f"[info] Loaded raw data: {data_before.shape}")

    # Optional eta-window preselection before preprocessing and inference.
    data_before, selected_indices = filter_eta_window(
        data_before,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        use_abs_eta=args.abs_eta,
    )
    eta_selection_label = format_eta_selection_label(
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        use_abs_eta=args.abs_eta,
    )
    selection_axis = "|eta|" if args.abs_eta else "eta"
    print(
        "[info] Eta selection "
        f"({selection_axis}, min={args.eta_min}, max={args.eta_max}) -> "
        f"{data_before.shape[0]} clusters"
    )
    if eta_selection_label is not None:
        print(f"[info] Eta selection label: {eta_selection_label}")
    np.save(os.path.join(output_path, "selected_input_indices.npy"), selected_indices)

    if data_before.shape[0] == 0:
        raise ValueError("No clusters remain after eta selection.")

    scales = np.load(args.scales, allow_pickle=True).item()
    data_after, applied_transforms = apply_preprocessing(data_before, config, scales, use_tower_moments, default_setup)
    print_preprocessing_summary(applied_transforms)
    data_after = select_columns(data_after, use_tower_moments, default_setup)

    batch_size_tst = config["model"]["train_args"]["batch_size_tst"]
    data_tensor = torch.from_numpy(data_after).float().to(device)
    dataloader = DataLoader(data_tensor, batch_size=batch_size_tst, shuffle=False)

    dim_input = data_after.shape[1] - 1
    layers = config["model"]["init_args"]["layers"]
    activation = config["model"]["init_args"]["activation"]
    num_mixtures = config["model"]["init_args"]["num_mixtures"]
    use_residual = config["model"]["init_args"].get("use_residual", False)

    energy_residual_cfg = config["model"].get("energy_residual", {})
    low_energy_cfg = config["model"].get("low_energy_head", {})

    model = GaussianMixtureDNN(
        device,
        layers,
        dim_input,
        num_mixtures,
        activation=activation,
        use_residual=use_residual,
        energy_residual=energy_residual_cfg.get("enabled", False),
        energy_residual_feature_index=energy_residual_cfg.get("feature_index", 0),
        energy_residual_hidden=energy_residual_cfg.get("hidden_dim", 32),
        low_energy_head=low_energy_cfg.get("enabled", False),
        low_energy_threshold=low_energy_cfg.get("threshold", 10.0),
        low_energy_hidden=low_energy_cfg.get("hidden_dim", 128),
    ).to(device)

    load_checkpoint(model, args.pt_model)
    model.eval()

    prediction_name = list(config["model"]["prediction"].keys())[0]
    apply_residual = energy_residual_cfg.get("enabled", False)

    y_truth, y_model, s_model, mus, sigma2s, alphas = model.get_predictions(
        dataloader,
        prediction_name,
        apply_residual=apply_residual,
    )

    r_truth = 10 ** y_truth
    r_model = 10 ** y_model
    s_model = np.abs(np.log(10) * r_model) * s_model
    sigmas = np.sqrt(sigma2s)

    np.save(os.path.join(output_path, "final_r_truth_tst.npy"), r_truth)
    np.save(os.path.join(output_path, "final_r_model_tst.npy"), r_model)
    np.save(os.path.join(output_path, "final_s_model_tst.npy"), s_model)
    np.save(os.path.join(output_path, "final_mus_tst.npy"), mus)
    np.save(os.path.join(output_path, "final_alphas_tst.npy"), alphas)
    np.save(os.path.join(output_path, "final_sigmas_tst.npy"), sigmas)

    if args.make_plots:
        e_truth = compute_true_energy(data_before, r_truth)
        e_model = compute_pred_energy(data_before, r_model)
        plot_energy_scatter(output_path, e_truth, e_model)
        plot_response_scatter(output_path, r_truth, r_model)
        #plot_iqr_vs_energy(output_path, e_truth, r_truth, r_model)
        plot_iqr_vs_energy(output_path, e_truth, r_truth, r_model, x_label='$E^{clus}_{truth}[\\mathrm{GeV}]$', out_name='sigma_rel_response_vs_truth_energy'
        )

        e_ems = data_before[:, FEATURE_INDEX["e_ems"]]
        plot_energy_comparison(e_truth, e_ems, e_model, output_path)
        plot_iqr_vs_energy( output_path, e_ems, r_truth, r_model, x_label='$E^{clus}_{EM}[\\mathrm{GeV}]$', out_name='sigma_rel_response_vs_em_energy'
        )
        eta_clus = data_before[:,FEATURE_INDEX["y_ems"]]
        plot_iqr_vs_energy_eta( output_path, eta_clus, r_truth, r_model, x_label='Cluster eta', out_name='sigma_rel_response_vs_cluster_eta'
        )
        plot_response_vs_features(output_path, data_before, data_after, r_truth, r_model,
                                  config, scales, use_tower_moments, default_setup)

    print("[info] Inference done")
    sys.stdout.close()


if __name__ == "__main__":
    main()
