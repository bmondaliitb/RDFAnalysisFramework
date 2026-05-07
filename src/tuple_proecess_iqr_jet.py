import argparse
import ROOT

import sys
sys.path.append("/mnt/nvme0n1p4/HEP/Work/Project-W-Z/MotorHead/ntuple-processing-tool/run/hadrecoil-studies/")

from NtupleProcessorRDF import NtupleProcessorRDF
from helper_function import *

import numpy as np

debug = False
ROOT.gROOT.SetBatch(True) # prevents fit window gui

class NtupleProcessorRDF_jet(NtupleProcessorRDF):
    def build_dataframe(self):
        df = self.df

        df = df.Define("d_jet_pt", "jet_pt") # this is already in GeV
        df = df.Define("d_jet_eta", "jet_eta")
        df = df.Define("d_jet_phi", "jet_phi")
        df = df.Define("d_jet_m", "jet_m")
        df = df.Define("d_jet_e_EM", "jet_e_EM")
        # truth jet kinematics
        df = df.Define("d_jet_pt_truth", "jet_pt_truth")
        df = df.Define("d_jet_eta_truth", "jet_eta_truth")
        df = df.Define("d_jet_phi_truth", "jet_phi_truth")
        df = df.Define("d_jet_e_truth", "jet_e_truth")

        # inside jet cluster variables
        df = df.Define("d_cluster_pt", "cluster_pt_inJet")
        df = df.Define("d_cluster_eta", "cluster_eta_inJet")
        df = df.Define("d_cluster_phi", "cluster_phi_inJet")
        df = df.Define("d_cluster_e_truth", "cluster_e_truth_inJet")
        df = df.Define("d_cluster_e_EM", "cluster_e_EM_inJet")
        df = df.Define("d_cluster_e_ML", "cluster_e_ML_inJet")
        df = df.Define("d_cluster_e_LC", "cluster_e_LC_inJet")

        self.df = df
        return df


def get_jet_energy(df_e):
    return np.sum(df_e)

def main(args, out_file):
    base_tdirectory = ROOT.gDirectory

    proc = NtupleProcessorRDF_jet(args.input, args.tree, args.nEvents)
    proc.build_dataframe()
    df_jet_pt = proc.to_pandas(["d_jet_pt"])
    df_jet_eta = proc.to_pandas(["d_jet_eta"])
    df_jet_phi = proc.to_pandas(["d_jet_phi"])
    df_jet_e_EM = proc.to_pandas(["d_jet_e_EM"])

    # truth jet kinematics
    df_jet_pt_truth = proc.to_pandas(["d_jet_pt_truth"])
    df_jet_e_truth = proc.to_pandas(["d_jet_e_truth"])


    df_cluster_pt = proc.to_pandas(["d_cluster_pt"])
    df_cluster_eta = proc.to_pandas(["d_cluster_eta"])
    df_cluster_phi = proc.to_pandas(["d_cluster_phi"])
    df_cluster_e_truth = proc.to_pandas(["d_cluster_e_truth"])
    df_cluster_e_EM = proc.to_pandas(["d_cluster_e_EM"])
    df_cluster_e_ML = proc.to_pandas(["d_cluster_e_ML"])
    df_cluster_e_LC = proc.to_pandas(["d_cluster_e_LC"])

    # loop over event
    total_events = df_jet_pt.shape[0]
    event_counter = 0

    # Dictionary of jet energy recalculated from clusters at different scales
    jet_energy_recal_dict = {
        "EM": [],
        "ML": [],
        "LC": [],
        "truth": []
    }

    jet_pt_dict = {
        "EM": [],
        "truth": []
    }

    jet_energy_dict = {
        "EM": [],
        "truth": []
    }

    # x-bin edges for IQR histograms
    #x_bin_edges = get_bins_log(20, 500, 30)
    #x_bin_edges = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500]
    #x_bin_edges = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240, 280]
    x_bin_edges = [20, 30, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
            160, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
            1800, 1900, 2000, 2100, 2300, 2500, 2700, 2900, 3200, 3900, 5000]
    #x_bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
    #        160, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
    #        1800, 1900, 2000, 2100, 2300, 2500, 2700, 2900, 3200, 3900, 5000]

    for event in range(0, total_events):
        # print every 10000 events
        if event_counter % 10000 == 0:
            print("[Info]:: Processing event {}/{}".format(event, total_events))

        event_jet_pt = np.array(df_jet_pt.iloc[event]["d_jet_pt"])
        event_jet_eta = np.array(df_jet_eta.iloc[event]["d_jet_eta"])
        event_jet_phi = np.array(df_jet_phi.iloc[event]["d_jet_phi"])
        event_jet_e_EM = np.array(df_jet_e_EM.iloc[event]["d_jet_e_EM"])
        event_jet_pt_truth = np.array(df_jet_pt_truth.iloc[event]["d_jet_pt_truth"])
        event_jet_e_truth = np.array(df_jet_e_truth.iloc[event]["d_jet_e_truth"])

        # loop over jets in event

        for jet in range(0, len(event_jet_pt)):
            event_cluster_pt = np.array(df_cluster_pt.iloc[event]["d_cluster_pt"][jet])
            event_cluster_eta = np.array(df_cluster_eta.iloc[event]["d_cluster_eta"][jet])
            event_cluster_phi = np.array(df_cluster_phi.iloc[event]["d_cluster_phi"][jet])
            event_cluster_e_truth = np.array(df_cluster_e_truth.iloc[event]["d_cluster_e_truth"][jet])
            event_cluster_e_EM = np.array(df_cluster_e_EM.iloc[event]["d_cluster_e_EM"][jet])
            event_cluster_e_ML = np.array(df_cluster_e_ML.iloc[event]["d_cluster_e_ML"][jet])
            event_cluster_e_LC = np.array(df_cluster_e_LC.iloc[event]["d_cluster_e_LC"][jet])

            jet_energy_recal_EM = get_jet_energy(event_cluster_e_EM)
            jet_energy_recal_ML = get_jet_energy(event_cluster_e_ML)
            jet_energy_recal_LC = get_jet_energy(event_cluster_e_LC)
            jet_energy_recal_truth = get_jet_energy(event_cluster_e_truth)

            # append to dict
            jet_energy_recal_dict["EM"].append(jet_energy_recal_EM)
            jet_energy_recal_dict["ML"].append(jet_energy_recal_ML)
            jet_energy_recal_dict["LC"].append(jet_energy_recal_LC)
            jet_energy_recal_dict["truth"].append(jet_energy_recal_truth)

            # append to dict jet pt
            jet_pt_dict["EM"].append(event_jet_pt[jet])
            jet_pt_dict["truth"].append(event_jet_pt_truth[jet])
            jet_energy_dict["EM"].append(event_jet_e_EM[jet])
            jet_energy_dict["truth"].append(event_jet_e_truth[jet])

        event_counter += 1

    # print shape of the jet_pt and jet_energy_recal arrays
    if debug:
        print("Shape of jet_pt_dict['EM']: ", np.array(jet_pt_dict["EM"]).shape)
        print("Shape of jet_energy_recal_dict['EM']: ", np.array(jet_energy_recal_dict["EM"]).shape)

    # Create response E^jet_Reco/E^jet_truth
    jet_response_dict = {
        "EM": [],
        "ML": [],
        "LC": [],
        "truth": []
    }
    for scale in ["EM", "ML", "LC", "truth"]:
        jet_response_dict[scale] = np.array(jet_energy_recal_dict[scale]) / np.array(jet_energy_dict["truth"])

    jet_response_cluster_truth_dict = {
        "EM": [],
        "ML": [],
        "LC": [],
        "truth": []
    }
    for scale in ["EM", "ML", "LC", "truth"]:
        jet_response_cluster_truth_dict[scale] = np.array(jet_energy_recal_dict[scale]) / np.array(jet_energy_recal_dict["truth"])
    # get IQR with gaussian fit

    # resolution of jet energy response (reco/truth) vs jet truth pt, truth energy, truth energy (cluster)
    mean_em_jet_response_vs_truth_jet_energy, sigma_em_jet_response_vs_truth_jet_energy = calculate_gaussian_fit_params_from_arrays(
        jet_energy_dict["truth"], jet_response_dict["EM"], x_bin_edges, out_file=out_file, y_name="em_jet_response_vs_truth_jet_energy")
    mean_ml_jet_response_vs_truth_jet_energy, sigma_ml_jet_response_vs_truth_jet_energy = calculate_gaussian_fit_params_from_arrays(
        jet_energy_dict["truth"], jet_response_dict["ML"], x_bin_edges, out_file=out_file, y_name="ml_jet_response_vs_truth_jet_energy")
    mean_lc_jet_response_vs_truth_jet_energy, sigma_lc_jet_response_vs_truth_jet_energy = calculate_gaussian_fit_params_from_arrays(
        jet_energy_dict["truth"], jet_response_dict["LC"], x_bin_edges, out_file=out_file, y_name="lc_jet_response_vs_truth_jet_energy")

    ## truth jet pt
    mean_em_jet_response_vs_truth_jet_pt, sigma_em_jet_response_vs_truth_jet_pt = calculate_gaussian_fit_params_from_arrays(
        jet_pt_dict["truth"], jet_response_dict["EM"], x_bin_edges, out_file=out_file, y_name="em_jet_response_vs_truth_jet_pt")
    mean_ml_jet_response_vs_truth_jet_pt, sigma_ml_jet_response_vs_truth_jet_pt = calculate_gaussian_fit_params_from_arrays(
        jet_pt_dict["truth"], jet_response_dict["ML"], x_bin_edges, out_file=out_file, y_name="ml_jet_response_vs_truth_jet_pt")
    mean_lc_jet_response_vs_truth_jet_pt, sigma_lc_jet_response_vs_truth_jet_pt = calculate_gaussian_fit_params_from_arrays(
        jet_pt_dict["truth"], jet_response_dict["LC"], x_bin_edges, out_file=out_file, y_name="lc_jet_response_vs_truth_jet_pt")

    ## resolution of jet response (reco/truth) vs jet truth pt, truth energy, truth energy (cluster)
    mean_em_jet_response_cluster_truth_vs_truth_jet_energy, sigma_em_jet_response_cluster_truth_vs_truth_jet_energy = calculate_gaussian_fit_params_from_arrays(
        jet_energy_dict["truth"], jet_response_cluster_truth_dict["EM"], x_bin_edges, out_file=out_file, y_name="em_jet_response_cluster_truth_vs_truth_jet_energy")
    mean_ml_jet_response_cluster_truth_vs_truth_jet_energy, sigma_ml_jet_response_cluster_truth_vs_truth_jet_energy = calculate_gaussian_fit_params_from_arrays(
        jet_energy_dict["truth"], jet_response_cluster_truth_dict["ML"], x_bin_edges, out_file=out_file, y_name="ml_jet_response_cluster_truth_vs_truth_jet_energy")
    mean_lc_jet_response_cluster_truth_vs_truth_jet_energy, sigma_lc_jet_response_cluster_truth_vs_truth_jet_energy = calculate_gaussian_fit_params_from_arrays(
        jet_energy_dict["truth"], jet_response_cluster_truth_dict["LC"], x_bin_edges, out_file=out_file, y_name="lc_jet_response_cluster_truth_vs_truth_jet_energy")

    ## truth jet pt
    mean_em_jet_response_cluster_truth_vs_truth_jet_pt, sigma_em_jet_response_cluster_truth_vs_truth_jet_pt = calculate_gaussian_fit_params_from_arrays(
        jet_pt_dict["truth"], jet_response_cluster_truth_dict["EM"], x_bin_edges, out_file=out_file, y_name="em_jet_response_cluster_truth_vs_truth_jet_pt")
    mean_ml_jet_response_cluster_truth_vs_truth_jet_pt, sigma_ml_jet_response_cluster_truth_vs_truth_jet_pt = calculate_gaussian_fit_params_from_arrays(
        jet_pt_dict["truth"], jet_response_cluster_truth_dict["ML"], x_bin_edges, out_file=out_file, y_name="ml_jet_response_cluster_truth_vs_truth_jet_pt")
    mean_lc_jet_response_cluster_truth_vs_truth_jet_pt, sigma_lc_jet_response_cluster_truth_vs_truth_jet_pt = calculate_gaussian_fit_params_from_arrays(
        jet_pt_dict["truth"], jet_response_cluster_truth_dict["LC"], x_bin_edges, out_file=out_file, y_name="lc_jet_response_cluster_truth_vs_truth_jet_pt")

    ## vs jet energy (cluster truth energy)
    mean_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy, sigma_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy = calculate_gaussian_fit_params_from_arrays(
        jet_energy_recal_dict["truth"], jet_response_cluster_truth_dict["EM"], x_bin_edges, out_file=out_file, y_name="em_jet_response_cluster_truth_vs_cluster_truth_jet_energy")
    mean_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy, sigma_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy = calculate_gaussian_fit_params_from_arrays(
        jet_energy_recal_dict["truth"], jet_response_cluster_truth_dict["ML"], x_bin_edges, out_file=out_file, y_name="ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy")
    mean_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy, sigma_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy = calculate_gaussian_fit_params_from_arrays(
        jet_energy_recal_dict["truth"], jet_response_cluster_truth_dict["LC"], x_bin_edges, out_file=out_file, y_name="lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy")

    # make histograms of sigmas
    h_sigma_em_jet_response_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_sigma_em_jet_response_vs_truth_jet_energy", "", x_bin_edges,
        sigma_em_jet_response_vs_truth_jet_energy/mean_em_jet_response_vs_truth_jet_energy)
    h_sigma_ml_jet_response_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_sigma_ml_jet_response_vs_truth_jet_energy", "", x_bin_edges,
        sigma_ml_jet_response_vs_truth_jet_energy/mean_ml_jet_response_vs_truth_jet_energy)
    h_sigma_lc_jet_response_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_sigma_lc_jet_response_vs_truth_jet_energy", "", x_bin_edges,
        sigma_lc_jet_response_vs_truth_jet_energy/mean_lc_jet_response_vs_truth_jet_energy)
    h_sigma_em_jet_response_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_sigma_em_jet_response_vs_truth_jet_pt", "", x_bin_edges, sigma_em_jet_response_vs_truth_jet_pt/mean_em_jet_response_vs_truth_jet_pt)
    h_sigma_ml_jet_response_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_sigma_ml_jet_response_vs_truth_jet_pt", "", x_bin_edges,
        sigma_ml_jet_response_vs_truth_jet_pt/mean_ml_jet_response_vs_truth_jet_pt)
    h_sigma_lc_jet_response_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_sigma_lc_jet_response_vs_truth_jet_pt", "", x_bin_edges,
        sigma_lc_jet_response_vs_truth_jet_pt/mean_lc_jet_response_vs_truth_jet_pt)
    h_sigma_em_jet_response_cluster_truth_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_sigma_em_jet_response_cluster_truth_vs_truth_jet_energy","", x_bin_edges,
        sigma_em_jet_response_cluster_truth_vs_truth_jet_energy/mean_em_jet_response_cluster_truth_vs_truth_jet_energy)
    h_sigma_ml_jet_response_cluster_truth_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_sigma_ml_jet_response_cluster_truth_vs_truth_jet_energy", "",
        x_bin_edges, sigma_ml_jet_response_cluster_truth_vs_truth_jet_energy/mean_ml_jet_response_cluster_truth_vs_truth_jet_energy
    )
    h_sigma_lc_jet_response_cluster_truth_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_sigma_lc_jet_response_cluster_truth_vs_truth_jet_energy", "",
        x_bin_edges, sigma_lc_jet_response_cluster_truth_vs_truth_jet_energy/mean_lc_jet_response_cluster_truth_vs_truth_jet_energy
    )
    h_sigma_em_jet_response_cluster_truth_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_sigma_em_jet_response_cluster_truth_vs_truth_jet_pt","",
        x_bin_edges, sigma_em_jet_response_cluster_truth_vs_truth_jet_pt/mean_em_jet_response_cluster_truth_vs_truth_jet_pt
    )
    h_sigma_ml_jet_response_cluster_truth_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_sigma_ml_jet_response_cluster_truth_vs_truth_jet_pt","",
        x_bin_edges,sigma_ml_jet_response_cluster_truth_vs_truth_jet_pt/mean_ml_jet_response_cluster_truth_vs_truth_jet_pt
    )
    h_sigma_lc_jet_response_cluster_truth_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_sigma_lc_jet_response_cluster_truth_vs_truth_jet_pt", "", x_bin_edges,
        sigma_lc_jet_response_cluster_truth_vs_truth_jet_pt/mean_lc_jet_response_cluster_truth_vs_truth_jet_pt
    )
    h_sigma_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy = make_hist_from_bin_contents(
        f"h_sigma_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy","",
        x_bin_edges, sigma_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy/mean_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy
    )
    h_sigma_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy = make_hist_from_bin_contents(
        f"h_sigma_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy","", x_bin_edges,
        sigma_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy/mean_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy
    )
    h_sigma_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy = make_hist_from_bin_contents(
        f"h_sigma_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy","", x_bin_edges,
        sigma_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy/mean_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy
    )

    ## make the same histograms but for mean
    h_mean_em_jet_response_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_mean_em_jet_response_vs_truth_jet_energy", "", x_bin_edges,
        mean_em_jet_response_vs_truth_jet_energy)
    h_mean_ml_jet_response_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_mean_ml_jet_response_vs_truth_jet_energy", "", x_bin_edges,
        mean_ml_jet_response_vs_truth_jet_energy)
    h_mean_lc_jet_response_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_mean_lc_jet_response_vs_truth_jet_energy", "", x_bin_edges,
        mean_lc_jet_response_vs_truth_jet_energy)
    h_mean_em_jet_response_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_mean_em_jet_response_vs_truth_jet_pt", "", x_bin_edges, mean_em_jet_response_vs_truth_jet_pt)
    h_mean_ml_jet_response_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_mean_ml_jet_response_vs_truth_jet_pt", "", x_bin_edges,
        mean_ml_jet_response_vs_truth_jet_pt)
    h_mean_lc_jet_response_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_mean_lc_jet_response_vs_truth_jet_pt", "", x_bin_edges,
        mean_lc_jet_response_vs_truth_jet_pt)
    h_mean_em_jet_response_cluster_truth_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_mean_em_jet_response_cluster_truth_vs_truth_jet_energy","", x_bin_edges,
        mean_em_jet_response_cluster_truth_vs_truth_jet_energy)
    h_mean_ml_jet_response_cluster_truth_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_mean_ml_jet_response_cluster_truth_vs_truth_jet_energy", "",x_bin_edges,
        mean_ml_jet_response_cluster_truth_vs_truth_jet_energy)
    h_mean_lc_jet_response_cluster_truth_vs_truth_jet_energy = make_hist_from_bin_contents(
        f"h_mean_lc_jet_response_cluster_truth_vs_truth_jet_energy", "", x_bin_edges,
        mean_lc_jet_response_cluster_truth_vs_truth_jet_energy)
    h_mean_em_jet_response_cluster_truth_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_mean_em_jet_response_cluster_truth_vs_truth_jet_pt","", x_bin_edges,
        mean_em_jet_response_cluster_truth_vs_truth_jet_pt)
    h_mean_ml_jet_response_cluster_truth_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_mean_ml_jet_response_cluster_truth_vs_truth_jet_pt","", x_bin_edges,
        mean_ml_jet_response_cluster_truth_vs_truth_jet_pt)
    h_mean_lc_jet_response_cluster_truth_vs_truth_jet_pt = make_hist_from_bin_contents(
        f"h_mean_lc_jet_response_cluster_truth_vs_truth_jet_pt", "", x_bin_edges,
        mean_lc_jet_response_cluster_truth_vs_truth_jet_pt)
    h_mean_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy = make_hist_from_bin_contents(
        f"h_mean_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy","", x_bin_edges,
        mean_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy)
    h_mean_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy = make_hist_from_bin_contents(
        f"h_mean_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy","", x_bin_edges,
        mean_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy)
    h_mean_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy = make_hist_from_bin_contents(
        f"h_mean_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy","", x_bin_edges,
        mean_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy)

    base_tdirectory.cd()
    h_sigma_em_jet_response_vs_truth_jet_energy.Write()
    h_sigma_ml_jet_response_vs_truth_jet_energy.Write()
    h_sigma_lc_jet_response_vs_truth_jet_energy.Write()
    h_sigma_em_jet_response_vs_truth_jet_pt.Write()
    h_sigma_ml_jet_response_vs_truth_jet_pt.Write()
    h_sigma_lc_jet_response_vs_truth_jet_pt.Write()
    h_sigma_em_jet_response_cluster_truth_vs_truth_jet_energy.Write()
    h_sigma_ml_jet_response_cluster_truth_vs_truth_jet_energy.Write()
    h_sigma_lc_jet_response_cluster_truth_vs_truth_jet_energy.Write()
    h_sigma_em_jet_response_cluster_truth_vs_truth_jet_pt.Write()
    h_sigma_ml_jet_response_cluster_truth_vs_truth_jet_pt.Write()
    h_sigma_lc_jet_response_cluster_truth_vs_truth_jet_pt.Write()
    h_sigma_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy.Write()
    h_sigma_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy.Write()
    h_sigma_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy.Write()

    # mean histograms
    h_mean_em_jet_response_vs_truth_jet_energy.Write()
    h_mean_ml_jet_response_vs_truth_jet_energy.Write()
    h_mean_lc_jet_response_vs_truth_jet_energy.Write()
    h_mean_em_jet_response_vs_truth_jet_pt.Write()
    h_mean_ml_jet_response_vs_truth_jet_pt.Write()
    h_mean_lc_jet_response_vs_truth_jet_pt.Write()
    h_mean_em_jet_response_cluster_truth_vs_truth_jet_energy.Write()
    h_mean_ml_jet_response_cluster_truth_vs_truth_jet_energy.Write()
    h_mean_lc_jet_response_cluster_truth_vs_truth_jet_energy.Write()
    h_mean_em_jet_response_cluster_truth_vs_truth_jet_pt.Write()
    h_mean_ml_jet_response_cluster_truth_vs_truth_jet_pt.Write()
    h_mean_lc_jet_response_cluster_truth_vs_truth_jet_pt.Write()
    h_mean_em_jet_response_cluster_truth_vs_cluster_truth_jet_energy.Write()
    h_mean_ml_jet_response_cluster_truth_vs_cluster_truth_jet_energy.Write()
    h_mean_lc_jet_response_cluster_truth_vs_cluster_truth_jet_energy.Write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process had recoil tuples with RDF and save IQR histograms to ROOT.")
    parser.add_argument("--input", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", required=True, help="Tree name.")
    parser.add_argument("--output", default="iqr_histograms.root", help="Output ROOT file path.")
    parser.add_argument("--nEvents", type=int, default=10000, help="Number of events to process.")
    args = parser.parse_args()

    out_file = ROOT.TFile(args.output, "RECREATE")
    main(args, out_file)
    out_file.Close()
