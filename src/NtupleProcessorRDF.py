#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import ROOT

from hadrecoil import *
from helper_function import *


class NtupleProcessorRDF:
    def __init__(self, input_file, tree_name, n_events):
        self.input_file = input_file
        self.tree_name = tree_name
        self.n_events = n_events
        self.df = ROOT.RDataFrame(tree_name, input_file)
        if n_events is not None and n_events > 0:
            self.df = self.df.Range(n_events)

    def build_dataframe(self):
        df = self.df

        # Event selection
        df = df.Filter("mu_pt.size() > 1")

        # Scalar projections (adjust if branch layout differs)
        df = df.Define("mu_pt0", "mu_pt[0]/1000")
        # Recoil definitions
        # index 9  : EM scale / baseline
        # index 10 : LCW calibration
        # index 11 : ML calibration
        df = df.Define("u_pt_truth", "tu_pt.second[0]/1000")

        df = df.Define("u_pt_em",  "u_pt.second[9]/1000")
        df = df.Define("u_pt_lcw", "u_pt.second[10]/1000")
        df = df.Define("u_pt_ml",  "u_pt.second[11]/1000")

        df = df.Define("z_x", "mu_pt[0]/1000*cos(mu_phi[0]) + mu_pt[1]/1000*cos(mu_phi[1])")
        df = df.Define("z_y", "mu_pt[0]/1000*sin(mu_phi[0]) + mu_pt[1]/1000*sin(mu_phi[1])")
        df = df.Define("z_r", "sqrt(z_x*z_x + z_y*z_y)")

        df = df.Define("u_x_truth",  "tu_pt.second[0]/1000*cos(tu_phi.second[0])")
        df = df.Define("u_y_truth",  "tu_pt.second[0]/1000*sin(tu_phi.second[0])")
        df = df.Define("u_x_em",  "u_pt.second[9]/1000*cos(u_phi.second[9])")
        df = df.Define("u_y_em",  "u_pt.second[9]/1000*sin(u_phi.second[9])")
        df = df.Define("u_x_lcw", "u_pt.second[10]/1000*cos(u_phi.second[10])")
        df = df.Define("u_y_lcw", "u_pt.second[10]/1000*sin(u_phi.second[10])")
        df = df.Define("u_x_ml",  "u_pt.second[11]/1000*cos(u_phi.second[11])")
        df = df.Define("u_y_ml",  "u_pt.second[11]/1000*sin(u_phi.second[11])")

        df = df.Define("u_perp_truth",  "(z_x*u_y_truth- z_y*u_x_truth)/std::max(z_r, 1e-9f)")
        df = df.Define("u_perp_em",  "(z_x*u_y_em  - z_y*u_x_em )/std::max(z_r, 1e-9f)")
        df = df.Define("u_perp_lcw", "(z_x*u_y_lcw - z_y*u_x_lcw)/std::max(z_r, 1e-9f)")
        df = df.Define("u_perp_ml",  "(z_x*u_y_ml  - z_y*u_x_ml )/std::max(z_r, 1e-9f)")
        
        # Truth / boson kinematics aliases
        df = df.Define("qT", "z_r")
        df = df.Define("q_x", "z_x")
        df = df.Define("q_y", "z_y")

        # Unit vectors along and perpendicular to boson direction
        df = df.Define("qhat_x", "z_x/std::max(z_r, 1e-9f)")
        df = df.Define("qhat_y", "z_y/std::max(z_r, 1e-9f)")
        df = df.Define("qhat_perp_x", "-qhat_y")
        df = df.Define("qhat_perp_y", " qhat_x")

        # Optional recoil magnitudes
        df = df.Define("u_mag_truth",  "sqrt(u_x_truth*u_x_truth + u_y_truth*u_y_truth)")
        df = df.Define("u_mag_em",  "sqrt(u_x_em*u_x_em + u_y_em*u_y_em)")
        df = df.Define("u_mag_lcw", "sqrt(u_x_lcw*u_x_lcw + u_y_lcw*u_y_lcw)")
        df = df.Define("u_mag_ml",  "sqrt(u_x_ml*u_x_ml + u_y_ml*u_y_ml)")

        # Parallel recoil components: u_parallel = u · qhat
        df = df.Define("u_par_truth",  "(u_x_truth*z_x   + u_y_truth*z_y )/std::max(z_r, 1e-9f)")
        df = df.Define("u_par_em",  "(u_x_em*z_x   + u_y_em*z_y )/std::max(z_r, 1e-9f)")
        df = df.Define("u_par_lcw", "(u_x_lcw*z_x  + u_y_lcw*z_y)/std::max(z_r, 1e-9f)")
        df = df.Define("u_par_ml",  "(u_x_ml*z_x   + u_y_ml*z_y )/std::max(z_r, 1e-9f)")

        # Perpendicular recoil components: u_perp = u · qhat_perp
        # equivalent to current cross-product form, but named to match the note
        df = df.Define("u_perp_truth_alt",  "u_x_truth*qhat_perp_x   + u_y_truth*qhat_perp_y")
        df = df.Define("u_perp_em_alt",  "u_x_em*qhat_perp_x   + u_y_em*qhat_perp_y")
        df = df.Define("u_perp_lcw_alt", "u_x_lcw*qhat_perp_x  + u_y_lcw*qhat_perp_y")
        df = df.Define("u_perp_ml_alt",  "u_x_ml*qhat_perp_x   + u_y_ml*qhat_perp_y")

        # Response-style observable: u_parallel + qT
        df = df.Define("u_par_plus_qT_truth",  "u_par_truth  + qT")
        df = df.Define("u_par_plus_qT_em",  "u_par_em  + qT")
        df = df.Define("u_par_plus_qT_lcw", "u_par_lcw + qT")
        df = df.Define("u_par_plus_qT_ml",  "u_par_ml  + qT")

        # define lepton pt, eta, phi
        #df = df.Define("lep_pt_0", "mu_pt[0]/1000")
        #df = df.Define("lep_eta_0", "mu_eta[0]")
        #df = df.Define("lep_phi_0", "mu_phi[0]")
        #df = df.Define("lep_pt_1", "mu_pt[1]/1000")
        #df = df.Define("lep_eta_1", "mu_eta[1]")
        #df = df.Define("lep_phi_1", "mu_phi[1]")
        df = df.Define("lep_pt", "mu_pt/1000")
        df = df.Define("lep_eta", "mu_eta")
        df = df.Define("lep_phi", "mu_phi")

        # define cluster energy, eta, phi
        df = df.Define("clus_e_truth", "fCluster_truthE/1000")
        df = df.Define("clus_e_em", "fCluster_rawE/1000")
        df = df.Define("clus_e_lcw", "fCluster_calE/1000")
        df = df.Define("clus_e_ml", "fCluster_MLE/1000")
        df = df.Define("clus_eta", "fClusterEta")
        df = df.Define("clus_phi", "fClusterPhi")

        # MET truth particles pt, eta, phi energy
        df = df.Define("met_truth_particle_pt", "fMETTruthPt/1000")
        df = df.Define("met_truth_particle_eta", "fMETTruthEta")
        df = df.Define("met_truth_particle_phi", "fMETTruthPhi")
        df = df.Define("met_truth_particle_e", "fMETTruthE/1000")

        self.df = df
        return df

    def to_pandas(self, columns):
        rdf_cols = self.df.AsNumpy(columns)
        return pd.DataFrame(rdf_cols)

