import numpy as np

from rdf_analysis.core import NtupleProcessorRDF
from rdf_analysis.physics import hadrecoil_from_truth_particles, hadronic_recoil_from_clusters_python


class NtupleProcessorRDF_hadrecoil(NtupleProcessorRDF):
    def build_dataframe(self):
        df = self.df

        df = df.Filter("mu_pt.size() > 1")
        df = df.Define("mu_pt0", "mu_pt[0]/1000")
        df = df.Define("u_pt_truth", "tu_pt.second[0]/1000")
        df = df.Define("u_pt_em", "u_pt.second[9]/1000")
        df = df.Define("u_pt_lcw", "u_pt.second[10]/1000")
        df = df.Define("u_pt_ml", "u_pt.second[11]/1000")

        df = df.Define("z_x", "mu_pt[0]/1000*cos(mu_phi[0]) + mu_pt[1]/1000*cos(mu_phi[1])")
        df = df.Define("z_y", "mu_pt[0]/1000*sin(mu_phi[0]) + mu_pt[1]/1000*sin(mu_phi[1])")
        df = df.Define("z_r", "sqrt(z_x*z_x + z_y*z_y)")

        df = df.Define("u_x_truth", "tu_pt.second[0]/1000*cos(tu_phi.second[0])")
        df = df.Define("u_y_truth", "tu_pt.second[0]/1000*sin(tu_phi.second[0])")
        df = df.Define("u_x_em", "u_pt.second[9]/1000*cos(u_phi.second[9])")
        df = df.Define("u_y_em", "u_pt.second[9]/1000*sin(u_phi.second[9])")
        df = df.Define("u_x_lcw", "u_pt.second[10]/1000*cos(u_phi.second[10])")
        df = df.Define("u_y_lcw", "u_pt.second[10]/1000*sin(u_phi.second[10])")
        df = df.Define("u_x_ml", "u_pt.second[11]/1000*cos(u_phi.second[11])")
        df = df.Define("u_y_ml", "u_pt.second[11]/1000*sin(u_phi.second[11])")

        df = df.Define("u_perp_truth", "(z_x*u_y_truth- z_y*u_x_truth)/std::max(z_r, 1e-9f)")
        df = df.Define("u_perp_em", "(z_x*u_y_em  - z_y*u_x_em )/std::max(z_r, 1e-9f)")
        df = df.Define("u_perp_lcw", "(z_x*u_y_lcw - z_y*u_x_lcw)/std::max(z_r, 1e-9f)")
        df = df.Define("u_perp_ml", "(z_x*u_y_ml  - z_y*u_x_ml )/std::max(z_r, 1e-9f)")

        df = df.Define("qT", "z_r")
        df = df.Define("q_x", "z_x")
        df = df.Define("q_y", "z_y")

        df = df.Define("qhat_x", "z_x/std::max(z_r, 1e-9f)")
        df = df.Define("qhat_y", "z_y/std::max(z_r, 1e-9f)")
        df = df.Define("qhat_perp_x", "-qhat_y")
        df = df.Define("qhat_perp_y", " qhat_x")

        df = df.Define("u_mag_truth", "sqrt(u_x_truth*u_x_truth + u_y_truth*u_y_truth)")
        df = df.Define("u_mag_em", "sqrt(u_x_em*u_x_em + u_y_em*u_y_em)")
        df = df.Define("u_mag_lcw", "sqrt(u_x_lcw*u_x_lcw + u_y_lcw*u_y_lcw)")
        df = df.Define("u_mag_ml", "sqrt(u_x_ml*u_x_ml + u_y_ml*u_y_ml)")

        df = df.Define("u_par_truth", "(u_x_truth*z_x   + u_y_truth*z_y )/std::max(z_r, 1e-9f)")
        df = df.Define("u_par_em", "(u_x_em*z_x   + u_y_em*z_y )/std::max(z_r, 1e-9f)")
        df = df.Define("u_par_lcw", "(u_x_lcw*z_x  + u_y_lcw*z_y)/std::max(z_r, 1e-9f)")
        df = df.Define("u_par_ml", "(u_x_ml*z_x   + u_y_ml*z_y )/std::max(z_r, 1e-9f)")

        df = df.Define("u_perp_truth_alt", "u_x_truth*qhat_perp_x   + u_y_truth*qhat_perp_y")
        df = df.Define("u_perp_em_alt", "u_x_em*qhat_perp_x   + u_y_em*qhat_perp_y")
        df = df.Define("u_perp_lcw_alt", "u_x_lcw*qhat_perp_x  + u_y_lcw*qhat_perp_y")
        df = df.Define("u_perp_ml_alt", "u_x_ml*qhat_perp_x   + u_y_ml*qhat_perp_y")

        df = df.Define("u_par_plus_qT_truth", "u_par_truth  + qT")
        df = df.Define("u_par_plus_qT_em", "u_par_em  + qT")
        df = df.Define("u_par_plus_qT_lcw", "u_par_lcw + qT")
        df = df.Define("u_par_plus_qT_ml", "u_par_ml  + qT")

        df = df.Define("lep_pt", "mu_pt/1000")
        df = df.Define("lep_eta", "mu_eta")
        df = df.Define("lep_phi", "mu_phi")

        df = df.Define("clus_e_truth", "fCluster_truthE/1000")
        df = df.Define("clus_e_em", "fCluster_rawE/1000")
        df = df.Define("clus_e_lcw", "fCluster_calE/1000")
        df = df.Define("clus_e_ml", "fCluster_MLE/1000")
        df = df.Define("clus_eta", "fClusterEta")
        df = df.Define("clus_phi", "fClusterPhi")

        df = df.Define("met_truth_particle_pt", "fMETTruthPt/1000")
        df = df.Define("met_truth_particle_eta", "fMETTruthEta")
        df = df.Define("met_truth_particle_phi", "fMETTruthPhi")
        df = df.Define("met_truth_particle_e", "fMETTruthE/1000")

        self.df = df
        return df


def build_threshold_scan_observables(lep_pt, lep_eta, lep_phi, clus_e, clus_eta, clus_phi, thresholds):
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
            clus_phi_sel,
        )

        out[thr] = {
            "u_x": hadrecoil["u_x"],
            "u_y": hadrecoil["u_y"],
            "u_pt": hadrecoil["u_pt"],
        }

    return out


def build_truth_particle_threshold_scan_observables(truth_particle_pt, truth_particle_eta, truth_particle_phi, thresholds):
    out = {}

    truth_particle_pt = np.asarray(truth_particle_pt)
    truth_particle_eta = np.asarray(truth_particle_eta)
    truth_particle_phi = np.asarray(truth_particle_phi)

    for thr in thresholds:
        hadrecoil = hadrecoil_from_truth_particles(
            truth_particle_pt,
            truth_particle_eta,
            truth_particle_phi,
            threshold=thr,
        )

        out[thr] = {
            "u_x": hadrecoil["u_x"],
            "u_y": hadrecoil["u_y"],
            "u_pt": hadrecoil["u_pt"],
        }

    return out


def default_study2_x_bin_edges():
    return np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64)
