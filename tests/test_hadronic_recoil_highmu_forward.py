"""Numerical checks for recoil signs, jet weights, acceptance and binning.

ROOT is stubbed only when unavailable; these tests do not exercise ROOT output.
Run with: python -m unittest discover -s RDFAnalysisFramework/tests
"""
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import ROOT
except ModuleNotFoundError:
    ROOT = types.ModuleType("ROOT")
    ROOT.gROOT = types.SimpleNamespace(SetBatch=lambda flag: None)
    sys.modules["ROOT"] = ROOT

study = importlib.import_module("rdf_analysis.analyses.hadronic_recoil_study_highmu_forward")
from rdf_analysis.analyses.met_inputs_common import (
    ClusterCollection, LeptonCollection, METInputs, METInputTerm, TransverseVector,
)


def settings():
    return types.SimpleNamespace(eta_min=2.5, eta_max=4.9, jet_pt_min=30.0,
        jet_radius=0.4, cluster_threshold=0.0, threshold_on="momentum",
        lepton_cone=0.2, random_cone_distance=0.4, min_avg_mu=0.0)


def event_data(count):
    vectors = {
        "mu_pt": [40000.], "mu_eta": [0.], "mu_phi": [0.],
        "met_input_el_pt": [0.], "met_input_el_phi": [0.], "met_input_el_weight": [1.],
        "met_input_mu_pt": [40000.], "met_input_mu_phi": [0.], "met_input_mu_weight": [1.],
        "met_input_jet_pt": [40000.], "met_input_jet_eta": [3.],
        "met_input_jet_phi": [np.pi], "met_input_jet_weight": [0.5],
        "tu_pt.second": [0., 0., 0., 20000.], "tu_phi.second": [0., 0., 0., np.pi],
        "fClusterEta": [3.], "fClusterPhi": [np.pi],
    }
    for scale, (_, branch) in study.METConfig.CLUSTER_SCALES.items():
        vectors[branch] = [20 * np.cosh(3.) / study.METConfig.CLUSTER_SCALE_FACTORS[scale]]
    data = {name: np.tile(values, (count, 1)) for name, values in vectors.items()}
    for name, value in {"tboson_pt": 20000., "tboson_eta": 0., "tboson_phi": 0.,
                        "avgMu": 50., "nPrimVtx": 20., "met_softtrk_mpx": 0., "met_softtrk_mpy": 0.}.items():
        data[name] = np.full(count, value)
    return data


class ForwardRecoilTests(unittest.TestCase):
    def setUp(self):
        self.args = settings()
        self.leptons = LeptonCollection(np.array([40.]), np.array([0.]), np.array([0.]))
        self.met = METInputs(
            METInputTerm([], [], []), METInputTerm([40.], [0.], [1.]),
            METInputTerm([40.], [np.pi], [0.5], eta=[3.]), TransverseVector.zero(),
        )
        eta = np.array([3., 3., 1.])
        phi = np.array([np.pi, 0., 0.])
        energy = np.array([30., 5., 7.]) * np.cosh(eta)
        self.clusters = ClusterCollection(eta, phi, {s: energy.copy() for s in study.METConfig.CLUSTER_SCALES})

    def test_signs_weights_and_no_double_counting(self):
        recoils, lepton_sum, count = study.calculate_event_recoils(self.met, self.leptons, self.clusters, self.args)
        self.assertEqual(count, 1)
        self.assertEqual(lepton_sum.x, 40.)
        self.assertAlmostEqual(recoils['standard_met'].x, -20.)
        for scale in study.METConfig.CLUSTER_SCALES:
            self.assertAlmostEqual(recoils[f'central_clusters_{scale}'].x, 7.)
            self.assertAlmostEqual(recoils[f'all_clusters_{scale}'].x, -18.)
            self.assertAlmostEqual(recoils[f'keep_jets_forward_clusters_{scale}'].x, -15.)
            self.assertAlmostEqual(recoils[f'replace_forward_jets_with_all_forward_clusters_{scale}'].x, -25.)

    def test_no_jets_and_strict_pt_cut(self):
        for pt in ([], [30.]):
            self.met.jets = METInputTerm(pt, [np.pi] * len(pt), [1.] * len(pt), eta=[3.] * len(pt))
            recoils, _, count = study.calculate_event_recoils(self.met, self.leptons, self.clusters, self.args)
            self.assertEqual(count, 0)
            for scale in study.METConfig.CLUSTER_SCALES:
                a = recoils[f'keep_jets_forward_clusters_{scale}']
                b = recoils[f'replace_forward_jets_with_all_forward_clusters_{scale}']
                np.testing.assert_allclose([a.x, a.y], [b.x, b.y])

    def test_eta_boundaries_and_threshold_units(self):
        eta = np.array([2.5, 4.9])
        energy = np.array([2., 100.]) * np.cosh(eta)
        clusters = ClusterCollection(eta, np.zeros(2), {'em': energy})
        mask = clusters.in_abs_eta_range(2.5, 4.9)
        self.args.cluster_threshold = 3.
        self.args.threshold_on = 'pt'
        self.assertAlmostEqual(study.cluster_recoil(self.leptons, clusters, 'em', mask, self.args).pt, 0.)
        self.args.threshold_on = 'momentum'
        self.assertAlmostEqual(study.cluster_recoil(self.leptons, clusters, 'em', mask, self.args).pt, 2.)

    def test_helper_lepton_cone_removal(self):
        clusters = ClusterCollection(np.array([0.]), np.array([0.]), {'em': np.array([10.])})
        u = study.cluster_recoil(self.leptons, clusters, 'em', np.array([True]), self.args)
        np.testing.assert_allclose([u.x, u.y], [0., 0.], atol=1e-12)

    def test_ideal_response_and_iqr_definition(self):
        response, entries = study.calculate_binned_response(np.ones(10), np.full(10, 20.), np.full(10, -20.), [0, 1])
        np.testing.assert_allclose(response, [1.])
        np.testing.assert_array_equal(entries, [10])
        _, width = study.calculate_sigma_iqr_in_x_bins(np.ones(10), np.arange(10.), [0, 1])
        self.assertAlmostEqual(width[0] / 2, 3.06)
        response, _ = study.calculate_binned_response(np.ones(9), np.full(9, 20.), np.full(9, -20.), [0, 1])
        self.assertTrue(np.isnan(response[0]))

    def test_real_reader_global_limit_and_unit_conversions(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [str(Path(directory) / f'{i}.root') for i in range(2)]
            for path in paths:
                with uproot.recreate(path) as output:
                    output['treeAnaWZ'] = event_data(3)
            # Use the local-file source to avoid the environment's fsspec reader stall.
            original_open = uproot.open
            def open_local(*args, **kwargs):
                kwargs['handler'] = uproot.source.file.MultithreadedFileSource
                return original_open(*args, **kwargs)
            with patch.object(study.uproot, 'open', open_local):
                chunks = study.read_input_files(paths, 'treeAnaWZ', 4)
                events, values, counts = study.calculate_recoil_performance(chunks, self.args)
                self.assertEqual(counts['total'], 4)
                self.assertEqual(counts['selected'], 4)
                self.assertEqual(len(values), 21)
                for scale in study.METConfig.CLUSTER_SCALES:
                    np.testing.assert_allclose(values[f'all_clusters_{scale}']['balance'], 0., atol=1e-12)
                np.testing.assert_allclose(values['standard_met']['balance'], 0., atol=1e-12)
                np.testing.assert_allclose(values['standard_met']['met_residual_x'], 0., atol=1e-12)
                self.assertEqual(len(events['qT']), 4)
                chunks = study.read_input_files(paths, 'treeAnaWZ', 0)
                events, _, counts = study.calculate_recoil_performance(chunks, self.args)
                self.assertEqual(counts['total'], 0)
                self.assertEqual(len(events['qT']), 0)

    def test_pileup_selection_and_common_sample(self):
        data = event_data(2)
        data['avgMu'][0] = 20.
        self.args.min_avg_mu = 40.
        events, values, counts = study.calculate_recoil_performance([data], self.args)
        self.assertEqual(counts['selected'], 1)
        self.assertTrue(all(len(v['balance']) == 1 for v in values.values()))
        np.testing.assert_array_equal(events['avgMu'], [50.])


if __name__ == '__main__':
    unittest.main()
