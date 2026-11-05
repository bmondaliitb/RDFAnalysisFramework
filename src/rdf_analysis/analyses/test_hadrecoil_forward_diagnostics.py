"""ROOT-independent synthetic checks. Run with Python + NumPy via unittest.

Extract actual analysis functions using AST to avoid importing the ROOT runtime.
Histogram doubles check names, bin values/errors and invalid-bin preservation;
they do not substitute for a real ROOT serialization integration test.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

SOURCE = Path(__file__).with_name('hadrecoil_study_highmu_forward.py')
tree = ast.parse(SOURCE.read_text())
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
         or isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets)]
written = {}

class Hist:
    def __init__(self, name, contents):
        self.name, self.contents = name, np.nan_to_num(contents).copy()
        self.errors = np.zeros(len(contents))
    def SetBinContent(self, i, value):
        if 1 <= i <= len(self.contents): self.contents[i-1] = value
    def SetBinError(self, i, value): self.errors[i-1] = value
    def Write(self):
        if self.name in written: raise AssertionError('Duplicate name: ' + self.name)
        written[self.name] = self
    def GetXaxis(self): return self
    def SetTitle(self, title): pass

ns = dict(METConfig=SimpleNamespace(GEV=0.001), np=np, Mapping=dict, SimpleNamespace=SimpleNamespace, MIN_MEAN_ENTRIES=10,
          make_hist_from_bin_contents=lambda name, title, edges, values: Hist(name, np.asarray(values)),
          make_numpy_hist=lambda name, title, edges, values: Hist(name, np.histogram(values, edges)[0]))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
# Exercise the actual project's IQR utility without importing its ROOT fits.
fits_source = SOURCE.parent.parent / "stats" / "fits.py"
fit_node = next(n for n in ast.parse(fits_source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "calculate_sigma_iqr_in_x_bins")
exec(compile(ast.Module(body=[fit_node], type_ignores=[]), str(fits_source), 'exec'), ns)

class Hist2D:
    def __init__(self, name, title, nx, ex, ny, ey):
        self.name = name
        self.contents = np.zeros((nx+2, ny+2))
        self.errors = np.zeros_like(self.contents)
    def SetBinContent(self, i, j, value): self.contents[i, j] = value
    def SetBinError(self, i, j, value): self.errors[i, j] = value
    def SetEntries(self, value): self.entries = value
    def Write(self):
        if self.name in written: raise AssertionError('Duplicate name: ' + self.name)
        written[self.name] = self


class DiagnosticsTest(unittest.TestCase):
    def test_mean_edges_finite_sem_and_empty(self):
        mean, error, count = ns['calculate_binned_mean'](
            [0, .5, 1, 2, np.nan], [1, 3, 5, 7, 100], [0, 1, 2, 3], 2)
        np.testing.assert_equal(count, [2, 1, 1])
        self.assertEqual(mean[0], 2)
        self.assertEqual(error[0], 1)
        self.assertTrue(np.isnan(mean[1:]).all())
        mean, _, count = ns['calculate_binned_mean']([0, 1], [1, 3], [0, 1], 2)
        self.assertEqual(count[0], 2)
        self.assertEqual(mean[0], 2)
        with self.assertRaises(ValueError): ns['calculate_binned_mean']([1], [], [0, 1])

    def test_flatness_offset(self):
        f, offset = ns['angular_flatness'](np.array([2., 4.]), np.array([10, 30]), [True, True])
        self.assertAlmostEqual(offset, 3.5)
        self.assertAlmostEqual(f, np.sqrt(.75))
        self.assertTrue(np.isnan(ns['angular_flatness'](np.array([2.]), np.array([10]), [True])[0]))

    def test_activity_thresholds(self):
        clusters = SimpleNamespace(energy={'em': np.array([10., 20., -1., np.nan])},
            eta=np.array([3., 3., 3., 3.]), phi=np.zeros(4),
            in_abs_eta_range=lambda low, high: np.ones(4, dtype=bool))
        args = ns['default_settings']()
        args.cluster_threshold = 15
        total, count = ns['forward_activity'](clusters, args)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(total, 20 / np.cosh(3))
        args.threshold_on = 'pt'
        self.assertEqual(ns['forward_activity'](clusters, args), (0., 0))

    def test_zero_qt(self):
        obj = object.__new__(ns['HadRecoil'])
        data = {'tboson_pt': [0.], 'tboson_phi': [0.7],
                'tu_pt.second': [[1000.]], 'tu_phi.second': [[.2]],
                'u_pt.second': [np.ones(20)*1000], 'u_phi.second': [np.ones(20)*.2]}
        with np.errstate(divide='raise', invalid='raise'):
            self.assertTrue(all(np.isfinite(x) for x in obj.calculate_recoil(data, 0).values()))

    def test_histograms_and_paired_sign(self):
        written.clear()
        events = {key: np.ones(20) for key in ('qT', 'avgMu', 'nPrimVtx',
                  'forward_sum_pt_em', 'n_forward_clusters_em', 'n_forward_jets')}
        events['deltaPhi_W_mu'] = np.repeat([-.5, .5], 10)
        values = {'standard_met': {'balance': np.full(20, 4.), 'u_perp': np.full(20, 2.)},
                  'test': {'balance': np.full(20, 1.), 'u_perp': np.full(20, 1.)}}
        ns['save_bias_diagnostics'](events, values)
        self.assertEqual(len(written), 97)  # 15 means * 3 companions + 2 scalars per scenario + 3 shared
        h = written['h_mean_paired_delta_abs_balance_vs_qT_test']
        self.assertEqual(h.contents[0], -3)
        self.assertEqual(h.errors[0], 0)
        self.assertTrue(np.isnan(h.contents[1:]).all())
        self.assertEqual(written['h_angular_flatness_u_perp_test'].contents[0], 0)
        self.assertEqual(written['h_angular_offset_u_perp_test'].contents[0], 1)
        values['test']['balance'] = np.ones(19)
        with self.assertRaises(ValueError): ns['save_bias_diagnostics'](events, values)

class TruthMassTest(unittest.TestCase):
    def truth_event(self, singleton=False):
        values = dict(tlep_bare_pt=40000., tlep_bare_phi=0., tlep_bare_pdgId=13,
                      tneutrino_pt=40000., tneutrino_phi=np.pi, tneutrino_pdgId=-14)
        return {key: [[value]] if singleton else [value] for key, value in values.items()}

    def test_truth_mass_units_shapes_and_charge(self):
        for singleton in (False, True):
            data = self.truth_event(singleton)
            self.assertAlmostEqual(ns['truth_transverse_mass'](data, 0), 80.)
        data = self.truth_event()
        data['tlep_bare_pdgId'], data['tneutrino_pdgId'] = [-13], [14]
        self.assertAlmostEqual(ns['truth_transverse_mass'](data, 0), 80.)
        data['tneutrino_phi'] = [2*np.pi]
        self.assertAlmostEqual(ns['truth_transverse_mass'](data, 0), 0.)
        data['tlep_bare_pt'] = [0.]
        self.assertEqual(ns['truth_transverse_mass'](data, 0), 0.)

    def test_invalid_truth_and_missing_branch(self):
        for key, value in [('tlep_bare_pt', -1), ('tneutrino_phi', np.nan),
                           ('tlep_bare_pdgId', 11), ('tneutrino_pdgId', 14),
                           ('tneutrino_pdgId', -12), ('tlep_bare_pt', []),
                           ('tlep_bare_pt', [40000, 50000])]:
            data = self.truth_event()
            data[key] = [value]
            self.assertTrue(np.isnan(ns['truth_transverse_mass'](data, 0)), key)
        data = self.truth_event()
        del data['tneutrino_pt']
        with self.assertRaises(KeyError): ns['truth_transverse_mass'](data, 0)

    def test_migration_flows_and_empty_rows(self):
        counts, norm, totals = ns['mass_migration'](
            [5, 5, 5, 20, np.nan], [5, 15, 30, 5, 5], [0, 10, 20])
        self.assertEqual(counts.sum(), 4)
        np.testing.assert_allclose(norm[1, 1:], [1/3, 1/3, 1/3])
        self.assertEqual(totals[3], 1)  # final edge overflows
        self.assertTrue(np.isnan(norm[2]).all())
        np.testing.assert_allclose(norm[totals > 0].sum(axis=1), 1)
        with self.assertRaises(ValueError): ns['mass_migration']([1], [], [0, 10])

    def test_mass_histograms(self):
        written.clear()
        truth = np.r_[np.full(20, 80.), np.nan]
        events = dict(mT_truth=truth, qT=np.ones(21), nPrimVtx=np.ones(21))
        values = {name: dict(mT=np.full(21, mt), mT_residual=np.full(21, mt)-truth)
                  for name, mt in [('standard_met', 90.), ('improved', 82.)]}
        original = ns['make_numpy_hist']
        ns['make_numpy_hist'] = lambda name, title, *args: Hist(name, np.histogram(
            args[-1], args[0] if len(args) == 2 else np.linspace(args[1], args[2], args[0]+1))[0])
        try:
            with patch.dict('sys.modules', {'ROOT': SimpleNamespace(TH2D=Hist2D)}):
                ns['save_mass_diagnostics'](events, values)
        finally:
            ns['make_numpy_hist'] = original
        self.assertEqual(len(written), 39)
        self.assertEqual(written['h_mT_truth'].contents.sum(), 20)
        self.assertEqual(written['h_mean_mT_residual_vs_qT_improved'].contents[0], 2)
        self.assertEqual(written['h_mean_paired_delta_abs_mT_residual_vs_qT_improved'].contents[0], -8)
        self.assertEqual(written['h_sigma_mT_residual_vs_qT_improved'].contents[0], 0)
        self.assertTrue(np.isnan(written['h_sigma_mT_residual_vs_qT_improved'].contents[1:]).all())
        self.assertEqual(written['h_mT_truth_vs_reco_counts_improved'].contents.sum(), 20)
        self.assertEqual(written['h_mT_truth_vs_reco_normalized_improved'].contents[17, 17], 1)
        # A nonzero residual spread verifies the q84-q16 convention and /2.
        written.clear()
        residual = np.r_[np.arange(20, dtype=float), np.nan]
        values['improved'] = dict(mT=truth+residual, mT_residual=residual)
        ns['make_numpy_hist'] = lambda name, title, *args: Hist(name, np.histogram(
            args[-1], args[0] if len(args) == 2 else np.linspace(args[1], args[2], args[0]+1))[0])
        try:
            with patch.dict('sys.modules', {'ROOT': SimpleNamespace(TH2D=Hist2D)}):
                ns['save_mass_diagnostics'](events, values)
        finally:
            ns['make_numpy_hist'] = original
        self.assertAlmostEqual(written['h_sigma_mT_residual_vs_qT_improved'].contents[0], 6.46)

if __name__ == '__main__': unittest.main()
