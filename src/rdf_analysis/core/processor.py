import pandas as pd
import ROOT

from rdf_analysis.stats.fits import calculate_gaussian_fit_params_from_arrays


class NtupleProcessorRDF:
    """Thin RDataFrame wrapper shared by analysis scripts."""

    def __init__(self, input_file, tree_name, n_events):
        self.input_file = input_file
        self.tree_name = tree_name
        self.n_events = n_events
        self.df = ROOT.RDataFrame(tree_name, input_file)
        if n_events is not None and n_events > 0:
            self.df = self.df.Range(n_events)

    def build_dataframe(self):
        return self.df

    def materialize(self, columns):
        return self.df.AsNumpy(list(columns))

    def to_numpy(self, columns):
        return self.materialize(columns)

    def to_pandas(self, columns):
        return pd.DataFrame(self.materialize(columns))

    def get_column(self, column):
        return self.materialize([column])[column]

    def calculate_gaussian_fit_params(self, x_column, y_column, x_bin_edges, out_file=None, y_name=None):
        arrays = self.materialize([x_column, y_column])
        return calculate_gaussian_fit_params_from_arrays(
            arrays[x_column],
            arrays[y_column],
            x_bin_edges,
            out_file=out_file,
            y_name=y_name or y_column,
        )

