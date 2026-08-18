import awkward as ak
import numpy as np
import uproot

from rdf_analysis.stats.fits import calculate_gaussian_fit_params_from_arrays, calculate_mean_sigma_in_x_bins


def awkward_to_numpy(values, dtype=np.float64):
    return ak.to_numpy(values).astype(dtype, copy=False)


def _to_numpy_if_regular(values):
    if isinstance(values, ak.Array):
        try:
            return ak.to_numpy(values)
        except (TypeError, ValueError):
            return values

    return np.asarray(values)


def _concatenate_chunks(chunks):
    if not chunks:
        return np.array([], dtype=np.float64)

    if any(isinstance(chunk, ak.Array) for chunk in chunks):
        return _to_numpy_if_regular(ak.concatenate(chunks))

    return np.concatenate([np.asarray(chunk) for chunk in chunks])


class NtupleProcessor:
    """uproot/awkward reader shared by analysis scripts."""

    def __init__(self, input_file, tree_name, n_events, step_size=10000):
        self.input_file = input_file if isinstance(input_file, (list, tuple)) else [input_file]
        self.tree_name = tree_name
        self.n_events = n_events
        self.step_size = step_size
        self.total_entries = self._count_entries()
        self.n_events_to_process = (
            self.total_entries
            if n_events is None or n_events < 0
            else min(n_events, self.total_entries)
        )

    def build_arrays(self):
        return self

    def _input_trees(self):
        return {input_file: self.tree_name for input_file in self.input_file}

    def _count_entries(self):
        total_entries = 0
        for input_file in self.input_file:
            with uproot.open(input_file) as root_file:
                total_entries += root_file[self.tree_name].num_entries
        return total_entries

    def branch_names(self):
        common_branches = None
        for input_file in self.input_file:
            with uproot.open(input_file) as root_file:
                branches = set(root_file[self.tree_name].keys())
            common_branches = branches if common_branches is None else common_branches & branches

        return common_branches or set()

    def _required_branches(self, columns):
        return sorted(set(columns))

    def _select_events(self, raw_arrays):
        return raw_arrays

    def iter_arrays(self, columns):
        """
        Stream requested columns from the input trees in bounded uproot/awkward chunks.

        The method reads only the requested raw branches, applies the processor
        event selection, then yields one dictionary per chunk keyed by the
        requested column names.

        retrun look like this:
        {
            "jet_eta": awkward_array_for_this_chunk,
            "cluster_e_truth": awkward_array_for_this_chunk,
            "cluster_e_ML_correct": awkward_array_for_this_chunk,
        }
        """
        if self.n_events_to_process <= 0:
            return

        required_branches = self._required_branches(columns)
        missing_branches = sorted(set(required_branches) - self.branch_names())
        if missing_branches:
            print(
                "Input tree is missing required branches: {}".format(
                    ", ".join(missing_branches)
                )
            )

        processed_events = 0
        step_size = min(self.step_size, self.n_events_to_process)

        for raw_arrays in uproot.iterate(
            self._input_trees(),
            filter_name=required_branches,
            step_size=step_size,
            library="ak",
            how=dict,
        ):
            if processed_events >= self.n_events_to_process:
                break

            chunk_events = len(raw_arrays[required_branches[0]])
            remaining_events = self.n_events_to_process - processed_events
            if chunk_events > remaining_events:
                raw_arrays = {key: value[:remaining_events] for key, value in raw_arrays.items()}
                chunk_events = remaining_events

            processed_events += chunk_events
            raw_arrays = self._select_events(raw_arrays)
            if not raw_arrays or len(next(iter(raw_arrays.values()))) == 0:
                continue

            yield {column: raw_arrays[column] for column in columns}

    def materialize(self, columns):
        buffers = {column: [] for column in columns}
        for arrays in self.iter_arrays(columns):
            for column in columns:
                buffers[column].append(arrays[column])

        return {
            column: _concatenate_chunks(chunks)
            for column, chunks in buffers.items()
        }

    def to_numpy(self, columns):
        return self.materialize(columns)

    def get_column(self, column):
        return self.materialize([column])[column]
