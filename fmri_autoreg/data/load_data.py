from typing import Dict, List, Tuple, Union
from tqdm.auto import tqdm
import logging

from pathlib import Path
import os
import re
import numpy as np
import h5py
import json
import pickle as pk
import torch
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler


def load_params(params):
    """Load parameters from json file or json string."""
    if isinstance(params, Path):
        params = str(params)
    if os.path.splitext(params)[1] == ".json":
        with open(params) as json_file:
            param_dict = json.load(json_file)
    else:
        param_dict = json.loads(params)
    return param_dict


def load_data(
    path: Union[Path, str],
    h5dset_path: Union[List[str], str],
    standardize: bool = False,
    dtype: str = "data",
) -> List[Union[np.ndarray, str, int, float]]:
    """Load time series or phenotype data from the hdf5 files.

    Args:
        path (Union[Path, str]): Path to the hdf5 file.
        h5dset_path (Union[List[str], str]): Path to data inside the
            h5 file.
        standardize (bool, optional): Whether to standardize the data.
            Defaults to False. Only applicable to dtype='data'.
        dtype (str, optional): Attribute label for each subject or
            "data" to load the time series. Defaults to "data".

    Returns:
        List[Union[np.ndarray, str, int, float]]: loaded data.
    """
    if isinstance(h5dset_path, str):
        h5dset_path = [h5dset_path]
    data_list = []
    if dtype == "data":
        with h5py.File(path, "r") as h5file:
            for p in h5dset_path:
                data_list.append(h5file[p][:])
        if standardize and data_list:
            means = np.concatenate(data_list, axis=0).mean(axis=0)
            stds = np.concatenate(data_list, axis=0).std(axis=0)
            data_list = [(data - means) / stds for data in data_list]
        return data_list
    else:
        with h5py.File(path, "r") as h5file:
            for p in h5dset_path:
                subject_node = "/".join(p.split("/")[:-1])
                data_list.append(h5file[subject_node].attrs[dtype])
        return data_list


def load_h5_data_path(
    path: Union[Path, str],
    data_filter: Union[str, None] = None,
    shuffle: bool = False,
    random_state: int = 42,
) -> List[str]:
    """Load dataset path data from HDF5 file.

    Args:
      path (str): path to the HDF5 file
      data_filter (str): regular expression to apply on run names
        (default=None)
      shuffle (bool): whether to shuffle the data (default=False)

    Returns:
      (list of str): HDF5 path to data
    """
    data_list = []
    with h5py.File(path, "r") as h5file:
        for dset in _traverse_datasets(h5file):
            if data_filter is None or re.search(data_filter, dset):
                data_list.append(dset)
    if shuffle and data_list:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(data_list)
    return data_list


def _traverse_datasets(hdf_file):
    """Load nested hdf5 files.
    https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
    """  # ruff: noqa: W505
    def h5py_dataset_iterator(g, prefix=""):
        for key in g.keys():
            item = g[key]
            path = f"{prefix}/{key}"
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)
    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


def make_input_labels(
    data_file,
    dset_paths,
    params,
    output_file_path=None,
    compute_edges=False,
    log=logging
):
    """Generate pairs of inputs and labels from time series.

    Args:
      tng_data (list of numpy arrays): training data
      val_data (list of numpy arrays): validation data
      seq_length (int): length of input sequences
      time_stride (int): stride of the sliding window
      lag (int): time points difference between the end of the input sequence and the time point used for label
      compute_edges (bool): wether to compute a connectivity graph (for graph convolutions)
      thres (float): threshold used for the connectivity graph, e.g. 0.9 means that only the 10%
        strongest edges are kept (default=0.9)

    Returns:
      (tuple): tuple containing:
        X_tng (numpy array): training input
        Y_tng (numpy array): training labels
        X_val (numpy array): validation input
        Y_val (numpy array): validation labels, validation input, validation labels,
        edge_index (tuple of numpy arrays): edges of the connectivity matrix (None if compute_edge_index is False)
    """
    # create connectome from data set
    n_parcels = params["n_embed"]
    if compute_edges:
        edges = get_edge_index(
            data_file=data_file,
            dset_paths=dset_paths,
        )
        log.info("Graph created")
    else:
        edges = None

    if output_file_path is None:
        output_file_path = "data.h5"
    log.info(f"Saving label and input to {output_file_path}.")
    for dset in tqdm(dset_paths):
        data = load_data(
            path=data_file,
            h5dset_path=dset,
            standardize=False,
            dtype="data"
        )
        x, y = make_seq(
            data,
            params["seq_length"],
            params["time_stride"],
            params["lag"]
        )
        if x.shape[0] == 0 or x is None:
            log.warning(
                f"Skipping {dset} as label couldn't be created."
            )
            continue
        with h5py.File(output_file_path, "a") as h5file:
            if h5file.get("input") is None:
                h5file.create_dataset(
                    name="input",
                    data=x,
                    dtype=np.float32,
                    maxshape=(None, n_parcels, params["seq_length"]),
                    chunks=(x.shape[0], n_parcels, params["seq_length"])
                )
                h5file.create_dataset(
                    name="label",
                    data=y,
                    dtype=np.float32,
                    maxshape=(None, n_parcels),
                    chunks=(y.shape[0], n_parcels)
                )
            h5file["input"].resize((h5file["input"].shape[0] + x.shape[0]), axis=0)
            h5file["input"][-x.shape[0]:] = x
            h5file["label"].resize((h5file["label"].shape[0] + y.shape[0]), axis=0)
            h5file["label"][-y.shape[0]:] = y
            input_size = h5file["input"].shape
    return output_file_path, edges, input_size


def make_seq(data_list, length, stride=1, lag=1):
    """Slice a list of timeseries with sliding windows and get corresponding labels.

    For each data in data list, pairs genreated will correspond to :
    `data[k:k+length]` for the sliding window and `data[k+length+lag-1]` for the label, with k
    iterating with the stride value.

    Args:
      data_list (list of numy arrays): list of data, data must be of shape (time_steps, features)
      length (int): length of the sliding window
      stride (int): stride of the sliding window (default=1)
      lag (int): time step difference between last time step of sliding window and label time step (default=1)

    Returns:
      (tuple): a tuple containing:
        X_tot (numpy array): sliding windows array of shape (nb of sequences, features, length)
        Y_tot (numpy array): labels array of shape (nb of sequences, features)
    """
    X_tot = []
    Y_tot = []
    delta = lag - 1
    for data in data_list:
        X = []
        Y = []
        for i in range(0, data.shape[0] - length - delta, stride):
            X.append(np.moveaxis(data[i : i + length], 0, 1))
            Y.append(data[i + length + delta])
        X_tot.append(np.array(X))
        Y_tot.append(np.array(Y))
    if len(X_tot) > 0:
        X_tot = np.concatenate(X_tot)
        Y_tot = np.concatenate(Y_tot)
        return X_tot, Y_tot
    return None, None


def get_edge_index(data_file, dset_paths):
    """Create connectivity matrix with more memory efficient way.

    Args:
      data_file: path to datafile
      dset_path (list of str): path to time series data

    Returns:
      (numpy array): connectivity matrix
    """
    connectome_measure = ConnectivityMeasure(kind="correlation", discard_diagonal=True)
    avg_corr_mats = None
    for dset in tqdm(dset_paths):
        data = load_data(
            path=data_file,
            h5dset_path=dset,
            standardize=False,
            dtype="data"
        )
        corr_mat = connectome_measure.fit_transform(data)[0]
        if avg_corr_mats is None:
            avg_corr_mats = corr_mat
        else:
            avg_corr_mats += corr_mat
        del data
        del corr_mat
    avg_corr_mats /= len(dset_paths)
    return avg_corr_mats


def get_edge_index_threshold(avg_corr_mats, threshold=0.9):
    thres_index = int(avg_corr_mats.shape[0] * avg_corr_mats.shape[1] * threshold)
    thres_value = np.sort(avg_corr_mats.flatten())[thres_index]
    adj_mat = avg_corr_mats * (avg_corr_mats >= thres_value)
    del avg_corr_mats
    del thres_value
    edge_index = np.nonzero(adj_mat)
    del adj_mat
    return edge_index


class Dataset:
    """Simple dataset for pytorch training loop"""

    def __init__(self, data_file):
        self.data_file = data_file

    def __len__(self):
        with h5py.File(self.data_file, "r") as h5file:
            length = h5file["label"].shape[0]
        return length

    def __getitem__(self, index):
        # read the data
        with h5py.File(self.data_file, "r") as h5file:
            X = h5file["input"][index, :, :]
            Y = h5file["label"][index, :]
        sample = {
            "input": torch.tensor(X, dtype=torch.float32),
            "label": torch.tensor(Y, dtype=torch.float32)
        }
        del X
        del Y
        return sample
