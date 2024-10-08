import numpy as np
import warnings
import h5py
import os
import argparse
import pickle as pk
from math import ceil
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available
from sklearn.metrics import r2_score
from fmri_autoreg.data.load_data import load_params, make_seq, Dataset


def predict_model(model, params, dataset):
    """Use trained model to predict on data and compute R2 score."""
    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params["num_workers"],
        pin_memory=cuda_is_available()
    )
    r2 = []
    for sampled_batch in dataloader:
        x = sampled_batch["input"].to(params["torch_device"])
        y = sampled_batch["label"].to(params["torch_device"])
        z = model.predict(x)
        y = y.detach().cpu().numpy()
        batch_r2 = r2_score(y, z, multioutput="raw_values")
        r2.append(batch_r2)
    r2 = np.concatenate(r2, axis=0)
    return r2


def predict_horizon(
    model, seq_length, horizon, data_file, dset_path, batch_size, stride=1
):
    """For models trained to predict t+1, reuse predictions to iteratively predict to t+horizon."""
    with h5py.File(data_file, "r") as h5file:
        data_list = [h5file[dset_path][:]]
    X, _ = make_seq(data_list, length=seq_length + horizon, stride=stride, lag=0)
    del data_list
    if not len(X):
        warnings.warn(f"No data found in {data_file} for {dset_path}", RuntimeWarning)
        return (None,) * 3
    Z = []
    Y = X[:, :, seq_length:]
    if batch_size is None:
        batch_size = X.shape[0]
        n_batch = 1
    else:
        n_batch = (X.shape[0] - 1) // batch_size + 1

    for i_batch in range(n_batch):
        x_batch = X[i_batch * batch_size : (i_batch + 1) * batch_size, :, :seq_length]
        z_batch = []
        for _ in range(1, horizon + 1):
            z_batch.append(np.expand_dims(model.predict(x_batch), axis=-1))
            x_batch = np.concatenate((x_batch[:, :, 1:], z_batch[-1]), axis=-1)
        Z.append(np.concatenate(z_batch, axis=-1))

    Z = np.concatenate(Z, axis=0)

    r2 = []
    for i in range(horizon):
        r2.append(r2_score(Y[:, :, i], Z[:, :, i], multioutput="raw_values"))

    return np.array(r2), Z, Y

