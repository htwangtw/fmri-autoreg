import h5py
import numpy as np
import os
import argparse
import logging
import json
import pickle as pk
import csv
from math import ceil
from sklearn.metrics import r2_score
from fmri_autoreg.data.load_data import load_data, Dataset
from fmri_autoreg.models.make_model import make_model
from fmri_autoreg.models.predict_model import predict_model
from fmri_autoreg.tools import check_path
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available


def train(params, data, verbose=1, logger=logging):
    """Train a model according to params dict.

    Args:
      params (dict): paramter dictionary
      data (tuple): tuple containing the training and validation h5 dataset path and edge index
        tuple
      base_dir (str): path to a directory to prepend to data file paths in parameters dict (default=None)
      verbose (int): level of verbosity (default=1)

    Returns:
      (tuple): tuple containing:
        model: trained model
        r2_tng (numpy array): training r2 score
        r2_val (numpy array): validation r2 score
        Z_tng (numpy array): training prediction
        Y_tng (numpy array): training label
        Z_val (numpy array): validation prediction
        Y_val (numpy array): validation label
        losses (numpy array): losses
        checkpoints (dict): scores and mean losses at checkpoint epochs
    """
    tng_data_h5, val_data_h5, edge_index = data  # unpack data

    # make model
    if verbose > 1:
        logger.info("Making model.")
    model, train_model = make_model(params, edge_index)
    if verbose > 1:
        logger.info("Creating dataloader.")
    tng_dataset = Dataset(tng_data_h5)
    val_dataset = Dataset(val_data_h5)
    tng_dataloader = DataLoader(
        tng_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params["num_workers"],
        pin_memory=cuda_is_available()
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params["num_workers"],
        pin_memory=cuda_is_available()
    )
    # train model
    model, losses, checkpoints = train_model(
        model=model,
        params=params,
        tng_dataloader=tng_dataloader,
        val_dataloader=val_dataloader,
        verbose=verbose,
        logger=logger
    )

    # compute r2 score
    r2_mean = {}
    for name, dset in zip(["tng", "val"], [tng_data_h5, val_data_h5]):
        r2 = predict_model(
            model=model,
            params=params,
            data_h5=dset,
        )
        r2_mean[name] = np.mean(r2)
    return model, r2_mean['tng'], r2_mean['val'], losses, checkpoints
