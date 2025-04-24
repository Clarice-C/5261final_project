#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 20:10:25 2025

@author: jiaotian
"""


import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

# -------------------------
# 1. Load and Preprocess Data
# -------------------------

# Load only the first two columns: Date and Close
data = pd.read_csv("/Users/jiaotian/Downloads/Data collected - Boeing stock price.csv", usecols=[0, 1])
data.columns = ["Date", "Close"]  # Remove any dot-based duplicates

# Parse dates and sort by time (handle errors)
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data = data.dropna(subset=["Date"])
data = data.sort_values("Date")

# Add required columns for TFT
data["time_idx"] = range(len(data))
data["group_id"] = "BOEING"  # one group
data["target"] = data["Close"]  # target variable to forecast

# -------------------------
# 2. Create TFT Dataset
# -------------------------

training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="target",
    group_ids=["group_id"],
    max_encoder_length=30,
    max_prediction_length=7,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["target"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

train_dataloader = training.to_dataloader(train=True, batch_size=64)

# -------------------------
# 3. Build TFT Model
# -------------------------

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
    log_val_interval=1,
)

# Convert to LightningModule compatible with latest PyTorch Lightning



# -------------------------
# 4. Train Model
# -------------------------

trainer = Trainer(
    max_epochs=30,
    accelerator="cpu"  # use "gpu" if running on a GPU-enabled system
)

trainer.fit(tft, train_dataloader)
# -------------------------
# 5. (Optional) Forecast + Interpret
# -------------------------
# test_dataloader = training.to_dataloader(train=False, batch_size=64)
# raw_predictions, x = tft_model.predict(test_dataloader, mode="raw", return_x=True)
# tft_model.interpret_output(raw_predictions, reduction="mean")