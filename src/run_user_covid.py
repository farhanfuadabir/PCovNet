from visualize import plot
from models.lstm import basic_lstm
from models.lstm_ae import lstm_autoencoder
from models import lstm_vae, cnn_vae
from process.evaluate import *
from process.embed_gen import EmbedGen
from process.dataloader import Data
from config.config import *
from util.util import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.backend import clear_session
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from joblib import load, dump
from datetime import date, datetime, timedelta
from os.path import join
import os
idx = int(input("Input User Index: "))

##########################################################################


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

#########################################################################
#########################################################################

# Dictionaries of dataset paths
DATA_DIR_DICT = {
    'phase1': join(os.getcwd(), os.pardir, "data", "raw", "phase1"),
    'phase2': join(os.getcwd(), os.pardir, "data", "raw", "phase2")
}
INFO_DIR_DICT = {
    'covid_phase1': join(os.getcwd(), os.pardir, "data", "external", "covid_phase1_info.csv"),
    'covid_phase2': join(os.getcwd(), os.pardir, "data", "external", "covid_phase2_info.csv"),
    'healthy_phase1': join(os.getcwd(), os.pardir, "data", "external", "healthy_phase1_info.csv"),
    'non-covid_phase1': join(os.getcwd(), os.pardir, "data", "external", "non-covid_phase1_info.csv"),
}

# Import subject info
subject_info = pd.read_csv(
    INFO_DIR_DICT[f"{config['EXP_GROUP']}_{config['EXP_PHASE']}"])


# Assign experiment directory
config['EXP_DIR'] = join(os.getcwd(), os.pardir, "experiment",
                         config['EXP_NAME'])
# timestamp = datetime.now().strftime(f"%Y-%m-%d %H-%M__")
# config['EXP_DIR'] = join(os.getcwd(), os.pardir, "experiment",
#                          timestamp + config['EXP_NAME'])
handle_dir(config['EXP_DIR'])


# Add DATA_DIR
config['DATA_DIR'] = DATA_DIR_DICT[config['EXP_PHASE']]


# Export config
export_json(config, join(config['EXP_DIR'], "config.json"),
            print_json=True)

# Start logging
with open(join(config['EXP_DIR'], "log.txt"), 'w', encoding='utf-8') as f:
    for key, value in config.items():
        f.write(f"{key}: {value}\n")
    f.write("\n\n")


print(f"\nIndex: {idx}")
print("===========", end="\n")

# Prepare data
data = Data(config=config, data_info=subject_info, index=idx)

# Log data properties
if data.error == True:
    print(data.error_message)
    with open(join(config['EXP_DIR'], "log.txt"), 'a', encoding='utf-8') as f:
        f.write(
            f"Index {idx}\t {data.id} {'-'*4} Data properties... ✘ ... {data.error_message}\n")
    # continue
    raise ValueError("Data does not meet experiment criteria!")
else:
    print("Data properties... ✔")
    with open(join(config['EXP_DIR'], "log.txt"), 'a', encoding='utf-8') as f:
        f.write(f"Index {idx}\t {data.id} {'-'*4} Data properties... ✔\n")

data.print_info()

# Get VAE model
vae_model = cnn_vae.VAE(n_timesteps=config['LEN_WIN'],
                        n_channels=data.train_dataset_vae.shape[-1],
                        latent_dim=config['LATENT_DIM'])
vae_model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(
    learning_rate=config['LEARNING_RATE']),
    metrics=[tf.metrics.MeanSquaredError()])

# Show VAE model summary
# print("\nVAE Model Summary")
# print("=================", end="\n\n")
# vae_model.print_summary()

# Assign checkpoint paths
vae_ckpt_path = join(
    config['EXP_DIR'], data.id + "_vae_checkpoint", "ckpt")

# Callbacks for VAE
early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        patience=config['PATIENCE'],
                                        mode='min',
                                        restore_best_weights=True)
checkpoint_callback = ModelCheckpoint(vae_ckpt_path,
                                      monitor='val_loss',
                                      verbose=0,
                                      mode='min',
                                      save_best_only=True,
                                      save_weights_only=True)

if not os.path.isfile(vae_ckpt_path + ".index"):
    # Train VAE
    vae_history = vae_model.fit(data.train_aug_dataset_vae,
                                validation_split=config['VAL_SPLIT'],
                                batch_size=config['BATCH_SIZE'],
                                epochs=config['EPOCH'],
                                shuffle=False,
                                verbose=1,
                                callbacks=[early_stopping_callback, checkpoint_callback])

    # Export model history
    export_history(vae_history, join(
        config['EXP_DIR'], data.id + "_vae_history.csv"))

    # Plot loss curve
    # print("\nVAE Loss Curve")
    # print("==============", end="\n")
    plot.loss_curve(config, vae_history, ref=data.id + "_VAE", save_plot=True,
                    close_plot=True)

else:
    vae_model.load_weights(vae_ckpt_path)
    print("VAE model weights loaded from:")
    print(vae_ckpt_path)

if not os.path.isfile(join(config['EXP_DIR'], f"{data.id}_embeddings.joblib")):
    # Get embedding dataset
    embed_gen = EmbedGen(config, vae_model,
                         data, verbose=True)

    # Save embed_gen object
    dump(embed_gen, join(config['EXP_DIR'],
                         f"{data.id}_embeddings.joblib"))
else:
    # Load embed_gen object
    embed_gen = load(
        join(config['EXP_DIR'], f"{data.id}_embeddings.joblib"))
    print("Embeddings loaded from:")
    print(join(config['EXP_DIR'], f"{data.id}_embeddings.joblib"))

# Get LSTM MODEL
lstm_model = lstm_autoencoder(n_timesteps=config['N_WIN'] - 1,
                              n_features=config['LATENT_DIM'])
lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                   optimizer=tf.optimizers.Adam(
    learning_rate=config['LEARNING_RATE']),
    metrics=['mse'])

# Show VAE model summary
# print("\nLSTM Model Summary")
# print("==================", end="\n\n")
# lstm_model.summary()

# Assign checkpoint paths
lstm_ckpt_path = join(
    config['EXP_DIR'], data.id + "_lstm_checkpoint", "ckpt")

# Callbacks for LSTM
early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        patience=config['PATIENCE'],
                                        mode='min',
                                        restore_best_weights=True)
checkpoint_callback = ModelCheckpoint(lstm_ckpt_path,
                                      monitor='val_loss',
                                      verbose=0,
                                      mode='min',
                                      save_best_only=True,
                                      save_weights_only=True)

if not os.path.isfile(lstm_ckpt_path + ".index"):
    # Train LSTM
    lstm_history = lstm_model.fit(embed_gen.x_train, embed_gen.y_train,
                                  validation_data=(
                                      embed_gen.x_test, embed_gen.y_test),
                                  batch_size=config['BATCH_SIZE'],
                                  epochs=config['EPOCH'],
                                  callbacks=[
                                      early_stopping_callback, checkpoint_callback],
                                  verbose=1)

    # Export model history
    export_history(lstm_history, join(
        config['EXP_DIR'], data.id + "_lstm_history.csv"))

    # Plot loss curve
    # print("\nLSTM Loss Curve")
    # print("===============", end="\n")
    plot.loss_curve(config, lstm_history, ref=data.id + "_LSTM", save_plot=True,
                    close_plot=True)
else:
    lstm_model.load_weights(lstm_ckpt_path)
    print("LSTM model weights loaded from:")
    print(lstm_ckpt_path)

# Calculate vae losses
print("\n")
print("Calculate vae train loss... ", end="")
train_loss = vae_model.get_loss_array(data.train_dataset_vae)
print("✓")
print("Calculate vae test loss... ", end="")
test_loss = vae_model.get_loss_array(data.test_dataset_vae)
print("✓")
print("Calculate vae all loss... ", end="")
seq_loss = vae_model.get_loss_array(data.merged_dataset_vae)
print("✓")
print("\n")

threshold_dict = {
    'MTE': train_loss.max(),
    # 'STE': train_loss.mean() + (3 * train_loss.std())
}

# print("\nLoss Distribution Plot")
# print("======================", end="\n")
plot.loss_dist(config, train_loss, test_loss, threshold_dict, ref=f"{data.id}_VAE",
               save_plot=True, close_plot=True)

metrics, loss_df_dict = evaluate_model(config, data, train_loss, test_loss,
                                       seq_loss, threshold_dict, ref=f"{data.id}_VAE")
print("\nMetrics")
print("=======", end="\n")
print(metrics)

# print("\nRHR Plot")
# print("========", end="\n")
# Plot RHR wrt infectious period
plot.rhr_plot(config, loss_df_dict['all'], data.date_dict, title=f"{data.id} - RHR Plot",
              ref=f"{data.id}_VAE", save_plot=True, close_plot=True)

# print("\nAnomaly Plot")
# print("=============", end="\n")
# Plot anomalies
plot.anomaly_plot(config, loss_df_dict['all'], data.date_dict, threshold_dict['MTE'],
                  metrics, title=f"{data.id} - Anomaly Plot", ref=f"{data.id}_VAE",
                  save_plot=True, close_plot=True)

# Calculate vae-lstm losses
print("\n")
print("Calculate lstm train loss... ", end="")
train_loss = get_vae_lstm_loss(data.train_dataset_lstm,
                               vae_model, lstm_model)
print("Calculate lstm test loss... ", end="")
test_loss = get_vae_lstm_loss(data.test_dataset_lstm,
                              vae_model, lstm_model)
print("Calculate lstm all loss... ", end="")
seq_loss = get_vae_lstm_loss(data.merged_dataset_lstm,
                             vae_model, lstm_model)
print("\n")

threshold_dict = {
    'MTE': train_loss.max(),
    # 'STE': train_loss.mean() + (3 * train_loss.std())
}

# print("\nLoss Distribution Plot")
# print("======================", end="\n")
plot.loss_dist(config, train_loss, test_loss, threshold_dict, ref=f"{data.id}_LSTM",
               save_plot=True, close_plot=True)

metrics, loss_df_dict = evaluate_model(config, data, train_loss, test_loss,
                                       seq_loss, threshold_dict, ref=f"{data.id}_LSTM")
print("\nMetrics")
print("=======", end="\n")
print(metrics)

# print("\nRHR Plot")
# print("========", end="\n")
# Plot RHR wrt infectious period
plot.rhr_plot(config, loss_df_dict['all'], data.date_dict, title=f"{data.id} - RHR Plot",
              ref=f"{data.id}_LSTM", save_plot=True, close_plot=True)

# print("\nAnomaly Plot")
# print("=============", end="\n")
# Plot anomalies
plot.anomaly_plot(config, loss_df_dict['all'], data.date_dict, threshold_dict['MTE'],
                  metrics, title=f"{data.id} - Anomaly Plot", ref=f"{data.id}_LSTM",
                  save_plot=True, close_plot=True)
