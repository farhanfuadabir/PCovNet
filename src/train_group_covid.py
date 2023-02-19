#####################################################################
# Dependencies
#####################################################################

# Default packages
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
import os
from os.path import join
from datetime import date, datetime, timedelta
from joblib import load, dump

# Set current directory to "src"
os.chdir(join(os.getcwd(), os.pardir, "src"))
print(f"Current working directory: {os.getcwd()}")

# Installed packages

# Defined modules

# Set TF log level to minimum
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Check GPU availability
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


#####################################################################
# Process Config
#####################################################################


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


#####################################################################
# Import and Process Data
#####################################################################


# Choose which data index from subject_info to run the experiment on
for idx in range(0, len(subject_info)):
    # Prepare data
    data = Data(config=config, data_info=subject_info, index=idx)

    # Log data properties
    if data.error == True:
        print("\n", data.error_message, "\n")
        with open(join(config['EXP_DIR'], "log.txt"), 'a', encoding='utf-8') as f:
            f.write(
                f"Index {idx}\t {data.id} {'-'*4} Data properties... ✘ ... {data.error_message}\n")
        # Use continue if run in a loop; otherwise raise error
        continue
        # raise ValueError("Data does not meet experiment criteria!")
    elif data.error_message != "":
        print("\n", data.error_message, "\n")
        with open(join(config['EXP_DIR'], "log.txt"), 'a', encoding='utf-8') as f:
            f.write(
                f"Index {idx}\t {data.id} {'-'*4} Data properties... ! ... {data.error_message}\n")
    else:
        print("\nData properties... ✔\n")
        with open(join(config['EXP_DIR'], "log.txt"), 'a', encoding='utf-8') as f:
            f.write(f"Index {idx}\t {data.id} {'-'*4} Data properties... ✔\n")

    # Print data info
    data.print_info()

    # Export data object
    dump(data, join(config['EXP_DIR'], f"{data.id}_data.joblib"))

    # Export dates
    pd.DataFrame(data.date_dict, index=[0]).to_csv(
        join(config['EXP_DIR'], data.id + "_dates.csv"), index=False)

    #####################################################################
    # VAE Model
    #####################################################################

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
    tl_vae_ckpt_path = join(os.pardir, "experiment",
                            f"trained_model_healthy_phase1_{config['LEN_WIN']}-{config['N_WIN']}",
                            "vae_checkpoint", "ckpt")
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
    # Load learned weights for VAE
    if os.path.isfile(vae_ckpt_path + ".index"):
        vae_model.load_weights(vae_ckpt_path)
        print(f"\nVAE model weights loaded ... ✔\n")
    else:
        vae_model.load_weights(tl_vae_ckpt_path)
        print(f"\nVAE model weights loaded ... ✔\n")

        if config['TRAIN_VAE'] and len(data.train_dataset_vae) != 0:
            print("\nTraining VAE model ... \n")
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

        # Calculate vae losses
        print("\n")
        print("Calculate vae train loss... ", end="")
        vae_train_loss = vae_model.get_loss_array(data.train_dataset_vae)
        print("✓")
        print("Calculate vae test loss... ", end="")
        vae_test_loss = vae_model.get_loss_array(data.test_dataset_vae)
        print("✓")
        print("Calculate vae all loss... ", end="")
        vae_seq_loss = vae_model.get_loss_array(data.merged_dataset_vae)
        print("✓")
        print("\n")

        vae_loss_dict = {
            'train_loss': vae_train_loss,
            'test_loss': vae_test_loss,
            'all_loss': vae_seq_loss
        }

        dump(vae_loss_dict, join(
            config['EXP_DIR'], f"{data.id}_VAE_loss_dict.joblib"))

    #####################################################################
    # Generate Embeddings
    #####################################################################

    if config['TRAIN_LSTM'] and len(data.train_dataset_lstm) != 0:
        print("\nGenerating embeddings for LSTM Model ... \n")

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

    #####################################################################
    # LSTM Model
    #####################################################################

    # Get LSTM MODEL
    lstm_model = lstm_autoencoder(n_timesteps=config['N_WIN'] - 1,
                                  n_features=config['LATENT_DIM'])
    lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                       optimizer=tf.optimizers.Adam(
        learning_rate=config['LEARNING_RATE']),
        metrics=['mse'])

    # Show LSTM model summary
    # print("\nLSTM Model Summary")
    # print("==================", end="\n\n")
    # lstm_model.summary()

    # Assign checkpoint paths
    tl_lstm_ckpt_path = join(os.pardir, "experiment",
                             f"trained_model_healthy_phase1_{config['LEN_WIN']}-{config['N_WIN']}",
                             "lstm_checkpoint", "ckpt")
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
    # Load learned weights for VAE
    if os.path.isfile(lstm_ckpt_path + ".index"):
        lstm_model.load_weights(lstm_ckpt_path)
        print(f"\nVAE model weights loaded ... ✔\n")
    else:
        lstm_model.load_weights(tl_lstm_ckpt_path)
        print(f"\nLSTM model weights loaded ... ✔\n")

        if config['TRAIN_LSTM'] and len(data.train_dataset_lstm) != 0:

            print("\nTraining LSTM model ... \n")

            # Train LSTM
            lstm_history = lstm_model.fit(embed_gen.x_train, embed_gen.y_train,
                                          validation_split=config['VAL_SPLIT'],
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

    if len(data.train_dataset_lstm) != 0:
        # Calculate vae-lstm losses
        print("\n")
        print("Calculate lstm train loss... ", end="")
        lstm_train_loss = get_vae_lstm_loss(data.train_dataset_lstm,
                                            vae_model, lstm_model)
        print("Calculate lstm test loss... ", end="")
        lstm_test_loss = get_vae_lstm_loss(data.test_dataset_lstm,
                                           vae_model, lstm_model)
        print("Calculate lstm all loss... ", end="")
        lstm_seq_loss = get_vae_lstm_loss(data.merged_dataset_lstm,
                                          vae_model, lstm_model)
        print("\n")

        lstm_loss_dict = {
            'train_loss': lstm_train_loss,
            'test_loss': lstm_test_loss,
            'all_loss': lstm_seq_loss
        }

        dump(lstm_loss_dict, join(
            config['EXP_DIR'], f"{data.id}_LSTM_loss_dict.joblib"))

    print(f"\n\nCompleted training for {data.index} - {data.id} ... ✔\n\n")
