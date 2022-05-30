import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def loss_curve(config, history, ref="", save_plot=False, close_plot=False, annotate=False):

    plt.style.use('seaborn')
    plt.rcParams['font.family'] = "Arial"  # "Bahnschrift"
    plt.rcParams['figure.figsize'] = 8, 6

    history = pd.DataFrame(history.history)
    min_id = history['val_loss'].idxmin()
    fig, ax = plt.subplots()

    plt.style.use('seaborn')
    plt.rcParams['font.family'] = "Arial"

    ax.plot(history['loss'], lw=2, c='tab:blue', label='Train Loss')
    ax.plot(history['val_loss'], lw=2, c='tab:orange', label='Validation Loss')

    if annotate == True:
        ax.plot(history.index[min_id], history['val_loss'][min_id],
                marker="o", markersize=3, markerfacecolor="tab:gray")
        ax.axvline(history.index[min_id],
                   color='tab:gray',
                   label='Min Validation Loss',
                   linestyle='dashed',
                   lw=2,
                   alpha=0.8)

    ax.tick_params(axis='both', labelsize=13)
    plt.title(f"{ref} - Loss Curve", fontsize=18, pad=45)
    plt.legend(bbox_to_anchor=(0, 1.01, 1, 0.1), fontsize=13,
               loc="lower center", borderaxespad=0, ncol=3)
    plt.ylabel('Loss', fontsize=15, labelpad=15)
    plt.xlabel('Epoch', fontsize=15, labelpad=15)

    if save_plot == True:
        plt.tight_layout()
        fig.savefig(f"{config['EXP_DIR']}/{ref}_loss_curve.pdf")

    if close_plot == True:
        plt.close()
    else:
        plt.show()


def loss_dist(config, train_loss, test_loss, threshold_dict, ref="",
              save_plot=False, close_plot=False):
    plt.style.use('seaborn')
    plt.rcParams['font.family'] = "Arial"  # "Bahnschrift"
    plt.rcParams['figure.figsize'] = 8, 6

    fig, ax = plt.subplots()

    sns.kdeplot(np.squeeze(train_loss), fill=True,
                label='Train Loss',
                common_norm=False,
                alpha=0.7,
                color='tab:blue')
    sns.kdeplot(np.squeeze(test_loss),
                label='Test Loss',
                fill=True,
                common_norm=False,
                alpha=0.7,
                color='tab:orange')
    # ax = sns.distplot(train_loss, bins=50, kde=True, color='tab:blue')
    # ax = sns.distplot(test_loss, bins=50, kde=True, color='tab:red')
    if 'STE' in threshold_dict.keys():
        ax.axvline(threshold_dict['STE'],
                   color='tab:gray',
                   label='Threshold (STE)',
                   linestyle='solid',
                   lw=2,
                   alpha=1)
    if 'MTE' in threshold_dict.keys():
        ax.axvline(threshold_dict['MTE'],
                   color='tab:red',
                   label='Threshold (MTE)',
                   linestyle='dashed',
                   lw=1.5,
                   alpha=1)

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize=12,
               loc="lower center", borderaxespad=0, ncol=4)
    plt.tick_params(axis='both', labelsize=13)
    plt.title(f"{ref} - Loss Distribution", fontsize=18, pad=45)
    plt.ylabel('Density', fontsize=15, labelpad=15)
    plt.xlabel('Epoch', fontsize=15, labelpad=15)
    plt.tight_layout()

    if save_plot == True:
        plt.tight_layout()
        fig.savefig(f"{config['EXP_DIR']}/{ref}_loss_dist.pdf")

    if close_plot == True:
        plt.close()
    else:
        plt.show()


def anomaly_plot(config, result_df, date_dict, threshold, metrics, title,
                 ref="", save_plot=False, close_plot=False):

    plt.style.use('seaborn')
    plt.rcParams['font.family'] = "Arial"  # "Bahnschrift"
    plt.rcParams['figure.figsize'] = 15, 5

    result_df.index.name = 'datetime'

    fig, ax = plt.subplots()

    ax.axvspan(date_dict['onset'], date_dict['after_14'],
               color='#d90429',
               label='Infectious Period',
               alpha=0.2)

    ax.axvline(date_dict['onset'],
               color='#d90429',
               label='Symptom Onset',
               linestyle='solid',
               lw=2,
               alpha=0.8)

    plt.axhline(y=threshold,
                color='tab:purple',
                linestyle='solid',
                lw=2,
                alpha=0.8,
                label='Threshold')

    # plot anomaly scores
    col_name = f"pred_anomaly"
    normal = result_df.loc[result_df[col_name] == False].reset_index()
    anomaly = result_df.loc[result_df[col_name] == True].reset_index()

    ax.scatter(normal['datetime'], normal['loss'], s=20,
               c='#003566',
               label='Normal')
    ax.scatter(anomaly['datetime'], anomaly['loss'], s=20,
               c='#d90429',
               label='Anomaly')

    # Format x ticks
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))

    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')

    plt.xticks(rotation=90)

    # Plot info
    plt.title(title, fontsize=20, pad=60)
    # Add subtitle
    subtitle = f'TP: {metrics["TP"]}, FP: {metrics["FP"]}, TN: {metrics["TN"]}, ' + \
               f'FN: {metrics["FN"]}, Precision: {metrics["Precision"]:0.2f}, ' + \
               f'Recall: {metrics["Recall"]:0.2f}, F1: {metrics["F1"]:0.2f}, ' + \
               f'Fbeta: {metrics["Fbeta"]:0.2f}'
    plt.suptitle(subtitle, size=13, x=0.513,
                 y=0.835, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, labelpad=15)
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize=12,
               loc="lower center", borderaxespad=0, ncol=5)

    if save_plot == True:
        plt.tight_layout()
        fig.savefig(f"{config['EXP_DIR']}/{ref}_anomaly_plot.pdf",
                    bbox_inches='tight')

    if close_plot == True:
        plt.close()
    else:
        plt.show()


def rhr_plot(config, data_df, date_dict, title, ref="",
             save_plot=False, close_plot=False):

    plt.style.use('seaborn')
    plt.rcParams['font.family'] = "Arial"  # "Bahnschrift"
    plt.rcParams['figure.figsize'] = 15, 5

    data_df.index.name = 'datetime'
    data_df = data_df.reset_index()

    fig, ax = plt.subplots()

    # plot infectious period
    ax.axvspan(date_dict['before_7'], date_dict['after_21'],
               color='#d90429',
               label='Infectious Period',
               alpha=0.2)

    # normal_mask = (data_df['datetime'] <= date_dict['before_7']) | \
    #               (data_df['datetime'] > date_dict['after_21'])

    # anomaly_mask = (data_df['datetime'] > date_dict['before_7']) & \
    #                (data_df['datetime'] <= date_dict['after_21'])

    # # plot anomaly scores
    # normal = data_df.loc[normal_mask]
    # anomaly = data_df.loc[anomaly_mask]

    # ax.scatter(normal['datetime'], normal['RHR'], s=20,
    #            c='#003566',
    #            label='Normal RHR')
    # ax.scatter(anomaly['datetime'], anomaly['RHR'], s=20,
    #            c='#d90429',
    #            label='Anomalous RHR')

    ax.scatter(data_df['datetime'], data_df['RHR'], s=20,
               c='#003566',
               label='Resting Heart Rate')

    ax.axvline(date_dict['onset'],
               color='#d90429',
               label='Symptom Onset',
               linestyle='solid',
               lw=2,
               alpha=0.8)

    # Format x ticks
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))

    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')

    plt.xticks(rotation=90)

    # Plot info
    plt.title(title, fontsize=20, pad=38)
    plt.ylabel('Resting Heart Rate', fontsize=12, labelpad=15)
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize=12,
               loc="lower center", borderaxespad=0, ncol=5)
    # Set y-limit
    rhr_max = data_df['RHR'].max()
    rhr_min = data_df['RHR'].min()
    plt.ylim((rhr_min if rhr_min < 50 else 50,
              rhr_max if (rhr_max > 120) else 120))

    if save_plot == True:
        plt.tight_layout()
        fig.savefig(f"{config['EXP_DIR']}/{ref}_rhr_plot.pdf",
                    bbox_inches='tight')

    if close_plot == True:
        plt.close()
    else:
        plt.show()
