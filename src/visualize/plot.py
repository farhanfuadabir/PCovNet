import matplotlib.pyplot as plt
import pandas as pd
from config.config import config

def loss_curve(history, ref="", save_plot=False, close_plot=False, annotate=False):

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
        ax.plot(history.index[min_id], history['val_loss'][min_id], marker="o", markersize=3, markerfacecolor="tab:gray")
        ax.axvline(history.index[min_id],
                color='tab:gray',
                label='Min Validation Loss',
                linestyle='dashed',
                lw=2,
                alpha=0.8)

    ax.tick_params(axis='both', labelsize=13)
    plt.title(f"Loss Curve", fontsize=18, pad=45)
    plt.legend(bbox_to_anchor=(0, 1.01, 1, 0.1), fontsize=13,
                loc="lower center", borderaxespad=0, ncol=3)
    plt.ylabel('Loss', fontsize=15, labelpad=15)
    plt.xlabel('Epoch', fontsize=15, labelpad=15)
    
    if save_plot == True:
        plt.tight_layout()
        fig.savefig(f"{config['EXP_DIR']}/plots/{ref}_loss_curve.pdf")
    
    if close_plot == True:
        plt.close()
    else:
        plt.show()