import os
from os.path import join
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import dump, load


def get_vae_lstm_loss(lstm_dataset, vae_model, lstm_model):
    reconst_loss = np.zeros(len(lstm_dataset))
    for i in tqdm(range(len(lstm_dataset))):
        data = lstm_dataset[i]
        vae_embeddings = vae_model.encoder_predict(data)
        vae_embeddings = np.expand_dims(vae_embeddings[:data.shape[0]-1], 0)
        lstm_embeddings = lstm_model(vae_embeddings, training=False)
        reconst = vae_model.decoder_predict(np.squeeze(lstm_embeddings))
        reconst_loss[i] = np.sum(np.square(data[1:] - reconst))
    return reconst_loss


def evaluate_model(config, data, train_loss, test_loss, all_loss,
                   threshold_dict, ref=""):
    for method, threshold in threshold_dict.items():
        threshold_info = {method: threshold}

        loss_df_dict = {}

        train_result_df = data.train_df.iloc[-len(train_loss):, :]
        train_result_df = train_result_df.assign(**threshold_info)
        train_result_df = train_result_df.assign(loss=train_loss)
        train_result_df = train_result_df.assign(
            pred_anomaly=train_result_df['loss'] > threshold)
        train_result_df.to_csv(join(config['EXP_DIR'], f"{ref}_train_result.csv"),
                               index_label="datetime")
        loss_df_dict.update({'train': train_result_df})

        test_result_df = data.test_df.iloc[-len(test_loss):, :]
        test_result_df = test_result_df.assign(**threshold_info)
        test_result_df = test_result_df.assign(loss=test_loss)
        test_result_df = test_result_df.assign(
            pred_anomaly=test_result_df['loss'] > threshold)
        test_result_df.to_csv(join(config['EXP_DIR'], f"{ref}_test_result.csv"),
                              index_label="datetime")
        loss_df_dict.update({'test': test_result_df})

        result_df = data.rhr.iloc[-len(all_loss):, :]
        result_df = result_df.assign(**threshold_info)
        result_df = result_df.assign(loss=all_loss)
        result_df = result_df.assign(
            pred_anomaly=result_df['loss'] > threshold)
        result_df.to_csv(join(config['EXP_DIR'], f"{ref}_all_result.csv"),
                         index_label="datetime")
        loss_df_dict.update({'all': result_df})

        # Evaluate metrics
        metrics = {}
        metrics['ID'] = data.id
        metrics['Threshold'] = method

        detection_mask = (result_df.index > data.date_dict['before_7']) & \
                         (result_df.index <= data.date_dict['after_21']) & \
                         (result_df['pred_anomaly'] == True)

        temp = get_metrics(result_df, data.date_dict)

        if temp['TP'] > 0:
            if result_df[detection_mask].index[0] < data.date_dict['onset']:
                metrics['Detection'] = 'Early'
            else:
                metrics['Detection'] = 'Late'
        else:
            metrics['Detection'] = 'Failed'

        metrics.update(temp)

        # Export metrics
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(
            join(config['EXP_DIR'], f"{ref}_metrics.csv"), index=False)

        # Export loss dict
        dump(loss_df_dict, join(
            config['EXP_DIR'], f"{ref}_loss_dict_df.joblib"))

        return metrics, loss_df_dict


def get_metrics_old(result_df, date_dict):

    anomalies = result_df[result_df['pred_anomaly'] == True]
    normals = result_df[result_df['pred_anomaly'] == False]

    tp_mask = (anomalies.index > date_dict['before_7']) & \
              (anomalies.index <= date_dict['after_21'])

    fp_mask = (anomalies.index <= date_dict['before_7']) | \
              (anomalies.index > date_dict['after_21'])

    fn_mask = (normals.index > date_dict['before_7']) & \
              (normals.index <= date_dict['after_21'])

    tn_mask = (normals.index <= date_dict['before_7']) | \
              (normals.index > date_dict['after_21'])

    tp = anomalies[tp_mask].shape[0]
    fp = anomalies[fp_mask].shape[0]
    fn = normals[fn_mask].shape[0]
    tn = normals[tn_mask].shape[0]

    Sensitivity = (tp / (tp+fn)) if (tp+fn) != 0 else 0
    Specificity = (tn / (tn+fp)) if (tn+fp) != 0 else 0
    PPV = (tp / (tp+fp)) if (tp+fp) != 0 else 0
    NPV = (tn / (tn+fn)) if (tn+fn) != 0 else 0
    Precision = (tp / (tp+fp)) if (tp+fp) != 0 else 0
    Recall = (tp / (tp+fn)) if (tp+fn) != 0 else 0
    if (Precision != 0 and Recall != 0):
        F1 = 2 * (((tp / (tp+fp)) * (tp / (tp+fn))) /
                  ((tp / (tp+fp)) + (tp / (tp+fn))))
        Fbeta = ((1+0.1**2) * ((tp / (tp+fp)) * (tp / (tp+fn)))) / \
            ((0.1**2) * (tp / (tp+fp)) + (tp / (tp+fn)))
        # F1 = 2 * ((Precision * Recall) / (Precision + Recall))
        # Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)
    else:
        F1 = 0
        Fbeta = 0

    metrics = {
        'Total': tp+fp+fn+tn,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'PPV': PPV,
        'NPV': NPV,
        'Precision': Precision,
        'Recall': Recall,
        'Fbeta': Fbeta,
        'F1': F1
    }

    return metrics


def get_metrics(result_df, date_dict):

    drop_mask = ((result_df.index >= date_dict['before_20']) &
                 (result_df.index < date_dict['onset'])) | \
                ((result_df.index > date_dict['after_14']) &
                 (result_df.index <= date_dict['after_21']))

    result_df = result_df.drop(result_df[drop_mask].index)
    result_df = result_df.drop(result_df.between_time("7:00", "22:00").index)

    anomalies = result_df[result_df['pred_anomaly'] == True]
    normals = result_df[result_df['pred_anomaly'] == False]

    tp_mask = (anomalies.index > date_dict['onset']) & \
              (anomalies.index <= date_dict['after_14'])

    fp_mask = (anomalies.index <= date_dict['onset']) | \
              (anomalies.index > date_dict['after_14'])

    fn_mask = (normals.index > date_dict['onset']) & \
              (normals.index <= date_dict['after_14'])

    tn_mask = (normals.index <= date_dict['onset']) | \
              (normals.index > date_dict['after_14'])

    tp = anomalies[tp_mask].shape[0]
    fp = anomalies[fp_mask].shape[0]
    fn = normals[fn_mask].shape[0]
    tn = normals[tn_mask].shape[0]

    Sensitivity = (tp / (tp+fn)) if (tp+fn) != 0 else 0
    Specificity = (tn / (tn+fp)) if (tn+fp) != 0 else 0
    PPV = (tp / (tp+fp)) if (tp+fp) != 0 else 0
    NPV = (tn / (tn+fn)) if (tn+fn) != 0 else 0
    Precision = (tp / (tp+fp)) if (tp+fp) != 0 else 0
    Recall = (tp / (tp+fn)) if (tp+fn) != 0 else 0
    if (Precision != 0 and Recall != 0):
        F1 = 2 * (((tp / (tp+fp)) * (tp / (tp+fn))) /
                  ((tp / (tp+fp)) + (tp / (tp+fn))))
        Fbeta = ((1+0.1**2) * ((tp / (tp+fp)) * (tp / (tp+fn)))) / \
            ((0.1**2) * (tp / (tp+fp)) + (tp / (tp+fn)))
        # F1 = 2 * ((Precision * Recall) / (Precision + Recall))
        # Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)
    else:
        F1 = 0
        Fbeta = 0

    metrics = {
        'Total': tp+fp+fn+tn,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'PPV': PPV,
        'NPV': NPV,
        'Precision': Precision,
        'Recall': Recall,
        'Fbeta': Fbeta,
        'F1': F1
    }

    return metrics


def compile_metrics(subject_info, exp_dir, ref):
    # Compile metrics
    df = pd.DataFrame()
    for idx in range(len(subject_info['ID'])):
        metrics_dir = join(
            exp_dir, subject_info['ID'][idx] + f"_{ref}_metrics.csv")
        if not os.path.isfile(metrics_dir):
            continue
        temp = pd.read_csv(metrics_dir)
        temp.insert(0, 'ID', subject_info['ID'][idx])
        df = pd.concat([df, temp], ignore_index=True)

    # Process metrics
    tp = df['TP'].sum()
    fp = df['FP'].sum()
    fn = df['FN'].sum()
    tn = df['TN'].sum()

    Sensitivity = (tp / (tp+fn)) if (tp+fn) != 0 else 0
    Specificity = (tn / (tn+fp)) if (tn+fp) != 0 else 0
    PPV = (tp / (tp+fp)) if (tp+fp) != 0 else 0
    NPV = (tn / (tn+fn)) if (tn+fn) != 0 else 0
    Precision = (tp / (tp+fp)) if (tp+fp) != 0 else 0
    Recall = (tp / (tp+fn)) if (tp+fn) != 0 else 0
    if (Precision != 0 and Recall != 0):
        # F1 = 2 * ((Precision * Recall) / (Precision + Recall))
        F1 = 2 * (((tp / (tp+fp)) * (tp / (tp+fn))) /
                  ((tp / (tp+fp)) + (tp / (tp+fn))))
        # Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)
        Fbeta = ((1+0.1**2) * ((tp / (tp+fp)) * (tp / (tp+fn)))) / \
            ((0.1**2) * (tp / (tp+fp)) + (tp / (tp+fn)))
    else:
        F1 = 0
        Fbeta = 0

    n_early = df['Detection'].value_counts()['Early']
    n_late = df['Detection'].value_counts()['Late']
    n_failed = df['Detection'].value_counts()['Failed']

    detection = f"{n_early/len(df)*100:0.2f}% ({n_early}) :" +\
                f"{n_late/len(df)*100:0.2f}% ({n_late}) :" +\
                f"{n_failed/len(df)*100:0.2f}% ({n_failed})"

    metrics = {
        'ID': ref + "_Summary",
        'Threshold': df['Threshold'][0],
        'Detection': detection,
        'Total': tp+fp+fn+tn,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'PPV': PPV,
        'NPV': NPV,
        'Precision': Precision,
        'Recall': Recall,
        'Fbeta': Fbeta,
        'F1': F1
    }
    metrics_df = pd.DataFrame(metrics, index=[0])

    return pd.concat([df, metrics_df], ignore_index=True, axis=0)


def summarize_metrics(df, exp_dir, ref):

    # Process metrics
    tp = df['TP'].sum()
    fp = df['FP'].sum()
    fn = df['FN'].sum()
    tn = df['TN'].sum()

    Sensitivity = (tp / (tp+fn)) if (tp+fn) != 0 else 0
    Specificity = (tn / (tn+fp)) if (tn+fp) != 0 else 0
    PPV = (tp / (tp+fp)) if (tp+fp) != 0 else 0
    NPV = (tn / (tn+fn)) if (tn+fn) != 0 else 0
    Precision = (tp / (tp+fp)) if (tp+fp) != 0 else 0
    Recall = (tp / (tp+fn)) if (tp+fn) != 0 else 0
    if (Precision != 0 and Recall != 0):
        # F1 = 2 * ((Precision * Recall) / (Precision + Recall))
        F1 = 2 * (((tp / (tp+fp)) * (tp / (tp+fn))) /
                  ((tp / (tp+fp)) + (tp / (tp+fn))))
        # Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)
        Fbeta = ((1+0.1**2) * ((tp / (tp+fp)) * (tp / (tp+fn)))) / \
            ((0.1**2) * (tp / (tp+fp)) + (tp / (tp+fn)))
    else:
        F1 = 0
        Fbeta = 0

    n_early = df['Detection'].value_counts()['Early']
    n_late = df['Detection'].value_counts()['Late']
    n_failed = df['Detection'].value_counts()['Failed']

    detection = f"{n_early/len(df)*100:0.2f}% ({n_early}) :" +\
                f"{n_late/len(df)*100:0.2f}% ({n_late}) :" +\
                f"{n_failed/len(df)*100:0.2f}% ({n_failed})"

    metrics = {
        'ID': ref + "_Summary",
        'Threshold': df['Threshold'][0],
        'Detection': detection,
        'Total': tp+fp+fn+tn,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'PPV': PPV,
        'NPV': NPV,
        'Precision': Precision,
        'Recall': Recall,
        'Fbeta': Fbeta,
        'F1': F1
    }
    metrics_df = pd.DataFrame(metrics, index=[0])

    return pd.concat([df, metrics_df], ignore_index=True, axis=0)
