import os
import json
import pandas as pd
import pickle


def handle_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def export_json(dict, path, print_json=False):
    with open(path, "w") as outfile:
        json.dump(dict, outfile, indent=4)
    if print_json:
        print("config")
        print(json.dumps(dict, indent=4))


def export_history(history_callback, path):
    df = pd.DataFrame.from_dict(history_callback.history)
    df.index.name = "epoch"
    df.to_csv(path)


def export_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def import_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
    # from joblib import load, dump

    # def splits(export_path, subject_id):
    # with open(export_path, 'w') as f:
    #     print("id,start_date,Day 0,Day -20,Day -7,Day -10,Day +21,end_date\n",
    #           f"{subject_id},{start},{symptom_date},{symptom_date_before_20},\
    #             {symptom_date_before_7},{symptom_date_before_10},\
    #             {symptom_date_after_21},{end}", file=f)
