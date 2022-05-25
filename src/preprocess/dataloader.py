import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from preprocess.augmentation import augment_dataset
from config.config import subject_info


class Subject():
    def __init__(self, config, index):
        self.config = config
        self.index = index
        self.exp_phase = self.config['EXP_PHASE']
        self.id = subject_info['ID'][index]
        self.symptom_onset = pd.to_datetime(
            subject_info['Symptom Onset'][index])
        self.date_dict = {
            'onset': self.symptom_onset,
            'before_7': self.symptom_onset + timedelta(days=-7),
            'before_10': self.symptom_onset + timedelta(days=-10),
            'before_20': self.symptom_onset + timedelta(days=-20),
            'after_21': self.symptom_onset + timedelta(days=21),
        }

        if self.exp_phase == 'phase1':
            self.device = 'Fitbit'
        elif self.exp_phase == 'phase2':
            self.device = subject_info['Device'][index]

        self.hr_path = os.path.join(self.config['DATA_DIR'],
                                    self.id + '_hr.csv')
        self.steps_path = os.path.join(self.config['DATA_DIR'],
                                       self.id + '_steps.csv')

        self.__get_hr()
        self.__get_steps()
        self.__get_rhr()
        self.__apply_filter(len_win=1600)
        self.__annotate_anomaly()
        self.__split_data()

        self.date_dict['start'] = self.rhr.index[0]
        self.date_dict['end'] = self.rhr.index[-1]
        self.baseline_rhr = self.train_df['RHR'].mean()
        self.delta_rhr = self.rhr['RHR'] - self.baseline_rhr

        self.__standardize_data()
        self.train_dataset_vae = self.__create_vae_dataset(self.train_data,
                                                           self.config['LEN_WIN'])
        self.test_dataset_vae = self.__create_vae_dataset(self.test_data,
                                                          self.config['LEN_WIN'])
        self.merged_dataset_vae = self.__create_vae_dataset(self.merged_data,
                                                            self.config['LEN_WIN'])

        self.train_dataset_lstm = self.__create_lstm_dataset(self.train_data,
                                                             self.config['LEN_WIN'],
                                                             self.config['N_WIN'])
        self.test_dataset_lstm = self.__create_lstm_dataset(self.test_data,
                                                            self.config['LEN_WIN'],
                                                            self.config['N_WIN'])
        self.merged_dataset_lstm = self.__create_lstm_dataset(self.merged_data,
                                                              self.config['LEN_WIN'],
                                                              self.config['N_WIN'])

        if self.config['AUGMENT']:
            self.train_aug_dataset_vae = augment_dataset(
                self.train_dataset_vae)
            # self.train_aug_dataset_lstm = augment_dataset(self.train_dataset_lstm)

        self.n_vae_train = len(self.train_data) - self.config['LEN_WIN'] + 1
        self.n_vae_test = len(self.train_data) - self.config['LEN_WIN'] + 1
        self.n_vae_merged = len(self.train_data) - self.config['LEN_WIN'] + 1

    def __get_hr(self):
        if self.exp_phase == 'phase1':
            self.hr = pd.read_csv(self.hr_path).set_index('datetime')
            self.hr.index.name = None
            self.hr.index = pd.to_datetime(self.hr.index)

        elif self.exp_phase == 'phase2':
            self.hr = pd.read_csv(self.hr_path)
            self.hr['datetime'] = pd.to_datetime(
                self.hr['datetime'], errors='coerce')
            self.hr['datetime'] = self.hr['datetime'].apply(
                lambda t: t.replace(second=0))
            self.hr = self.hr.set_index('datetime')
            self.hr.index.name = None
            self.hr.drop(['Unnamed: 0'], axis=1, inplace=True)
            self.hr['heartrate'] = pd.to_numeric(
                self.hr['heartrate'], errors='coerce')
            self.hr = self.hr.dropna()

    def __get_steps(self):
        if self.exp_phase == 'phase1':
            self.steps = pd.read_csv(self.steps_path).set_index('datetime')
            self.steps.index.name = None
            self.steps.index = pd.to_datetime(self.steps.index)

        elif self.exp_phase == 'phase2':
            self.steps = pd.read_csv(self.steps_path)
            if self.device == 'AppleWatch':
                self.steps['start_datetime'] = pd.to_datetime(
                    self.steps['start_datetime'], errors='coerce')
                self.steps['datetime'] = self.steps['start_datetime']
            self.steps['datetime'] = pd.to_datetime(self.steps['datetime'])
            self.steps['datetime'] = self.steps['datetime'].apply(
                lambda t: t.replace(second=0))
            self.steps = self.steps.set_index('datetime')
            self.steps.index.name = None
            self.steps.drop(['Unnamed: 0'], axis=1, inplace=True)
            self.steps['steps'] = pd.to_numeric(
                self.steps['steps'], errors='coerce')
            self.steps = self.steps.dropna()

    def __get_rhr(self):
        self.hr = self.hr.resample('1min').mean()
        self.steps = self.steps.resample('1min').sum()
        merged = pd.merge(self.hr, self.steps,
                          left_index=True, right_index=True)
        merged = merged.resample('1min').mean()
        merged = merged.dropna()
        # define RHR as the HR measurements recorded when there were zero steps taken
        # during a rolling time window of the preceding 12 minutes (including the current minute).
        merged['steps_window_12'] = merged['steps'].rolling(12).sum()
        self.rhr_raw = merged.loc[(merged['steps_window_12'] == 0)]
        self.rhr_raw = self.rhr_raw.drop(['steps', 'steps_window_12'], axis=1)
        self.rhr_raw = self.rhr_raw.rename(columns={"heartrate": "RHR"})

    def __apply_filter(self, len_win=400):
        self.rhr = self.rhr_raw.dropna()
        self.rhr = self.rhr.rolling(len_win).mean()
        self.rhr = self.rhr.resample('1H').mean()
        self.rhr = self.rhr.dropna()

    def __annotate_anomaly(self):
        self.rhr['anomaly'] = (self.rhr.index > self.date_dict['before_7']) \
            & (self.rhr.index < self.date_dict['after_21'])

    def __split_data(self):
        self.train_df = self.rhr.loc[self.rhr.index <
                                     self.date_dict['before_20']]
        self.test_df = self.rhr.loc[self.rhr.index >=
                                    self.date_dict['before_20']]
        self.test_anomaly_df = self.rhr.loc[(self.rhr.index >= self.date_dict['before_7'])
                                            & (self.rhr.index < self.date_dict['after_21'])]
        self.test_normal_df = self.rhr.loc[(self.rhr.index >= self.date_dict['before_20'])
                                           & (self.rhr.index < self.date_dict['before_10'])]

    def __standardize_data(self):
        scaler = StandardScaler()

        self.train_data = self.train_df['RHR'].to_numpy()
        self.train_data = np.expand_dims(self.train_data, axis=-1)
        self.test_data = self.test_df['RHR'].to_numpy()
        self.test_data = np.expand_dims(self.test_data, axis=-1)

        self.train_data = scaler.fit_transform(self.train_data)
        self.test_data = scaler.transform(self.test_data)

        # self.test_anomaly_data = temp.loc[(self.test_data.index >= self.date_dict['before_7'])
        #                                   & (self.test_data.index < self.date_dict['after_21'])]
        # self.test_normal_data = temp.loc[(self.test_data.index >= self.date_dict['before_20'])
        #                                  & (self.test_data.index < self.date_dict['before_10'])]

        self.merged_data = np.concatenate(
            (self.train_data, self.test_data), axis=0)

    def __create_vae_dataset(self, data, len_win):
        dataset = []
        for i in range(len(data) - len_win + 1):
            dataset.append(data[i: i+len_win])
        return np.array(dataset)

    def __create_lstm_dataset(self, data, len_win, n_win):
        dataset = []
        for i in range(len(data) - (len_win * n_win) + 1):
            dataset.append(data[i: i+(len_win * n_win)])
        dataset = np.array(dataset)
        return dataset.reshape(dataset.shape[0], n_win, len_win, 1)

    def print_info(self):
        print(f"\
            Subject Info\n\
            ============\n\
            Index:                {self.index}\n\
            Phase:                {self.exp_phase}\n\
            Group:                {self.config['EXP_GROUP']}\n\
            ID:                   {self.id}\n\
            Device:               {self.device}\n\
            \n\
            Dataset Shape\n\
            =============\n\
            VAE Train:            {self.train_dataset_vae.shape}\n\
            VAE Train-aug:        {self.train_aug_dataset_vae.shape}\n\
            VAE Test:             {self.test_dataset_vae.shape}\n\
            VAE Merged:           {self.merged_dataset_vae.shape}\n\
            LSTM Train:           {self.train_dataset_lstm.shape}\n\
            LSTM Train:           {self.test_dataset_lstm.shape}\n\
            LSTM Merged:          {self.merged_dataset_lstm.shape}\n\
        ")
