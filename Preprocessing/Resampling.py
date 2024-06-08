import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

class Resampling:
    def __init__(self, dataset, test_size=0.4, random_state=None):
        self.dataset = dataset
        self.test_size = test_size
        self.random_state = random_state

    def custom_train_test_split(self):
        unique_users = self.dataset['cc_user'].unique()
        unique_merchants = self.dataset['merchant_num'].unique()

        train_users, test_users = train_test_split(unique_users, test_size=self.test_size, random_state=self.random_state)
        train_merchants, test_merchants = train_test_split(unique_merchants, test_size=self.test_size, random_state=self.random_state)

        train_mask = self.dataset['cc_user'].isin(train_users) & self.dataset['merchant_num'].isin(train_merchants)
        test_mask = self.dataset['cc_user'].isin(test_users) & self.dataset['merchant_num'].isin(test_merchants)

        assert train_mask.sum() + test_mask.sum() <= len(self.dataset), "Overlap detected between train and test sets"

        self.train_data = self.dataset.loc[train_mask].copy()
        self.test_data = self.dataset.loc[test_mask].copy()

        assert set(self.train_data['cc_user']).isdisjoint(set(self.test_data['cc_user'])), "Overlap detected in cc_user"
        assert set(self.train_data['merchant_num']).isdisjoint(set(self.test_data['merchant_num'])), "Overlap detected in merchant_num"

        return self.train_data, self.test_data

    def check_fraud_rate(self, data):
        length = len(data)
        fraud_rate = (data['is_fraud'].sum() / length) * 100
        return fraud_rate

    def resample_data(self):
        oversampler = RandomOverSampler(sampling_strategy='minority')
        undersampler = RandomUnderSampler(sampling_strategy='majority')

        X = self.train_data.drop('is_fraud', axis=1)
        y = self.train_data['is_fraud']

        X_resampled, y_resampled = oversampler.fit_resample(X, y)
        X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)

        resampled_df = X_resampled.copy()
        resampled_df['is_fraud'] = y_resampled
        resampled_df = resampled_df.sample(frac=1).reset_index(drop=True)

        self.train_data = resampled_df

        self.train_data.drop(columns=['cc_user', 'merchant_num'], inplace=True)
        self.train_data['cc_user'] = self.train_data['cc_num'].rank(method='dense') - 1
        self.train_data['merchant_num'] = self.train_data['merchant'].rank(method='dense') - 1
        self.train_data.drop(columns=['cc_num', 'merchant'], inplace=True)

        self.test_data = self.test_data.copy()
        self.test_data.drop(columns=['cc_user', 'merchant_num'], inplace=True, errors='ignore')
        self.test_data.loc[:, 'cc_user'] = self.test_data['cc_num'].rank(method='dense').astype(int) + 588
        self.test_data.loc[:, 'merchant_num'] = self.test_data['merchant'].rank(method='dense').astype(int) + 414
        self.test_data.drop(columns=['cc_num', 'merchant'], inplace=True)

        self.dataset = pd.concat([self.train_data, self.test_data], ignore_index=True)

        return self.dataset

    def normalize_data(self):
        exclude_columns = ['index', 'gender', 'is_fraud', 'cc_user', 'merchant_num']
        columns_to_normalize = [col for col in self.dataset.columns if col not in exclude_columns]

        scaler = MinMaxScaler()
        self.dataset[columns_to_normalize] = scaler.fit_transform(self.dataset[columns_to_normalize])

        return self.dataset
    
    def reset_index(self):
        self.dataset.drop(columns='index')
        self.dataset['index'] = range(len(self.dataset))

    def apply_resampling(self):
        self.custom_train_test_split()
        print('Fraud rate in training set before resampling: {:.2f}%'.format(self.check_fraud_rate(self.train_data)))
        print('Fraud rate in testing set: {:.2f}%'.format(self.check_fraud_rate(self.test_data)))
        self.resample_data()
        print('Fraud rate in training set after resampling: {:.2f}%'.format(self.check_fraud_rate(self.train_data)))
        print('Fraud rate in testing set after resampling: {:.2f}%'.format(self.check_fraud_rate(self.test_data)))
        self.normalize_data()
        self.reset_index
        return self.dataset