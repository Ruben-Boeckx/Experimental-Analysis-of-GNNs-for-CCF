import pandas as pd
import numpy as np
from geopy.distance import great_circle
import category_encoders as ce

class FeatureEngineering:
    def __init__(self, dataset):
        self.dataset = dataset

    def assign_unique_ids(self):
        # Assigning unique IDs to each unique cc_num
        self.dataset['cc_user'] = self.dataset['cc_num'].rank(method='dense') - 1

        # Assigning unique IDs to each unique merchant
        self.dataset['merchant_num'] = self.dataset['merchant'].rank(method='dense') - 1

    def transform_dates(self):
        # Change date type
        self.dataset['trans_date_trans_time'] = pd.to_datetime(self.dataset['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
        self.dataset['hour'] = self.dataset['trans_date_trans_time'].dt.hour
        self.dataset['day'] = self.dataset['trans_date_trans_time'].dt.day
        self.dataset['month'] = self.dataset['trans_date_trans_time'].dt.month
        self.dataset['weekday'] = self.dataset['trans_date_trans_time'].dt.weekday

        # Date of birth --> Age of customer
        self.dataset['dob'] = pd.to_datetime(self.dataset['dob'], format='%Y-%m-%d')
        self.dataset['age'] = (self.dataset['trans_date_trans_time'].dt.year - self.dataset['dob'].dt.year).astype(int)
        self.dataset.drop(columns='dob', inplace=True)

    def calculate_distance(self):
        # Location between customer home and merchant
        self.dataset['distance_km'] = self.dataset.apply(
            lambda col: round(great_circle((col['lat'], col['long']), (col['merch_lat'], col['merch_long'])).kilometers, 2),
            axis=1
        )
        self.dataset.drop(columns=['lat', 'long', 'merch_lat', 'merch_long'], inplace=True)

    def encode_gender(self):
        # Encode gender
        self.dataset['gender'] = self.dataset['gender'].map({'F': 0, 'M': 1})

    def calculate_time_between_transactions(self):
        # Calculate the time between transactions for each card
        self.dataset.sort_values(['cc_num', 'trans_date_trans_time'], inplace=True)
        self.dataset['hours_diff_bet_trans'] = (self.dataset.groupby('cc_num')[['trans_date_trans_time']].diff() / np.timedelta64(1, 'h'))
        self.dataset.loc[self.dataset['hours_diff_bet_trans'].isna(), 'hours_diff_bet_trans'] = 0
        self.dataset['hours_diff_bet_trans'] = self.dataset['hours_diff_bet_trans'].astype(int)

    def transform_transaction_amount(self):
        # Log transformation on transaction amount
        self.dataset["amt_log"] = np.log1p(self.dataset["amt"])
        self.dataset.drop(['amt'], axis=1, inplace=True)

    def drop_unused_columns(self):
        # Drop unnecessary columns
        self.dataset.drop(['Unnamed: 0', 'street', 'first', 'last', 'trans_num', 'unix_time', 'trans_date_trans_time', 'city_pop', 'zip'], axis=1, inplace=True)

    def count_encode_categorical_variables(self):
        # Count encoding on categorical variables
        encoder = ce.CountEncoder(cols=['category', 'state', 'city', 'job'])
        self.dataset = encoder.fit_transform(self.dataset)

    def reset_index(self):
        # Define each unique transaction
        self.dataset.reset_index(inplace=True)

    def execute(self):
        self.assign_unique_ids()
        self.transform_dates()
        self.calculate_distance()
        self.encode_gender()
        self.calculate_time_between_transactions()
        self.transform_transaction_amount()
        self.drop_unused_columns()
        self.count_encode_categorical_variables()
        self.reset_index()
        return self.dataset