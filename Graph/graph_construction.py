import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

class GraphConstruction:
    def __init__(self, dataset):
        self.dataset = dataset
        self.client_unique_features = []
        self.merchant_unique_features = []
        self.length_resampled_df = 922802

    def identify_unique_features(self):
        client_grouped_dataset = self.dataset.groupby('cc_user')

        for cc_user, group in client_grouped_dataset:
            for column in group.columns:
                if group[column].nunique() == 1:
                    self.client_unique_features.append(column)

        self.client_unique_features = list(set(self.client_unique_features))

        for feature in self.dataset.columns:
            if self.dataset[feature].nunique() == client_grouped_dataset[feature].nunique().max() and feature in self.client_unique_features:
                self.client_unique_features.remove(feature)

        merchant_grouped_dataset = self.dataset.groupby('merchant_num')

        for merchant_num, group in merchant_grouped_dataset:
            for column in group.columns:
                if group[column].nunique() == 1:
                    self.merchant_unique_features.append(column)

        self.merchant_unique_features = list(set(self.merchant_unique_features))

        for feature in self.dataset.columns:
            if self.dataset[feature].nunique() == merchant_grouped_dataset[feature].nunique().max() and feature in self.merchant_unique_features:
                self.merchant_unique_features.remove(feature)

        # Manually set features
        self.client_unique_features = ['job', 'gender', 'age', 'state', 'city']
        self.merchant_unique_features = ['category']

    def generate_tables(self):
        clients = pd.DataFrame()
        merchants = pd.DataFrame()
        
        client_grouped_dataset = self.dataset.groupby('cc_user')
        merchant_grouped_dataset = self.dataset.groupby('merchant_num')

        for cc_user, group in client_grouped_dataset:
            for feature in self.client_unique_features:
                clients.at[cc_user, feature] = group[feature].iloc[0]

        for merchant_num, group in merchant_grouped_dataset:
            for feature in self.merchant_unique_features:
                merchants.at[merchant_num, feature] = group[feature].iloc[0]

        return clients, merchants

    def construct_graph(self, clients, merchants):
        data = HeteroData()

        # Nodes
        data['transaction'].x = torch.tensor(self.dataset[['hour', 'day', 'month', 'weekday', 'amt_log', 'distance_km', 'hours_diff_bet_trans']].values, dtype=torch.float)
        data['client'].x = torch.tensor(clients.values, dtype=torch.float)
        data['merchant'].x = torch.tensor(merchants.values, dtype=torch.float)

        # Edge Indices
        data['client', 'pays', 'transaction'].edge_index = torch.tensor(self.dataset[['cc_user', 'index']].values, dtype=torch.long).t().contiguous()
        data['transaction', 'received by', 'merchant'].edge_index = torch.tensor(self.dataset[['index', 'merchant_num']].values, dtype=torch.long).t().contiguous()

        # Target and Classes
        data['transaction'].y = torch.tensor(self.dataset['is_fraud'].values, dtype=torch.float)
        data['transaction'].num_classes = 2

        num_nodes = data['transaction'].x.size(0)
        split_test = self.length_resampled_df + int(0.5 * (num_nodes - self.length_resampled_df))

        train_idx = torch.arange(self.length_resampled_df)
        test_idx = torch.arange(self.length_resampled_df, split_test)
        val_idx = torch.arange(split_test, num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = 1
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_idx] = 1
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_idx] = 1

        data['transaction'].train_mask = train_mask
        data['transaction'].test_mask = test_mask
        data['transaction'].val_mask = val_mask

        data = T.ToUndirected()(data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.to(device)

        return data

    def calculate_fraud_percentage(self, data):
        train_mask_fraud_count = data['transaction'].y[data['transaction'].train_mask].sum().item()
        train_mask_total = data['transaction'].train_mask.sum().item()
        fraud_percentage_train_mask = (train_mask_fraud_count / train_mask_total) * 100

        test_mask_fraud_count = data['transaction'].y[data['transaction'].test_mask].sum().item()
        test_mask_total = data['transaction'].test_mask.sum().item()
        fraud_percentage_test_mask = (test_mask_fraud_count / test_mask_total) * 100

        val_mask_fraud_count = data['transaction'].y[data['transaction'].val_mask].sum().item()
        val_mask_total = data['transaction'].val_mask.sum().item()
        fraud_percentage_val_mask = (val_mask_fraud_count / val_mask_total) * 100

        return fraud_percentage_train_mask, fraud_percentage_test_mask, fraud_percentage_val_mask

    def apply_graph_construction(self):
        self.identify_unique_features()
        clients, merchants = self.generate_tables()
        data = self.construct_graph(clients, merchants)
        fraud_percentage_train_mask, fraud_percentage_test_mask, fraud_percentage_val_mask = self.calculate_fraud_percentage(data)

        print(f'Fraud Percentage in Train Mask: {fraud_percentage_train_mask:.2f}%')
        print(f'Fraud Percentage in Test Mask: {fraud_percentage_test_mask:.2f}%')
        print(f'Fraud Percentage in Val Mask: {fraud_percentage_val_mask:.2f}%')

        data.validate()
        print('Graph Construction Successful!')
        return data