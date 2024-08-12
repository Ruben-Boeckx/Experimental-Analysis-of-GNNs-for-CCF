import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class ClassifierEvaluator:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.rf_classifier = RandomForestClassifier(random_state=42)
        self.log_reg = LogisticRegression(max_iter=1000, random_state=42)
        self.grid_search_rf = None
        self.grid_search_lr = None
        self.best_rf = None
        self.best_log_reg = None
    
    def train_random_forest(self):
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.grid_search_rf = GridSearchCV(self.rf_classifier, param_grid_rf, cv=5, scoring='roc_auc')
        self.grid_search_rf.fit(self.X_train, self.y_train)
        self.best_rf = self.grid_search_rf.best_estimator_
    
    def evaluate_random_forest(self):
        y_proba_rf = self.best_rf.predict_proba(self.X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, y_proba_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        precision_rf, recall_rf, _ = precision_recall_curve(self.y_test, y_proba_rf)
        pr_auc_rf = average_precision_score(self.y_test, y_proba_rf)
        return {
            'roc_auc': roc_auc_rf,
            'fpr': fpr_rf,
            'tpr': tpr_rf,
            'precision': precision_rf,
            'recall': recall_rf,
            'pr_auc': pr_auc_rf
        }
    
    def train_logistic_regression(self):
        param_grid_lr = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        self.grid_search_lr = GridSearchCV(self.log_reg, param_grid_lr, cv=5, scoring='roc_auc')
        self.grid_search_lr.fit(self.X_train, self.y_train)
        self.best_log_reg = self.grid_search_lr.best_estimator_
    
    def evaluate_logistic_regression(self):
        y_proba_lr = self.best_log_reg.predict_proba(self.X_test)[:, 1]
        fpr_lr, tpr_lr, _ = roc_curve(self.y_test, y_proba_lr)
        roc_auc_lr = auc(fpr_lr, tpr_lr)
        precision_lr, recall_lr, _ = precision_recall_curve(self.y_test, y_proba_lr)
        pr_auc_lr = average_precision_score(self.y_test, y_proba_lr)
        return {
            'roc_auc': roc_auc_lr,
            'fpr': fpr_lr,
            'tpr': tpr_lr,
            'precision': precision_lr,
            'recall': recall_lr,
            'pr_auc': pr_auc_lr
        }