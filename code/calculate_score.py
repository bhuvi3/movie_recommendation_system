from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve

from utils import PredictionHandler


class PerformanceAnalyzer(object):
    
    def __init__(self, prediction_handler: PredictionHandler, 
                 roc_thresholds=[3], rmse_thresholds=[0]):
        self._prediction_handler = prediction_handler
        self._ground_truth_label = 'ground_truth'
        self._model_names = self._prediction_handler.get_models_list()
        self._model_names.remove(self._ground_truth_label)
        self._roc_thresholds = roc_thresholds
        self._rmse_thresholds = rmse_thresholds
    
    
    def _rmse_at_threshold(self, y_true, y_pred, threshold):
        indices = y_true >= threshold
        rmse = sqrt(mean_squared_error(y_true[indices], y_pred[indices]))
        return rmse
    
    
    def _roc_at_threshold(self, y_true, y_pred, threshold):
        
        # Threshold ground truth ratings
        y_true_thresh = np.where(y_true >= threshold, 1, 0)
        
        # Scale the predictions to bring them to 0-1 range from 0-5 range
        y_pred_scaled = y_pred / 5
        
        # Calculate the area under the ROC curve
        area = roc_auc_score(y_true_thresh, y_pred_scaled)
        
        return area
    
    
    def get_scores(self):
        scores = {}
        y_true = self._prediction_handler.get_predictions(self._ground_truth_label)
        
        for roc_threshold in self._roc_thresholds:
            metric_name = 'roc_auc_at_' + str(roc_threshold)
            score_metric = {}
            for model in self._model_names:
                score_metric[model] = self._roc_at_threshold(y_true, 
                                                             self._prediction_handler.get_predictions(model), 
                                                             roc_threshold)
            scores[metric_name] = score_metric
            
        for rmse_threshold in self._rmse_thresholds:
            metric_name = 'rmse_at_' + str(rmse_threshold)
            score_metric = {}
            for model in self._model_names:
                score_metric[model] = self._rmse_at_threshold(y_true, 
                                                              self._prediction_handler.get_predictions(model), 
                                                              rmse_threshold)
            scores[metric_name] = score_metric
        
        return scores
    
    
    def plot_roc_at_threshold(self, output_file, threshold=3):
        plt.rcParams['figure.figsize'] = [8, 8]
        
        # Threshold ground truth ratings
        y_true = np.where(self._prediction_handler.get_predictions(self._ground_truth_label) >= threshold, 
                          1, 0)
        plt.plot([0, 1], [0, 1], 'k--')
        for model in self._model_names:
            y_pred = self._prediction_handler.get_predictions(model)
            fpr, tpr, thresh = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (model, auc))
            
        plt.xlabel('Specificity (False Positive Rate)')
        plt.ylabel('Sensitivity (True Positive Rate)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(output_file)
    
    
    def get_models_list(self):
        return self._model_names
    