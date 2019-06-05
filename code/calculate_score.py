import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import PredictionHandler


class PerformanceMeasurer(object):
    
    def __init__(self, prediction_handler: PredictionHandler, metric_name: str, **kwargs):
        self._prediction_handler = prediction_handler
        self._valid_metrics = ['euclidean', 'roc']
        if metric_name not in self._valid_metrics:
            raise "Unknown metric: " + metric_name
        self._metric_name = metric_name
        self._kwargs = kwargs
        self._ground_truth_label = 'ground_truth'
        self._model_names = self._prediction_handler.get_models_list().remove(self._ground_truth_label)
        
    
    def _euclidean_score(self, y_true, y_pred):
        return np.sqrt(np.sum(np.square(y_pred - y_true)))
    
    
    def _roc_auc_score(self, y_true, y_pred):
        threshold = 3
        if 'threshold' in self._kwargs:
            threshold = self._kwargs['threshold']
        
        # Threshold ground truth ratings
        y_true_thresh = np.where(y_true >= threshold, 1, 0)
        
        # Scale the predictions to bring them to 0-1 range from 0-5 range
        y_pred_scaled = y_pred / 5
        
        # Calculate the area under the ROC curve
        area = roc_auc_score(y_true_thresh, y_pred_scaled)
        
        return area
    
    
    def get_scores(self):
        if self._metric_name == 'euclidean':
            metric = self._euclidean_score
        else:
            metric = self._roc_auc_score
            
        scores = {}
        y_true = self._prediction_handler.get_predictions(self._ground_truth_label)
        
        for model_name in self._model_names:
            scores[model_name] = metric(y_true, 
                                        self._prediction_handler.get_predictions(model_name))
        return scores
    
    
    def get_models_list(self):
        return self._model_names
    
    
def arg_parse():
    """
    Parse command line arguments passed to the module.

    """
    parser = argparse.ArgumentParser(description='Movie Recommendation System')

    parser.add_argument("--prediction_file", 
                        help="Path to the pickle file containing the predictions from all the models.",
                        type=str, required=True)
    parser.add_argument("--threshold", 
                        help="Confidence to threshold ground truth ratings for ROC.", 
                        default=3, type=float)
    parser.add_argument("--output_file", 
                        help="Path to the output csv file.",
                        type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()  # parse the command line arguments
    
    with open(args.prediction_file, "rb") as prediction_file:
        predictions_handler = pickle.load(prediction_file)
    
    threshold = args.threshold
    
    metrics = ['euclidean', 'roc']
    scores = {}
    
    for metric in metrics:
        scorer = PerformanceMeasurer(predictions_handler,
                                     metric_name=metric, threshold=threshold)
        scores[metric] = scorer.get_scores()
    
    df = pd.DataFrame.from_dict(scores)
    print(df)
    df.to_csv(args.output_file)
