import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rec_models import CollaborativeFiltering, pearson_similarity
from utils import PredictionHandler


def cf_item_model():
    ratings_train_file = '../data/ratings_mat_train.pickle'
    model_name = 'CF_Item'
    
    with open(ratings_train_file, "rb") as fp:
        ratings_mat_val = pickle.load(fp)
    cfm = CollaborativeFiltering(ratings_mat_val, pearson_similarity, 
                                 type='item-item')
    cfm.fit()
    return model_name, cfm


def get_predictions(models, data):
    predictions = []
        
    # Acquire the prediction on each model for each data point
    for data_pt in data:
        u_id = data_pt['user_id']
        m_id = data_pt['movie_id']
        y_true = data_pt['y_true']
        result = {'user_id': u_id, 'movie_id': m_id, 'y_true':y_true}
        
        for model_name, model in models:
            y_pred = model.predict(u_id, m_id)
            y_pred = max(y_pred, 5)
            y_pred = min(y_pred, 0.5)
            result[model_name] = y_pred
        
        predictions.append(result)
    
    return predictions


def generate_predictions(model_configs, data, output_csv_file, 
                         output_pickle_file, n_splits=5, n_jobs=-1):
    # Create indices for splitting the data
    indices = np.arange(len(data), dtype=int)
    split_indices = np.array_split(indices, n_splits)
    
    # Acquire and initialize the models
    models = []
    for model_cfg in model_configs:
        models.append(model_cfg())
        print('Initialized model ' + models[-1][0])
    
    # Launch get_predictions in parallel for each split
    predictions = Parallel(n_jobs=n_jobs)(delayed(get_predictions)(deepcopy(models),
                                                                   data[split_idx])
                                          for split_idx in split_indices)
    
    # Save the predictions to csv
    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(output_csv_file)
    
    # Create an instance of PredictionHandler and move data to the object
    y_true = np.array(df['y_true'].values)
    predicted_df = df.drop(columns=['user_id', 'movie_id', 'y_true'])
    columns = predicted_df.columns
    
    prediction_handler = PredictionHandler(ground_truth=y_true)
    for model_name in columns:
        prediction_handler.add_prediction(model_name, 
                                          np.array(df[model_name].values))
        
    # Pickle the PredictionHandler object for performance Analysis
    with open(output_pickle_file, 'wb') as pf:
        pickle.dump(prediction_handler, pf)
        

def read_and_flatten_dict(data_file):
    flattened_data = []
    
    with open(data_file, 'rb') as f:
        data_orig = pickle.load(f)
        
    for user_id, item_ratings_dict in data_orig.items():
        for movie_id, rating in item_ratings_dict.items():
            flattened_data.append({'user_id': user_id, 
                                   'movie_id': movie_id,
                                   'y_true': rating})       
    
    return np.array(flattened_data)


def main():
    test_data_file = '../data/ratings_mat_test.pickle'
    data = read_and_flatten_dict(test_data_file)
    
    model_configs = [cf_item_model]
    n_splits = 4
    n_jobs = 2
    output_csv_file = '../data/test_predictions.csv'
    output_pickle_file = '../data/test_predictions_handler.pickle'
    
    generate_predictions(model_configs, data, output_csv_file,
                         output_pickle_file, n_splits, n_jobs)
    

if __name__ == "__main__":
    main()
