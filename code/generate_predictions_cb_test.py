import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rec_models import CollaborativeFiltering, pearson_similarity
from utils import PredictionHandler

import os


def cf_item_model():
    ratings_train_file = '../data/ratings_mat_train.pickle'
    model_name = 'CF_Item'
    
    with open(ratings_train_file, "rb") as fp:
        ratings_mat_val = pickle.load(fp)
    cfm = CollaborativeFiltering(ratings_mat_val, pearson_similarity, 
                                 type='item-item')
    cfm.fit()
    return model_name, cfm


def cf_user_model():
    ratings_train_file = '../data/ratings_mat_train.pickle'
    model_name = 'CF_User'
    
    with open(ratings_train_file, "rb") as fp:
        ratings_mat_val = pickle.load(fp)
    cfm = CollaborativeFiltering(ratings_mat_val, pearson_similarity, 
                                 type='user-user')
    cfm.fit()
    return model_name, cfm


def lfm_model():
    trained_lfm_path = "../data/trained_lfm.pickle"
    model_name = 'LFM'
    
    with open(trained_lfm_path, "rb") as fp:
        lfm = pickle.load(fp)

    return model_name, lfm


def content_model_plots():
    with open('../data/new_movieid_plots.pickle','rb') as f:
        plots = pickle.load(f)

    with open('../data/ratings_mat_train.pickle','rb') as f:
        mat = pickle.load(f)

    cb = ContentBased(ratings_mat=mat, movie_embedding=plots, type_mode='plot')

    return cb


def content_model_posters():
    with open('../data/orig_to_new_poster_encodings.pickle','rb') as f:
        posters = pickle.load(f)

    with open('../data/ratings_mat_train.pickle','rb') as f:
        mat = pickle.load(f)

    cb = ContentBased(ratings_mat=mat, poster_embedding=posters, type_mode='poster')

    return cb


def get_predictions(model_configs, data, instance_idx, outdir, write_freq):
    print('Predictor started..')
    predictions = []
    
    # Acquire and initialize the models
    models = []
    for model_cfg in model_configs:
        models.append(model_cfg())
        print('Initialized model ' + models[-1][0])
        
    # Acquire the prediction on each model for each data point
    for idx, data_pt in enumerate(data):
        u_id = data_pt['user_id']
        m_id = data_pt['movie_id']
        y_true = data_pt['y_true']
        result = {'user_id': u_id, 'movie_id': m_id, 'y_true':y_true}
        try:
            for model_name, model in models:
                y_pred = model.predict(u_id, m_id)
                y_pred = min(y_pred, 5)
                y_pred = max(y_pred, 0.5)
                result[model_name] = y_pred
            
            predictions.append(result)
        except:
            continue
        
        if idx % write_freq == 0:
            # Save the predictions to temporary csv
            predictions_temp = list(np.ravel(predictions))
            df = pd.DataFrame.from_dict(predictions_temp).dropna()
            df.to_csv('%s/predictions_instance_%d_%d.csv'
                      % (outdir, instance_idx, idx), index=False)
    
        
    print('Predictor completed.')
    return predictions


def generate_predictions(model_configs, data, output_csv_file, 
                         output_pickle_file, n_splits, n_jobs, outdir, write_freq):
    # Create indices for splitting the data
    indices = np.arange(len(data), dtype=int)
    split_indices = np.array_split(indices, n_splits)
    
    # Launch get_predictions in parallel for each split
    predictions = Parallel(n_jobs=n_jobs)(delayed(get_predictions)(deepcopy(model_configs),
                                                                   data[split_idx],
                                                                   instance_idx,
                                                                   outdir,
                                                                   write_freq)
                                          for instance_idx, split_idx 
                                          in enumerate(split_indices))
    
    # Save the predictions to csv
    predictions = list(np.ravel(predictions))
    df = pd.DataFrame.from_dict(predictions).dropna()
    df.to_csv(output_csv_file, index=False)
    
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
    dataset_name = "test"
    write_freq = 5000
    n_jobs = 32
    model_configs = [content_model_plots, content_model_posters]

    # Running the inference pipeline with above params.
    temp_outputs_dir = "../data/CB_PlotPoster_%s_temp_preds" % dataset_name
    os.makedirs(temp_outputs_dir)
    print("Creating temp outputs at %s" % temp_outputs_dir)

    test_data_file = '../data/ratings_mat_%s.pickle' % dataset_name
    data = read_and_flatten_dict(test_data_file)
    print("Running inference on: %s" % test_data_file)

    n_splits = n_jobs
    output_csv_file = '../data/%s_predictions.csv' % dataset_name
    output_pickle_file = '../data/%s_predictions_handler.pickle' % dataset_name
    
    generate_predictions(model_configs, data, output_csv_file, output_pickle_file, n_splits, n_jobs, temp_outputs_dir, write_freq)


if __name__ == "__main__":
    main()
