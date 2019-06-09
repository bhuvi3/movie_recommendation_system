#!/usr/bin/env python
# coding: utf-8

import os
from collections import defaultdict
from io import BytesIO
from PIL import Image

import numpy as np
import pandas as pd
import pickle
import random
import requests


class ImageFetcher:
    """
    Pulls images from url present in a csv file and dumps them into a folder.
    Creates two text files consisting a list of successfully downloaded and 
    failed urls. Allows resizing of images.
    
    """
    
    def __init__(self, data_file_path, url_column_name, output_file_name_column, output_folder_path, 
                 output_postfix=None, url_prefix=None, resize_shape=None):
        """
        Just your standard class initialization. 🤷
        
        Input:
        -------
        data_file_path (str): Path to the csv file which contains the URLs. 
        url_column_name (str): Name of the column in the csv file which contains the URLs.
        output_file_name_column (str): Name of the column in the csv file which contains the output file name.
        output_folder_path (str): Path to the output folder where the images need to be dumped.
        output_postfix (str): Postfix string appended to the output file.
        url_prefix (str): Base URL string.
        resize_shape(tuple): Shape to which image needs to be resized. Preserves original size if set to None.
        
        """
        self._data_file_path = os.path.abspath(data_file_path)
        self._url_column_name = url_column_name
        self._output_file_name_column = output_file_name_column
        self._output_folder_path = os.path.abspath(output_folder_path)
        self._url_prefix = url_prefix
        self._resize_shape = resize_shape
        
        if not output_postfix:
            self._output_postfix = ''
        else:
            self._output_postfix = '_' + output_postfix 

    def _download_image(self, url, image_file_path):
        """
        Downloads the image and dumps to a file.

        Input:
        -------
        url (str): URL to the image which needs to be downloaded.
        image_file_path (str): Path to the output file.

        Output:
        --------
        successful (boolean): Flag for successful or failed download.

        """
        r = requests.get(url)
        if r.status_code != requests.codes.ok:
            return False

        with Image.open(BytesIO(r.content)) as im:
            if self._resize_shape:
                im = im.resize(self._resize_shape, resample=Image.LANCZOS)
            im.save(image_file_path)
        
        return True
            
    def _get_metadata(self):
        """
        Reads the csv file and returns a list of all the URLs after prepending base path 
        and filenames after appending the postfix string.
        
        Output:
        --------
        metadata (list): List of tuples of filenames, URLs.
        
        """
        metadata = []

        data_file = pd.read_csv(self._data_file_path, dtype=str)
        subset_data = data_file[[self._output_file_name_column, self._url_column_name]]
        subset_data = subset_data.dropna()
        final_data = subset_data[subset_data[self._url_column_name].astype(str).str.match("\/.+\.[a-zA-Z]+")]
        
        for filename, url in final_data.values.tolist():
            filename = self._url_to_filepath(filename, url)
            if self._url_prefix:
                url = self._url_prefix + url
            metadata.append((filename, url))
        
        return metadata
    
    def _url_to_filepath(self, filename, url):
        """
        Extracts extension from URL and returns the output path after appending the postfix string and extension.
        
        Input:
        -------
        filename (str): Filename for the image.
        url (str): URL of the image.
        
        Output:
        --------
        filepath (str): Output path for the image.
        
        """
        filepath, ext = os.path.splitext(url)
        
        filepath = os.path.join(self._output_folder_path, filename + self._output_postfix + ext)
        
        return filepath
    
    def fetch_images(self):
        """
        Download images and saves them to the folder. Creates two text 
        files consisting a list of successfully downloaded and failed urls.
        
        """
        successful_filename = os.path.join(self._output_folder_path, 
                                           self._output_postfix[1:] + '.txt')
        failed_filename = os.path.join(self._output_folder_path, 
                                       self._output_postfix[1:] + '_failed.txt')

        successful_list = []
        failed_list = []

        metadata = self._get_metadata()
        idx = 0
        
        for output_filepath, url in metadata:
            successful = self._download_image(url, output_filepath)
            if successful:
                successful_list.append(output_filepath)
            else:
                failed_list.append((output_filepath, url))
            idx += 1
            if idx % 1000 == 0:
                print(idx)  # for debug purpose
        
        with open(successful_filename, 'w') as f:
            for item in successful_list:
                f.write("%s\n" % item)
        
        with open(failed_filename, 'w') as f:
            for item in failed_list:
                f.write("%s\t%s\n" % item)



def create_ratings_matrix(ratings_file, outfile, sep=','):
    """
    Creates a ratings from the ratings file and returns a dictionary containing ratings matrix
    for optimized sparse storage.

    Assumes that the data csv file has a header and data in format: "user_id,movie_id,rating,timestamp\n"
    (MovieLens data format).

    Ratings matrix format: {'user_id': {'movie_id': rating, ...}, ...}

    """
    fp = open(ratings_file)
    header = fp.readline()

    ratings_mat = defaultdict(dict)
    users = set()
    movies = set()
    ratings_count = 0
    for line in fp:
        ratings_count += 1
        user_id, movie_id, rating, _ = line.strip().split(sep)
        user_id = int(user_id)
        movie_id = int(movie_id)
        users.add(user_id)
        movies.add(movie_id)
        rating = float(rating)
        if rating > 5:
            raise ValueError("The rating cannot be greater than 5.0: %s" % rating)
        ratings_mat[user_id][movie_id] = np.float16(rating)

    fp.close()
    print("Number of ratings: %d" % ratings_count)
    print("Number of unique users: %d, range: [%d, %d]" % (len(users), min(users), max(users)))
    print("Number of unique movies: %d, range: [%d, %d]" % (len(movies), min(movies), max(movies)))

    with open(outfile, "wb") as out_fp:
        pickle.dump(ratings_mat, out_fp)

    print("The ratings sparse matrix has been saved in %s" % outfile)


def relabel_item_ids(ratings_mat_orig_pickle_file, new_rating_mat_outfile, orig_to_new_relabel_map_outfile, new_to_orig_relabel_map_outfile):
    """
    Relabels the item ids to keep them consecutive in the ratings matrix and writes
    the relabelled ratings mat to new_rating_mat_outfile, and also saves the relabel_maps.

    """
    with open(ratings_mat_orig_pickle_file, "rb") as fp:
        ratings_mat_orig = pickle.load(fp)

    orig_to_new_relabel_map = {}
    k = 0
    for user_id, item_ratings_dict in ratings_mat_orig.items():
        for item_id in item_ratings_dict.keys():
            if item_id not in orig_to_new_relabel_map:
                orig_to_new_relabel_map[item_id] = k
                k += 1

    new_to_orig_relabel_map = dict([[v, k] for k, v in orig_to_new_relabel_map.items()])

    # Apply relabels.
    new_ratings_mat = defaultdict(dict)
    for user_id, item_ratings_dict in ratings_mat_orig.items():
        for item_id, rating in item_ratings_dict.items():
            new_ratings_mat[user_id][orig_to_new_relabel_map[item_id]] = rating

    # Save the files.
    with open(new_rating_mat_outfile, "wb") as out_fp:
        pickle.dump(new_ratings_mat, out_fp)
        print("New ratings mat is saved at %s" % new_rating_mat_outfile)

    with open(orig_to_new_relabel_map_outfile, "wb") as out_fp:
        pickle.dump(orig_to_new_relabel_map, out_fp)
        print("Original to New item ids map is saved at %s" % orig_to_new_relabel_map_outfile)

    with open(new_to_orig_relabel_map_outfile, "wb") as out_fp:
        pickle.dump(new_to_orig_relabel_map, out_fp)
        print("New to Original item ids map is saved at %s" % new_to_orig_relabel_map_outfile)



def get_ratings_mat_train_val_test_split(ratings_mat_pickle_file, outfile_prefix, val_split=0.005, test_split=0.005, random_state=None):
    """
    Creates the train, val and test split of the ratings matrix, and writes them
    to in the path referred by outfile_prefix as
    (<outfile_prefix>_train.pickle, <outfile_prefix>_val.pickle, <outfile_prefix>_test.pickle).

    """
    with open(ratings_mat_pickle_file, "rb") as fp:
        ratings_mat = pickle.load(fp)

    num_ratings_list = [len(v) for k, v in ratings_mat.items()]
    total_ratings = sum(num_ratings_list)

    val_size = int(val_split * total_ratings)
    test_size = int(test_split * total_ratings)
    train_size = total_ratings - (val_size + test_size)

    print("train_size: %s" % train_size)
    print("val_size: %s" % val_size)
    print("test_size: %s" % test_size)

    # 0: train_set, 1: val_set, 2: test_set
    split_indicator = [0] * train_size + [1] * val_size + [2] * test_size

    if random_state is not None:
        print("Setting the random seed to %s" % random_state)
        random.seed(random_state)

    random.shuffle(split_indicator)

    ratings_mat_train = defaultdict(dict)
    ratings_mat_val = defaultdict(dict)
    ratings_mat_train_val = defaultdict(dict)
    ratings_mat_test = defaultdict(dict)

    # Index to be considered from the split_indicator, starting from zero.
    i = 0 
    for user_id, movie_ratings in ratings_mat.items():
        for movie_id, rating in movie_ratings.items():
            split_ind = split_indicator[i]
            if split_ind == 0:
                ratings_mat_train[user_id][movie_id] = rating
                ratings_mat_train_val[user_id][movie_id] = rating
            if split_ind == 1:
                ratings_mat_val[user_id][movie_id] = rating
                ratings_mat_train_val[user_id][movie_id] = rating
            if split_ind == 2:
                ratings_mat_test[user_id][movie_id] = rating

            i += 1

    # Save them in pickle files.
    for cur_rating_dict, cur_split_name in zip([ratings_mat_train, ratings_mat_val, ratings_mat_train_val, ratings_mat_test],
                                               ["train", "val", "train_val", "test"]):
        outfile = "%s_%s.pickle" % (outfile_prefix, cur_split_name)
        with open(outfile, "wb") as out_fp:
            pickle.dump(cur_rating_dict, out_fp)

        print("%s ratings sparse matrix has been saved in %s" % (cur_split_name, outfile))


class PredictionHandler(object):
    def __init__(self, ground_truth):
        self._predictions = {'ground_truth': ground_truth}
        self._num_preds = len(ground_truth)
        
    def add_prediction(self, model_name, predictions):
        if len(predictions) != self._num_preds:
            raise "Number of predictions different from the ground truth."
        self._predictions[model_name] = predictions
    
    def get_models_list(self):
        return list(self._predictions.keys())
    
    def get_predictions(self, model_name=None):
        if model_name and model_name in self._predictions:
            return self._predictions[model_name]
        else:
            return self._predictions


def df_to_prediction_handler(df):
    y_true = np.array(df['y_true'].values)
    predicted_df = df.drop(columns=['user_id', 'movie_id', 'y_true'])
    columns = predicted_df.columns
    
    prediction_handler = PredictionHandler(ground_truth=y_true)
    for model_name in columns:
        prediction_handler.add_prediction(model_name, 
                                          np.array(df[model_name].values))
        
    return prediction_handler
        