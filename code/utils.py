#!/usr/bin/env python
# coding: utf-8

import os
from collections import defaultdict
from io import BytesIO
from PIL import Image

import numpy as np
import pandas as pd
import pickle
import requests


class ImageFetcher:
    """
    Pulls images from url present in a csv file and dumps them into a folder.
    Creates two text files consisting a list of successfully downloaded and 
    failed urls.
    
    """
    
    def __init__(self, data_file_path, url_column_name, output_file_name_column, output_folder_path, 
                 output_postfix=None, url_prefix=None):
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
        
        """
        self._data_file_path = os.path.abspath(data_file_path)
        self._url_column_name = url_column_name
        self._output_file_name_column = output_file_name_column
        self._output_folder_path = os.path.abspath(output_folder_path)
        self._url_prefix = url_prefix
        
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
        
        for output_filepath, url in metadata:
            successful = self._download_image(url, output_filepath)
            if successful:
                successful_list.append(output_filepath)
            else:
                failed_list.append((output_filepath, url))
        
        with open(successful_filename, 'w') as f:
            for item in successful_list:
                f.write("%s\n" % item)
        
        with open(failed_filename, 'w') as f:
            for item in failed_list:
                f.write("%s\n" % item)



def create_ratings_matrix(ratings_file, outfile, sep=','):
    """
    Creates a ratings from the ratings file and returns a dictionary containing ratings matrix
    for optimized sparse storage.

    Assumes that the data csv file has a header and data in format: "user_id,movie_id,rating,timestamp\n"
    (MovieLens data format).

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
