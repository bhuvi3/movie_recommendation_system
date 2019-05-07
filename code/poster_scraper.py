#!/usr/bin/env python
# coding: utf-8

from utils import ImageFetcher

# Pulls posters to all the movies using the links present in 'movie_metadata.csv'.

# Initialize variables
output_folder_path = '../data/posters'
data_file_path =  '../data/movies_metadata.csv'
url_column_name = 'poster_path'
output_file_name_column = 'id'
output_postfix = 'poster'
url_prefix = 'https://image.tmdb.org/t/p/w600_and_h900_bestv2'


# Initialize Fetcher
image_fetcher = ImageFetcher(data_file_path, url_column_name, output_file_name_column, 
                             output_folder_path, output_postfix, url_prefix)

# Fetch                             
image_fetcher.fetch_images()
