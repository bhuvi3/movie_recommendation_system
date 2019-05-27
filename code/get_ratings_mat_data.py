#!/usr/bin/env python
# coding: utf-8

from utils import create_ratings_matrix, get_ratings_mat_train_val_test_split

ratings_file = "../data/ratings.csv"
outfile = "../data/ratings_mat.pickle"
create_ratings_matrix(ratings_file, outfile)

ratings_mat_pickle_file = outfile
split_outfile_prefix = "../data/ratings_mat"
get_ratings_mat_train_val_test_split(ratings_mat_pickle_file,
                                     split_outfile_prefix,
                                     val_split=0.15,
                                     test_split=0.25, 
                                     random_state=0)
