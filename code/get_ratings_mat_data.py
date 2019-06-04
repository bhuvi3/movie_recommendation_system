#!/usr/bin/env python
# coding: utf-8

from utils import create_ratings_matrix, relabel_item_ids, get_ratings_mat_train_val_test_split

import os

# Step 1: Create sparse representation of the ratings matrix.
ratings_file = "../data/ratings.csv"
ratings_mat_orig_outfile = "../data/ratings_mat_orig.pickle"
create_ratings_matrix(ratings_file, ratings_mat_orig_outfile)

# Step 2: Create relabel ratings_mat
new_rating_mat_outfile = "../data/ratings_mat.pickle"
orig_to_new_relabel_map_outfile = "../data/orig_to_new_relabel_map.pickle"
new_to_orig_relabel_map_outfile = "../data/new_to_orig_relabel_map.pickle"
relabel_item_ids(ratings_mat_orig_outfile, new_rating_mat_outfile, orig_to_new_relabel_map_outfile, new_to_orig_relabel_map_outfile)

# Step 3: Create train, val and test splits.
ratings_mat_pickle_file = new_rating_mat_outfile
split_outfile_prefix = os.path.splitext(ratings_mat_pickle_file)[0]
get_ratings_mat_train_val_test_split(ratings_mat_pickle_file,
                                     split_outfile_prefix,
                                     val_split=0.025,
                                     test_split=0.025, 
                                     random_state=0)
