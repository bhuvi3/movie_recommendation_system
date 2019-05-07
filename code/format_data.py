#!/usr/bin/env python
# coding: utf-8

from utils import create_ratings_matrix

ratings_file = "../data/ratings.csv"
outfile = "../data/ratings_mat.pickle"
create_ratings_matrix(ratings_file, outfile)
