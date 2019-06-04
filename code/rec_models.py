#!/usr/bin/env python
# coding: utf-8

"""
This file contains the recommendation models used in this recommendation system.

"""

from collections import defaultdict
import numpy as np


def pearson_similarity(ratings_dict_i, ratings_dict_j):
    """
    Computes the value of Pearson Correlation Similarity for the two ratings
    dictionaries (sparse representations) for format {key: rating, ...}.

    """
    common_ratings_i = []
    common_ratings_j = []
    for k_i, rat_i in ratings_dict_i.items():
        if k_i in ratings_dict_j:
            # These are common keys.
            common_ratings_i.append(rat_i)
            common_ratings_j.append(ratings_dict_j[k_i])

    # If there are less than 2 common elements, then consider them not related.
    if len(common_ratings_i) < 2:
        return 0

    # Adjust standard deviation to avoid nan.
    for cur_list in [common_ratings_i, common_ratings_j]:
        if np.std(cur_list) == 0:
            cur_list[0] += 1e-9

    pearson_coef = np.corrcoef(common_ratings_i, common_ratings_j)[0, 1]
    return pearson_coef


def jaccard_pearson_similarity(ratings_dict_i, ratings_dict_j):
    """
    Computes the product of Jaccard Similarity Pearson Correlation Similarity for the two ratings
    dictionaries (sparse representations) for format {key: rating, ...}.

    """
    ki = set(ratings_dict_i.keys())
    kj = set(ratings_dict_j.keys())
    jaccard_sim = len(ki.intersection(kj)) / len(ki.union(kj))

    pearson_sim = pearson_similarity(ratings_dict_i, ratings_dict_j)

    return jaccard_sim * pearson_sim


class CollaborativeFiltering(object):
    """
    Collaborative Filtering model.

    Params:
    ratings_mat: A sparse matrix containing the ratings with format:
    {'user_id': {'item_id': rating, ...}, ...}.

    k: The 'k' nearest neighbours to be considered. Default: all.

    """
    def __init__(self, ratings_mat, similarity_func, k=10):
        self.ratings_mat = ratings_mat
        self.k = k
        self.get_sim_score = similarity_func


    def _build_inv_ratings_mat(self):
        print("Building inverse ratings matrix...")
        # Inverse Ratings matrix format: {'item_id': {'user_id': rating, ...}, ...}.
        self.inv_ratings_mat = defaultdict(dict)
        for user_id, item_ratings in self.ratings_mat.items():
            for item_id, rating in item_ratings.items():
                self.inv_ratings_mat[item_id][user_id] = rating


    def _build_bias_terms(self):
        print("Building bias terms...")
        self.user_bias_dict = defaultdict(int)

        overall_rat_sum = 0
        overall_rat_count = 0

        user_rat_sum = defaultdict(int)
        user_rat_count = defaultdict(int)

        item_rat_sum = defaultdict(int)
        item_rat_count = defaultdict(int)

        for user_id, item_ratings in self.ratings_mat.items():
            for item_id, rating in item_ratings.items():
                overall_rat_sum += rating
                overall_rat_count += 1

                user_rat_sum[user_id] += rating
                user_rat_count[user_id] += 1

                item_rat_sum[item_id] += rating
                item_rat_count[item_id] += 1


        self.mu = overall_rat_sum / overall_rat_count

        self.user_avg_dict = {}
        for user_id, rat_sum in user_rat_sum.items():
            self.user_avg_dict[user_id] = rat_sum / user_rat_count[user_id]

        self.item_avg_dict = {}
        for item_id, rat_sum in item_rat_sum.items():
            self.item_avg_dict[item_id] = rat_sum / item_rat_count[item_id]


    def _train(self):
        self._build_inv_ratings_mat()
        self._build_bias_terms()


    def _get_bias_term(self, user_id, item_id):
        b_xi = self.mu + (self.user_avg_dict[user_id] - self.mu) + (self.item_avg_dict[item_id] - self.mu)
        return b_xi


    def _get_k_nearest_items(self, user_id, item_id):
        # TODO: Use LSH.
        items_x = list(self.ratings_mat[user_id].keys())
        # Exclude the current item if present and raise warning.
        if item_id in self.ratings_mat[user_id]:
            items_x.remove(item_id)
            print("warning: rating exists for current (user_id, item_id) pair: "
                  % (user_id, item_id))

        user_ratings_i = self.inv_ratings_mat[item_id]

        item_sim_dict = {}
        for item_id_j in items_x:
            user_ratings_j = self.inv_ratings_mat[item_id_j]
            item_sim_dict[item_id_j] = self.get_sim_score(user_ratings_i, user_ratings_j)

        sorted_item_sim_tups = sorted(item_sim_dict.items(), key=lambda tup: tup[0], reverse=True)
        k_nearest_items = [tup[0] for tup in sorted_item_sim_tups[:self.k]]
        return k_nearest_items


    def _get_k_nearest_users(self, user_id, item_id):
        # TODO: Use LSH.
        users_i = list(self.inv_ratings_mat[item_id].keys())
        # Exclude the current user if present and raise warning.
        if user_id in self.inv_ratings_mat[item_id]:
            users_i.remove(user_id)
            print("warning: rating exists for current (user_id, item_id) pair: "
                  % (user_id, item_id))

        item_ratings_x = self.ratings_mat[user_id]

        user_sim_dict = {}
        for user_id_y in users_i:
            item_ratings_y = self.ratings_mat[user_id_y]
            user_sim_dict[user_id_y] = self.get_sim_score(item_ratings_x, item_ratings_y)

        sorted_user_sim_tups = sorted(user_sim_dict.items(), key=lambda tup: tup[1], reverse=True)
        k_nearest_users = [tup[0] for tup in sorted_user_sim_tups[:self.k]]
        return k_nearest_users


    def item_item_cf_predict(self, user_id, item_id):
        nearest_items = self._get_k_nearest_items(user_id, item_id)

        weighted_sum = 0
        similarity_sum = 0

        user_ratings_i = self.inv_ratings_mat[item_id]
        for item_id_j in nearest_items:
            user_ratings_j = self.inv_ratings_mat[item_id_j]

            s_ij = self.get_sim_score(user_ratings_i, user_ratings_j)
            r_xj = self.ratings_mat[user_id][item_id_j]
            b_xj = self._get_bias_term(user_id, item_id_j)

            weighted_sum += s_ij * (r_xj - b_xj)
            similarity_sum += s_ij

        b_xi = self._get_bias_term(user_id, item_id)
        r_xi = b_xi + (weighted_sum / similarity_sum)
        return r_xi


    def user_user_cf_predict(self, user_id, item_id):
        nearest_users = self._get_k_nearest_users(user_id, item_id)

        weighted_sum = 0
        similarity_sum = 0

        item_ratings_x = self.ratings_mat[user_id]
        for user_id_y in nearest_users:
            item_ratings_y =  self.ratings_mat[user_id_y]

            s_xy = self.get_sim_score(item_ratings_x, item_ratings_y)
            r_yi = self.ratings_mat[user_id_y][item_id]
            b_yi = self._get_bias_term(user_id_y, item_id)

            weighted_sum += s_xy * (r_yi - b_yi)
            similarity_sum += s_xy

        b_xi = self._get_bias_term(user_id, item_id)
        r_xi = b_xi + (weighted_sum / similarity_sum)
        return r_xi


class LatentFactorModel(object):
    """
    Latent Factor model.

    Params:
    ratings_mat: A sparse matrix containing the ratings with format:
    {'user_id': {'item_id': rating, ...}, ...}.

    k: The 'k' nearest neighbours to be considered. Default: all.

    """
    def __init__(self, ratings_mat, similarity_func, k=10):
        self.ratings_mat = ratings_mat
        self.k = k
        self.get_sim_score = similarity_func