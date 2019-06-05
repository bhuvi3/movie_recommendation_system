#!/usr/bin/env python
# coding: utf-8

"""
This file contains the recommendation models used in this recommendation system.

"""

from collections import defaultdict

import numpy as np
import time


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


    def fit(self):
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
    Latent Factor model using Stochastic Gradient Descent (SGD) with bias terms.
    Note: Assumes consecutive user_ids and item_ids which are non-negative.

    Params:
    ratings_mat: A sparse matrix containing the ratings with format:
    {'user_id': {'item_id': rating, ...}, ...}.

    k: The 'k' latent factors to be considered. Default: 20.

    lam_1: Regularization param for Q (item factors). Default: 0.1

    lam_2: Regularization param for P (user factors). Default: 0.1

    lam_3: Regularization param for Bx (user biases). Default: 0.1

    lam_4: Regularization param for Bi (item biases). Default: 0.1

    eta: The step-size (constant) to be used for SGD.

    random_state: Sets np.random seed before shuffling the data.

    max_iter: The number of iterations over the whole dataset. Default: 100.

    """
    def __init__(self, ratings_mat, k=20, lam_1=0.1, lam_2=0.1, lam_3=0.1, lam_4=0.1, eta=0.01, random_state=None, max_iter=100):
        self.ratings_mat = ratings_mat

        self.k = k
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.lam_3 = lam_3
        self.lam_4 = lam_4

        self.eta = eta
        self.random_state = random_state
        self.max_iter = max_iter


    # TODO: Duplicate code, as it is present even in CollaborativeFiltering.
    def _build_bias_terms(self):
        print("Building bias terms...")
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

        # XXX: Adjusts user id starting from 1.
        print("Adjusting user id 0, as it starts from 1.")
        if 0 not in self.user_avg_dict:
            self.user_avg_dict = {0: 0}


    def _get_triples_list(self):
        print("Getting triples (userd_id, item_id, rating)...")
        triples = []
        max_user_id = -1
        max_item_id = -1
        max_rating = -1
        for user_id, item_ratings_dict in self.ratings_mat.items():
            for item_id, rating in item_ratings_dict.items():
                triples.append((user_id, item_id, rating))

                if user_id > max_user_id:
                    max_user_id = user_id

                if item_id > max_item_id:
                    max_item_id = item_id

                if rating > max_rating:
                    max_rating = rating

        self.max_user_id = max_user_id
        self.max_item_id = max_item_id
        self.max_rating = max_rating

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # XXX: This might take a lot of time.
        np.random.shuffle(triples)
        return triples


    def _objective_func(self, Q, P, Bx, Bi):
        error = 0
        for rating_triple in self.rating_triples:
            u_id, i_id, r_xi = rating_triple

            q_i = Q[i_id]
            p_x = P[u_id]
            b_x = Bx[u_id]
            b_i = Bi[i_id]
            mu = self.mu

            pred_rating = mu + b_x + b_i + np.dot(q_i, p_x)
            error += np.square(r_xi - pred_rating)

        reg_err =   (self.lam_1 * np.sum(np.square(np.linalg.norm(Q))))  \
                  + (self.lam_2 * np.sum(np.square(np.linalg.norm(P))))  \
                  + (self.lam_3 * np.sum(np.square(np.linalg.norm(Bx)))) \
                  + (self.lam_4 * np.sum(np.square(np.linalg.norm(Bi))))

        objective_val = error + reg_err
        return objective_val


    def fit(self):
        """
        Train the Latent Factor Model using SGD.

        """
        # Parameter Initialization.
        print("Initializing Parameters...")
        self._build_bias_terms()
        self.rating_triples = self._get_triples_list()

        Q = np.random.uniform(0, np.sqrt(self.max_rating / self.k), size=(self.max_item_id + 1, self.k))
        P = np.random.uniform(0, np.sqrt(self.max_rating / self.k), size=(self.max_user_id + 1, self.k))
        Bx = np.array([self.user_avg_dict[user_id] - self.mu for user_id in range(self.max_user_id + 1)])
        Bi = np.array([self.item_avg_dict[item_id] - self.mu for item_id in range(self.max_item_id + 1)])

        print("Training for %s iterations..." % self.max_iter)
        self.objective_values_list = []
        obj_0 = self._objective_func(Q, P, Bx, Bi)
        self.objective_values_list.append(obj_0)
        print("Initial objective value: %s" % obj_0)

        for t in range(self.max_iter):
            st = time.time()
            for rating_triple in self.rating_triples:
                u_id, i_id, r_xi = rating_triple

                q_i = Q[i_id]
                p_x = P[u_id]
                b_x = Bx[u_id]
                b_i = Bi[i_id]
                mu = self.mu

                eps_xi = 2 * (r_xi - (mu + b_x + b_i + np.dot(q_i, p_x)))
                new_q_i = q_i + (self.eta * ((eps_xi * p_x) - (2 * self.lam_1 * q_i)))
                new_p_x = p_x + (self.eta * ((eps_xi * q_i) - (2 * self.lam_2 * p_x)))
                new_b_x = b_x + (self.eta * ((eps_xi)       - (2 * self.lam_3 * b_x)))
                new_b_i = b_i + (self.eta * ((eps_xi)       - (2 * self.lam_4 * b_i)))

                Q[i_id]  = new_q_i
                P[u_id]  = new_p_x
                Bx[u_id] = new_b_x
                Bi[i_id] = new_b_i

            # Compute error for this iteration over the data.
            obj_t = self._objective_func(Q, P, Bx, Bi)
            self.objective_values_list.append(obj_t)
            et = time.time()
            print("Iteration %s: Time taken: %s, Objective value: %s" % ((t + 1), (et - st), obj_t))

        print("Training complete.")
        self.Q = Q
        self.P = P
        self.Bx = Bx
        self.Bi = Bi

        return self


    def predict(self, user_id, item_id):
        """
        Get predicted rating for a given user_id, item_id pair.

        """
        q_i = self.Q[item_id]
        p_x = self.P[user_id]
        b_x = self.Bx[user_id]
        b_i = self.Bi[item_id]
        mu = self.mu

        pred_rating = mu + b_x + b_i + np.dot(q_i, p_x)
        return pred_rating
