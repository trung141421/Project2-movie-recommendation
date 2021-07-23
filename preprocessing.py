import gensim.downloader as api
import numpy as np
import pandas as pd
import re
import pickle
import os


N_MOVIES = 8621


def preprocess_data(filename, movie_vecs):
    X_train = pd.read_csv(filename, usecols=[0, 1, 2], engine='python').values
    X = {}
    y = {}
    users = X_train[:, 0].astype(np.int32)
    items = X_train[:, 1].astype(np.int32)
    ratings = X_train[:, 2].astype(np.float64)

    list_user = set(users)

    for u in list_user:
        ids = np.where(users == u)[0]
        items_id = items[ids] - 1
        X[u] = movie_vecs[items_id]
        y[u] = ratings[ids]
    return X, y


def w2v_transform():
    if os.path.isfile("data/train_wv.pickle") and os.path.isfile("data/test_wv.pickle"):
        print("Preprocessed w2v-transform data exist in /data")
        return

    print("Load pretrained w2v: ....\n")
    glove_vectors = api.load('glove-wiki-gigaword-100')
    tag = pd.read_csv("data/genome_tag.csv")
    movie_tag = pd.read_csv("data/genome_scores.csv")
    map_id = pd.read_csv("data/map_id.csv")
    movie = pd.read_csv('data/movie.csv')

    map_dict = {}
    for pair in map_id.values:
        key, value = pair
        map_dict[key] = value

    movie_tag = movie_tag.groupby('movieId').apply(
        lambda grp: grp.nlargest(10, columns=['relevance']))
    movie_tag['movieId'] = movie_tag['movieId'].map(map_dict)
    movie_tag = movie_tag[movie_tag.movieId.isin(movie.values[:, 0])]

    vector = np.zeros((1128, 100))
    for i, v in enumerate(tag.values[:, 1]):
        vector[i] = np.mean(
            [glove_vectors.get_vector(w)
                for w in re.split('[-:()/! ]', v) if w != ''], axis=0)

    vec_movies = np.zeros((N_MOVIES, 300))
    n_tags = 10

    for i in range(N_MOVIES):
        tmp_vec = np.zeros((n_tags, 100))
        for j in range(n_tags):
            tmp_vec[j] = vector[movie_tag.values[i*n_tags+j, 1].astype(np.uint8)] \
                * movie_tag.values[i*n_tags+j, 2]
        vec_movies[i] = np.concatenate([
            np.amax(tmp_vec, axis=0), np.amin(
                tmp_vec, axis=0), np.mean(tmp_vec, axis=0),
        ], axis=0)

    X_train_wv, y_train_wv = preprocess_data("data/train.csv", vec_movies)
    X_test_wv, y_test_wv = preprocess_data("data/test.csv", vec_movies)

    np.save("data/movie_wv.npy", vec_movies)
    print("Saving train data to ... '/data/train_wv.pickle'")
    with open('data/train_wv.pickle', 'wb') as handle:
        pickle.dump((X_train_wv, y_train_wv), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print("Saving test data to ... '/data/test_wv.pickle'")
    with open('data/test_wv.pickle', 'wb') as handle:
        pickle.dump((X_test_wv, y_test_wv), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def one_hot_transform():
    if os.path.isfile("data/train_oh.pickle") and os.path.isfile("data/test_oh.pickle"):
        print("Preprocessed oh-transform data exist in /data")
        return
    usecols = list(range(2, 22))
    one_hot = pd.read_csv("data/movie.csv", usecols=usecols).values
    X_train_oh, y_train_oh = preprocess_data("data/train.csv", one_hot)
    X_test_oh, y_test_oh = preprocess_data("data/test.csv", one_hot)

    np.save("data/movie_oh.npy", one_hot)
    print("Saving train data to ... '/data/train_oh.pickle'")
    with open('data/train_oh.pickle', 'wb') as handle:
        pickle.dump((X_train_oh, y_train_oh), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print("Saving test data to ... '/data/test_oh.pickle'")
    with open('data/test_oh.pickle', 'wb') as handle:
        pickle.dump((X_test_oh, y_test_oh), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    w2v_transform()
    one_hot_transform()
