import numpy as np
import argparse
import pickle
import os
import pandas as pd
from collaborative_filtering import CollaborativeFiltering


def recommend(user_id):
    items = pd.read_csv(os.path.join(
        args.data_dir, 'movie.csv'), usecols=[0, 1]).values
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.isfile(model_path):
        print("Model at {} doesn't exist".format(model_path))
        return
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Collaborative filtering
    if isinstance(model, CollaborativeFiltering):
        print("Load collaborative filtering model")
        print("Processing ...")
        ratings = np.zeros_like(items)
        for i, movie_id in enumerate(items[:, 0]):
            ratings[i, 0] = model.predict(user_id, movie_id - 1)
            ratings[i, 1] = items[i, 1]
        ratings = ratings[ratings[:, 0].argsort()]
        return ratings[-args.topk:][::-1]

    elif isinstance(model, dict):
        print("Load content-based model")
        print("Processing ...")
        movie_oh = np.load("data/movie_oh.npy")
        movie_wv = np.load("data/movie_wv.npy")
        movie_rep = np.concatenate((movie_oh, movie_wv), axis=1)
        ratings = np.copy(items)
        ratings[:, 0] = model[user_id+1].predict(movie_rep)
        ratings = ratings[ratings[:, 0].argsort()]
        return ratings[-args.topk:, :][::-1]

    else:
        print("Invalid model!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_id', type=int, default=-1,
                        help='user to recommend')
    parser.add_argument('--model_name', type=str, default='cf_40.pkl',
                        help='model to predict')
    parser.add_argument('--topk', type=int, default=5,
                        help='number of recommendations')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='model directory')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory')
    args = parser.parse_args()

    if args.user_id >= 1000 or args.user_id < 0:
        raise ValueError(
            "User id must be between 0 and 999, use --user_id <int>")
    recommendation = recommend(args.user_id)
    print("Top {} recommendation for user {}".format(args.topk, args.user_id))
    for i in range(args.topk):
        print("{}. {}".format(i+1, recommendation[i, 1],))
