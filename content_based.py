import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
import argparse
import os


def load_data():
    with open(os.path.join(args.data_dir, 'train_oh.pickle'), 'rb') as handle:
        X_train_oh, y_train_oh = pickle.load(handle)

    with open(os.path.join(args.data_dir, 'test_oh.pickle'), 'rb') as handle:
        X_test_oh, y_test_oh = pickle.load(handle)

    with open(os.path.join(args.data_dir, 'train_wv.pickle'), 'rb') as handle:
        X_train_wv, y_train_wv = pickle.load(handle)

    with open(os.path.join(args.data_dir, 'test_wv.pickle'), 'rb') as handle:
        X_test_wv, y_test_wv = pickle.load(handle)

    X_train, X_test = {}, {}
    y_train = y_train_oh
    y_test = y_test_oh
    for user in X_train_oh.keys():
        X_train[user] = np.concatenate(
            (X_train_oh[user], X_train_wv[user]), axis=1)
    for user in X_test_oh.keys():
        X_test[user] = np.concatenate(
            (X_test_oh[user], X_test_wv[user]), axis=1)
    return X_train, y_train, X_test, y_test


def train(X_train, y_train):
    model = {}
    for user in X_train.keys():
        model[user] = KernelRidge(kernel='laplacian', alpha=0.231)
        model[user].fit(X_train[user], y_train[user])
    return model


def test(model, X_test, y_test):
    rmse = 0
    cnt = 0
    for user in X_test.keys():
        y_pred = model[user].predict(X_test[user])
        rmse += ((y_pred - y_test[user]) ** 2).sum(axis=0)
        cnt += y_test[user].shape[0]

    return np.sqrt(rmse/cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Params for content based')
    parser.add_argument('--save_name', type=str, default='cb.pkl',
                        help='name of saved model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory')
    args = parser.parse_args()
    print("-----------loading data------------")
    X_train, y_train, X_test, y_test = load_data()

    print("\nTrainging content-based models")
    models = train(X_train, y_train)
    print("RMSE on test set:", test(models, X_test, y_test))
    print("Saving model to 'models/cb.pkl")
    with open('models/' + args.save_name, 'wb') as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
