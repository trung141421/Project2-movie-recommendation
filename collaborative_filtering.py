import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import pickle


class CollaborativeFiltering:
    def __init__(self, _data, dist_func=cosine_similarity, uuCF=1, num_neighbors=30):
        """ hàm khởi tạo (constructor) của class
            num_neighbors: số lượng hàng xóm để xét ấn 
            dist_func: hàm tính khoảng cách. Default là khoảng cách coisn
            uuCF: sử dụng user-user hay item-item (Default là 1 tức là user-user)
        """
        super().__init__()
        self._data = _data if uuCF else _data[:, [1, 0, 2]]
        self.dist_func = dist_func
        self.uuCF = uuCF
        self.utility_data = None
        self.num_neighbors = num_neighbors

    def _normalize_data(self):
        """Normalize dữ liệu _data
        Điền vào các rating còn thiếu của người dùng
        Trừ các rating đi giá trị trung bình(mean)
        Args:
            utility_data (sparse matrix): 3 trận thưa 3 chiều (user_id, movie_id, rating)
        Return: Ma trận đã được chuẩn hóa
        """

        max_id_user = np.max(self._data[:, 0])
        max_id_item = np.max(self._data[:, 1])

        user = np.array(self._data[:, 0], dtype=int)
        item = np.array(self._data[:, 1], dtype=int)
        rating = np.array(self._data[:, 2], dtype=np.float32)

        self.mu = np.zeros(max_id_user + 1)
        # tìm các rating của user
        for n in range(max_id_user + 1):
            id_user = np.where(user == n)[0].astype(np.int32)   # tìm user == n
            # tính giá trị trung bình(mean) của rating
            rating_mean = np.mean(rating[id_user])

            if np.isnan(rating_mean):
                rating_mean = 0
            self.mu[n] = rating_mean

            rating[id_user] = np.array(
                rating[id_user] - self.mu[n], dtype=np.float32)
            # print(rating[id_user])

        self.utility_data = sparse.csr_matrix(
            (rating, (user, item)), dtype=np.float32)

    def fit(self):
        self._normalize_data()
        self.matrix_cosine = self.dist_func(
            self.utility_data, self.utility_data)

    def predict(self, user, item):
        # Tìm tất cả user đã rate cho item
        item_id = np.where(self._data[:, 1] == item)[0].astype(np.int32)
        user_rate_item = self._data[item_id, 0].astype(np.int32)

        # Tìm những người dùng tương đồng với user
        sim = self.matrix_cosine[user, user_rate_item]

        # Tìm num_neighbors gần nhất với user
        index_nei_nearest = np.argsort(sim)[-self.num_neighbors:]
        nei_nearest = sim[index_nei_nearest]

        r = self.utility_data[user_rate_item[index_nei_nearest], item]

        return (nei_nearest*r)[0]/(np.abs(nei_nearest).sum() + 1e-8) + self.mu[user]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Params for collaborative filtering')
    parser.add_argument('-k', '--num_neighbors', type=int, default=30,
                        help='number of neighbors')
    parser.add_argument('--uu', type=bool, default=True,
                        help='user-user collaborative filtering')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory')
    args = parser.parse_args()

    r_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    train_data = pd.read_csv(os.path.join(args.data_dir, 'train.csv')).values
    test_data = pd.read_csv(os.path.join(args.data_dir, 'test.csv')).values

    train_data[:, :2] -= 1  # Index starts from 0
    test_data[:, :2] -= 1  # Index starts from 0

    cf = CollaborativeFiltering(
        train_data, uuCF=args.uu, num_neighbors=args.num_neighbors)
    cf.fit()

    print("Training finished")
    print("Model: Collaborative filtering")
    print("User-user: {}".format(args.uu))
    print("Num neighbors: {}\n".format(args.num_neighbors))

    print("Start predicting")
    num_test = test_data.shape[0]
    SE = 0
    for n in range(num_test):
        pred = cf.predict(test_data[n, 0], test_data[n, 1])
        SE += (pred - test_data[n, 2]) ** 2

    RMSE = np.sqrt(SE / num_test)
    print("RMSE on test set: {}".format(RMSE))
    print("Saving model to '/models/cf_{}.pkl'".format(args.num_neighbors))
    with open('models/cf_{}.pkl'.format(args.num_neighbors), 'wb') as f:
        pickle.dump(cf, f, protocol=pickle.HIGHEST_PROTOCOL)
