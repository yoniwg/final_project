
from numpy import dot, multiply
from numpy.core._multiarray_umath import subtract
from numpy.ma import exp
from scipy.sparse import spmatrix


class Trainer:
    def __init__(self, classes_vec, num_features, learning_rate) -> None:
        self._classes_vec = [] # classes_vec
        self._n = num_features
        self._weight_mat = spmatrix((len(self._classes_vec), num_features))
        self._bias_vec = [0] * len(self._classes_vec)
        self._l_rate = learning_rate

    def _predict_cls(self, features_vec, cls_idx):
        exp_c = exp(dot(features_vec, self._weight_mat[cls_idx]) + self._bias_vec[cls_idx])
        exp_sum = sum([exp(dot(features_vec, self._weight_mat[idx]) + self._bias_vec[idx])
                       for idx in range(len(self._classes_vec))])
        return exp_c / exp_sum

    def consume_x_y(self, features_vec, cls):
        cls_idx = self._classes_vec.index(cls)
        delta_mat = self._calc_delta(features_vec, cls_idx)
        self._weight_mat = subtract(self._weight_mat, multiply(self._l_rate, delta_mat))

    def _calc_delta(self, features_vec, cls_idx):
        return [self.calc_gradient(features_vec, idx, idx == cls_idx) for idx in range(len(self._classes_vec))]

    def calc_gradient(self, features_vec, idx, y):
        return -1 * multiply((y - self._predict_cls(features_vec, idx)), features_vec)

