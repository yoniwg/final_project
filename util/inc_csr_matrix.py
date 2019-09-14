import array
import numpy as np
import scipy.sparse as sp


class IncrementalCSRMatrix:

    def __init__(self, col_num):

        self.cols_num = col_num

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array('i')
        self.rows_num = 0

    def append_row(self, full_in_row):
        self.rows_num += 1
        for i in full_in_row:
            self.append(self.rows_num - 1, i, 1)

    def append(self, i, j, v):

        assert i < self.rows_num and j < self.cols_num, 'Index out of bounds'

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def to_csr(self):
        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=np.int32)

        return sp.csr_matrix((data, (rows, cols)),
                             shape=(self.rows_num, self.cols_num))
