import numpy as np
import matplotlib.pylab as plt
#matplotlib inline

class Numbers:
    """
    Class to store MNIST data
    """
    def __init__(self, location):

        import pickle, gzip

        # load data from file
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        # store for use later
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

data = Numbers("mnist.pklz")


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, X, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        from sklearn.neighbors import BallTree

        self._kdtree = BallTree(X)
        self._y = y
        self._k = k
        self._counts = self.label_counts()

    def label_counts(self):
        """
        Given the training labels, return a dictionary d where d[y] is
        the number of times that label y appears in the training set.
        """
        dict_label_count = {}
        for i in self._y:
            if i in dict_label_count:
                dict_label_count[i] += 1
            else:
                dict_label_count[i] = 1

        return dict_label_count

    def majority(self, neighbor_indices):
        """
        Given the indices of training examples, return the majority label. Break ties
        by choosing the tied label that appears most often in the training data.

        :param neighbor_indices: The indices of the k nearest neighbors
        """
        assert len(neighbor_indices) == self._k, "Did not get k neighbor indices"

        dict_label_freq = {}

        # Get the label and frequency of each k nearest neighbor

        for i in neighbor_indices:
            if self._y[i] in dict_label_freq:
                dict_label_freq[self._y[i]] = dict_label_freq[self._y[i]][0] + 1
            else:
                dict_label_freq[self._y[i]] = 1

        # Retrieve the max frequent label
        v = list(dict_label_freq.values())
        k = list(dict_label_freq.keys())
        max_freq = k[v.index(max(v))]
        max_labels = []

        for i in dict_label_freq:
            if dict_label_freq[i] == max_freq:
                max_labels.append(i)

        # In case of tie, return the labels which have more count frequency
        dict_label_count = {}

        if len(max_label) > 1:
            for i in max_label:
                dict_label_count[i] = self._count[i]

            # Retrieve the max count label
            v = list(dict_label_count.values())
            k = list(dict_label_count.keys())
            max_count = k[v.index(max(v))]
            max_count_label = []

            for i in dict_label_count:
                if dict_label_count[i] == max_count:
                    max_count_label.append(i)

            return max_count_label[0]

        else:
            return max_label[0]

    def classify(self, example):
        """
        Given an example, return the predicted label.

        :param example: A representation of an example in the same
        format as a row of the training data
        """
        dist, ind = self._kdtree.query(example, self._k)
        return self.majority(ind)

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a 2-dimensional
        numpy array of ints, C, where C[ii,jj] is the number of times an
        example with true label ii was labeled as jj.

        :param test_x: test data
        :param test_y: true test labels
        """
        C = np.zeros((10, 10), dtype=int)
        for xx, yy in zip(test_x, test_y):
            predicted = self.classify(xx)
            C[yy, predicted] = C[yy, predicted] + 1

        return C

    @staticmethod
    def accuracy(C):
        """
        Given a confusion matrix C, compute the accuracy of the underlying classifier.

        :param C: a confusion matrix
        """

        return np.sum(C.diagonal()) / C.sum()


import unittest


class TestKnn(unittest.TestCase):
    def setUp(self):
        self.x = np.array(
            [[2, 0], [4, 1], [6, 0], [1, 4], [2, 4], [2, 5], [4, 4], [0, 2], [3, 2], [4, 2], [5, 2], [5, 5]])
        self.y = np.array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1])
        self.knn = {}
        for ii in [1, 2, 3]:
            self.knn[ii] = Knearest(self.x, self.y, ii)

        self.queries = np.array([[1, 5], [0, 3], [6, 4]])

    def test0(self):
        """
        Test the label counter
        """
        self.assertEqual(self.knn[1]._counts[-1], 5)
        self.assertEqual(self.knn[1]._counts[1], 7)

    def test1(self):
        """
        Test 1NN
        """
        self.assertEqual(self.knn[1].classify(self.queries[0]), 1)
        self.assertEqual(self.knn[1].classify(self.queries[1]), -1)
        self.assertEqual(self.knn[1].classify(self.queries[2]), -1)

    def test2(self):
        """
        Test 2NN
        """
        self.assertEqual(self.knn[2].classify(self.queries[0]), 1)
        self.assertEqual(self.knn[2].classify(self.queries[1]), 1)
        self.assertEqual(self.knn[2].classify(self.queries[2]), 1)

    def test3(self):
        """
        Test 3NN
        """
        self.assertEqual(self.knn[3].classify(self.queries[0]), 1)
        self.assertEqual(self.knn[3].classify(self.queries[1]), 1)
        self.assertEqual(self.knn[3].classify(self.queries[2]), -1)


tests = TestKnn()
tests_to_run = unittest.TestLoader().loadTestsFromModule(tests)
unittest.TextTestRunner().run(tests_to_run)

