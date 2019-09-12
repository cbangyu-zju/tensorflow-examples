import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import logging


class SampleGenerator(object):

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._data = []

    def run(self):
        weight = self._init_logistic_weight()
        self._logger.warning("%s weight: %s", len(weight), weight)
        samples = self._generate_sample(weight)
        self._logger.warning("sample: %s", samples[0])
        return samples

    def _init_logistic_weight(self, weight=None, num_weight=10):
        if not weight:
            weight = []
            for i in range(num_weight):
                weight.append(random.uniform(-1.0, 1.0))
            return weight
        return weight

    # y = 1/(1+exp(-wx))
    def _generate_sample(self, weight, num_sample=200):
        num_feature = len(weight) - 1
        samples = []
        for i in range(num_sample):
            x = []
            for j in range(num_feature):
                x.append(random.randint(-10, 10))
            y = 1/(1+ math.exp(weight[num_feature] + sum([weight[i] * x[i] for i in range(num_feature)])))
            if y > 0.5:
                y = 1.0
            else:
                y = 0.0
            samples.append((x, y))
        return samples


class LogisticModel(object):

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def train(self, sample):
        X_train, X_test, y_train, y_test = self._set_sample(sample)
        self._logger.warning("y_train: %s", y_train)
        self._logger.warning("y_test: %s", y_test)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        model = self._train(X_train_std, y_train)

        y_predict_result = model.predict_proba(X_test_std)  # 查看第一个测试样本属于各个类别的概率
        y_predict = []
        for predict in y_predict_result:
            if predict[0] > 0.5:
                y_predict.append(0.0)
            else:
                y_predict.append(1.0)
        self._logger.warning("y_predict: %s", y_predict)
        self._logger.warning("y_test: %s", list(y_test))

    def _train(self, X_train, Y_train):
        lr = LogisticRegression(C=1000.0, random_state=0)
        lr.fit(X_train, Y_train)
        return lr

    def _set_sample(self, sample=None):
        sample_0 = sample[0]

        num_feature = len(sample_0[0])
        num_sample = len(sample)
        X = np.zeros((num_sample, num_feature), dtype=float)
        Y = np.zeros((num_sample,), dtype=float)
        for i in range(num_sample):
            X[i, ] = np.asarray(sample[i][0])
            Y[i] = sample[i][1]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        return X_train, X_test, y_train, y_test


def main():
    sample_generator = SampleGenerator()
    sample = sample_generator.run()

    model = LogisticModel()
    model.train(sample)


if __name__ == '__main__':
    main()