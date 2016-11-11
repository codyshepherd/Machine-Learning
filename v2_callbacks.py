from v2_data import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


class Backend:

    def __init__(self, filename='conn.log', act='logistic', slv='lbfgs', alph=1e-5, hls=(5,2), rnd=1):
        self.clf = MLPClassifier(activation=act, solver=slv, alpha=alph, hidden_layer_sizes=hls, random_state=rnd)
        self.scaler = StandardScaler()

        self.FILENAME=filename
        self.training_data = []
        self.predict_data = []
        self.anomalies = []
        self.start = 0

    def nn_train(self, num):
        print "nn_train called. arg: ", num
        with Data(self.FILENAME, Doc_t.BRO, self.start) as d:
            for series in d.get_lines(num):
                self.training_data.append(series.get_values())
            self.start = d.tell()

        """
        Put control structure here to look at training set and assign classes to 
        samples.
        """

        Y = [0] * len(self.training_data)

        self.clf.fit(self.training_data,Y)

    def nn_predict(self, num):
        print "predict"
        with Data(self.FILENAME, Doc_t.BRO, self.start) as d:
            for series in d.get_lines(num):
                self.predict_data.append(series.get_values())

        results = self.clf.predict(self.predict_data)

        self.anomalies.append("test")
        for i, item in enumerate(results):
            if item==1:
                self.anomalies.append(self.predict_data[i])

    def nn_results(self):
        for item in self.anomalies:
            yield item
        

"""
back = Backend()
back.nn_train(10008)
back.nn_predict(10000)
"""
