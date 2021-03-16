from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

import pickle
import numpy as np
import re

import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def scaler(data):
	return stats.zscore(np.array(d), ddof=1)


def term_vectorizer_fit(data):
	tf = TfidfVectorizer(stop_words='english')
	clean = [re.sub("[^A-Za-z,.']", " ", x) for x in data]

	tf.fit(clean)
	return tf

def vectorizer_transform(data, file):
	vectorizer = pickle.load(open(dir_path + "\\" + file, "rb"))
	clean = [re.sub("[^A-Za-z,.']", " ", x) for x in data]

	return vectorizer.transform(clean)
