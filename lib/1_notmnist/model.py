import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import FunctionTransformer

RANDOM_STATE = np.random.RandomState(10)


def flat_image(X):
    return np.reshape(X, (X.shape[0], 28 * 28))


def model1():
    pipeline = make_pipeline(
        FunctionTransformer(flat_image, validate=False),
        ExtraTreesClassifier(n_estimators=10, verbose=1)
    )
    return pipeline


def model2():
    pipeline = make_pipeline(
        FunctionTransformer(flat_image, validate=False),
        LogisticRegression(verbose=1)
    )
    return pipeline
