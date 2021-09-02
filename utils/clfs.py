import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.base import clone


class Clfs():

    # Faz o holdout com vários classficadores.
    def holdout(self, clfs, X, Y, test_size=0.2, clone=True):

        scores = {}
        # Separando o dado em treino e teste.
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        # Classificando o dado.
        for c in clfs:
            if clone:
                clf = clone(clfs[c])
            else:
                clf = clfs[c]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # Computando F1-Score.
            f1 = f1_score(y_test, y_pred, average="macro")
            acc = accuracy_score(y_test, y_pred)
            scores[c] = {}
            scores[c]["f1"] = f1
            scores[c]["acc"] = acc
        
        return scores

    def fast_avaliation(self, classifiers, X, Y, random=True, n_folds=5, n_jobs=3):
        
        r = np.random
        if random:
            seed = r.randint(0, 2147483647 * 2)
        else:
            seed = 42
        
        estimators = {}
        for alg in classifiers:
            # Clonando o classificador.
            clf = clone(classifiers[alg])
            scores = cross_val_score(clf, X, Y, cv=n_folds, n_jobs=n_jobs)
            estimators[alg] = {}
            estimators[alg]["scores"] = scores.tolist()
            estimators[alg]["mean_f1"] = np.mean(scores)
            estimators[alg]["std_f1"] = np.std(scores)
        return estimators
        

    def avaliation(self, classifiers, features, target, random=True):

        r = np.random
        if random:
            seed = r.randint(0, 2147483647 * 2)
        else:
            seed = 42

        results = {}
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        estimators = {}
        for alg in classifiers:    
            # Validação cruzada.
            for train_index, test_index in kf.split(features):
                X_train, X_test = features[train_index], features[test_index]
                y_train, y_test = target[train_index], target[test_index]
                # Clonando o classificador.
                clf = clone(classifiers[alg])
                # Predizendo 
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if alg not in estimators:
                    estimators[alg] = {}
                    estimators[alg]["accs"] = []
                    estimators[alg]["f1s"] = []
                estimators[alg]["accs"].append(accuracy_score(y_test, y_pred))
                estimators[alg]["f1s"].append(f1_score(y_test, y_pred, average="macro"))
        
            estimators[alg]["accs"] = np.array(estimators[alg]["accs"])
            estimators[alg]["f1s"] = np.array(estimators[alg]["f1s"])
            estimators[alg]["mean_accs"] = np.mean(estimators[alg]["accs"])
            estimators[alg]["mean_f1"] = np.mean(estimators[alg]["f1s"])
            estimators[alg]["std_accs"] = np.std(estimators[alg]["accs"])
            estimators[alg]["std_f1"] = np.std(estimators[alg]["f1s"])
        return estimators