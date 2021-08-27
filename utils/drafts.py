from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

r = np.random
seed = r.randint(0, 2147483647 * 2)

"""
target = np.zeros((len(df), len(set(df.cluster_id))))
for i in df.cluster_id:
    target[i] = 1

X_train, X_test, y_train, y_test = train_test_split(feats, target)# df.cluster_id)
"""

X_train, X_test, y_train, y_test = train_test_split(feats, df.cluster_id)

#clf = MultiOutputClassifier(LogisticRegression(max_iter=400, multi_class='multinomial', n_jobs=10, random_state=seed))
#clf = KNeighborsClassifier(n_jobs=10)
#clf = RandomForestRegressor(random_state=seed, n_jobs=10)
clf = LogisticRegression(max_iter=400, multi_class='multinomial', n_jobs=10, random_state=seed)

print("Treinando...")
clf.fit(X_train, y_train)
print("Predizendo...")
y_pred = clf.predict(X_test)
y_pred


#--------------------------------------------------------------------------------------------------------------------------------------------

from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance

r = np.random
seed = r.randint(0, 2147483647 * 2)

lgr = LogisticRegression(max_iter=400, multi_class='multinomial', n_jobs=10, random_state=seed)
#clf = BinaryRelevance(classifier=lgr, require_dense=[False, True])
#clf = MLkNN(k=3)

X_train, X_test, y_train, y_test = train_test_split(feats, np.array(df.cluster_id))

print("Treinando...")
clf.fit(X_train, y_train)
print("Predizendo...")
y_pred = clf.predict(X_test)
y_pred