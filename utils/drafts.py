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

#--------------------------------------------------------------------------------------------------------------------------------------------

def plot_graphs(results):

    total_class = len(results.keys())
    values_plot = {}

    # Para cada limite com classe.
    for n_class in results:
        values_plot[n_class] = {}
        # Para cada iteração.
        for it in results[n_class]:
            # Para cada representação.
            for rep in results[n_class][it]:
                if rep not in values_plot[n_class]:
                    values_plot[n_class][rep] = {}
                # Para cada classificador.
                for clf in results[n_class][it][rep]:
                    if clf not in values_plot[n_class][rep]:
                        values_plot[n_class][rep][clf] = []
                    mean_f1 = results[n_class][it][rep][clf]["mean_f1"]
                    values_plot[n_class][rep][clf].append(mean_f1)

    fig, axes = plt.subplots(4,2, figsize=(14,18))
    for n_class, index in zip(values_plot, range(total_class)):
        col = index % 2
        line = index // 2
        for rep in values_plot[n_class]:
            for clf in values_plot[n_class][rep]:
                y = values_plot[n_class][rep][clf]
                x = list(range(3, limit))
                label = rep+' '+clf
                axes[line, col].plot(x, y, label=label)
        axes[line, col].legend()
        axes[line, col].set_ylim((0,1))
        axes[line, col].set_xlabel("Number of Classes", fontsize=16)
        axes[line, col].set_ylabel("F1 (%)", fontsize=16)
        axes[line, col].set_title("Class with "+str(n_class), fontsize=18)
        axes[line, col].tick_params(axis='x', labelsize=16)
        axes[line, col].tick_params(axis='y', labelsize=16)
        axes[line, col].grid()
    fig.tight_layout()
    plt.savefig('graphs/f1_classes.pdf')


plot_graphs(results)
