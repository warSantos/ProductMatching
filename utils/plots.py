import numpy as np
import matplotlib.pyplot as plt

def plot_acc_classes(results, limit):

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

def plot_dists(dists, df):

    cat_ids = list(dists.keys())
    total = len(cat_ids)
    fig, ax = plt.subplots(figsize=(14,10))
    for i in range(total):
        col = i % 2
        line = i // 2
        ax = plt.subplot(3, 4, i+1)
        x = list(range(dists[cat_ids[i]].shape[0]))
        y = np.copy(dists[cat_ids[i]])
        y.sort()
        label_cat = df[df.cat_id == cat_ids[i]].iloc[0]["cat_title"]
        plt.plot(x, y, label=label_cat)
        plt.grid()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Comparasons", fontsize=16)
        plt.ylabel("Distances", fontsize=16)
        plt.tight_layout()
        plt.legend()
