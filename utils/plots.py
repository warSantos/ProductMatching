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

    index = 0
    fig, ax = plt.subplots(figsize=(20,18))
    for n_class in values_plot:
        for rep in values_plot[n_class]:
            ax = plt.subplot(4, 4, index+1)
            for clf in values_plot[n_class][rep]:
                y = values_plot[n_class][rep][clf]
                x = list(range(3, limit))
                label = rep[:3]+' '+clf[:3]
                plt.plot(x, y, label=label)
            plt.grid()
            plt.ylim((0,1))
            plt.xlabel("Number of Classes", fontsize=16)
            plt.ylabel("F1 (%)", fontsize=16)
            plt.title("Class with "+str(n_class), fontsize=18)
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)
            plt.legend()
            index += 1

    fig.tight_layout()
    plt.savefig('graphs/f1_classes.pdf')

def plot_dists(dists, df):

    cat_ids = list(dists.keys())
    total = len(cat_ids)
    fig, ax = plt.subplots(figsize=(18,14))
    for i in range(total):
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
