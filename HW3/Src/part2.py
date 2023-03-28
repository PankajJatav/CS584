import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from utils import run_k_means, run_k_means_bi, plot_k_vs_sse, plot_data_graph

df = pd.read_csv("./data/1632506945_4051344_image_new_test.csv", header=None)

tsne_data = pd.DataFrame(TSNE(2).fit_transform(df))

# df = df.loc[:, (df != 0).any(axis=0)]
# tsne_data = tsne_data / tsne_data.max()

def run_part_2(choice):
    if choice == 1:
        clusters = run_k_means(10, tsne_data, 10)
        np.savetxt('./data/out-final-image.dat', clusters + 1, delimiter=',', fmt='%i')
        print("File save as out-final-image.dat")

    elif choice == 2:
        clusters = run_k_means_bi(10, tsne_data, 1000)
        np.savetxt('./data/out-final-bi-image.dat', clusters + 1, delimiter=',', fmt='%i')
        print("File save as out-final-bi-image.dat")

    elif choice == 3:
        plot_k_vs_sse(tsne_data)
        # plot_data_graph(tsne_data)
        print("File save as k-e-rel.jpeg")