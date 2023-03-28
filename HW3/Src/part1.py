import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from utils import run_k_means, run_k_means_bi, plot_graph

df = pd.read_csv("./data/1632506309_3197203_iris_new_data.csv", names=['w', 'x', 'y', 'z'], delim_whitespace=True)
tsne_data = pd.DataFrame(TSNE(2).fit_transform(df))

tsne_data = tsne_data / 255

def run_part_1(choice):
    if choice == 1:
        clusters = run_k_means(3, tsne_data, 100)
        np.savetxt('./data/out-final.dat', clusters + 1, delimiter=',', fmt='%i')
        print("File save as out-final.dat")
    elif choice == 2:
        clusters = run_k_means_bi(3, tsne_data, 100)
        np.savetxt('./data/out-final-bi.dat', clusters + 1, delimiter=',', fmt='%i')
        print("File save as out-final-bi.dat")
    elif choice == 3:
        plot_graph(df)
        print("File save as fig.jpeg")




#