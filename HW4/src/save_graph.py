import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from utils import get_data
import matplotlib.pyplot as plt

def save_graph():
    train_data, test_data, users, movies, movie_tag_arr, actors, genres, directors, genre_arr, actor_arr, dir_arr = get_data()
    print(train_data)

    row = train_data.userID.values
    col = train_data.movieID.values
    data = train_data.rating.values

    row_data= np.unique(row)
    row_data={row_data[i] : i for i in range(len(row_data))}

    col_data = np.unique(col)
    col_data = {col_data[i]: i for i in range(len(col_data))}

    row_index = [row_data.get(i) for i in row]
    col_index = [col_data.get(i) for i in col]

    mtr= csr_matrix((data,(row_index,col_index)))

    dense=np.asarray(mtr.todense())
    nz=normalize(dense,axis=1)
    plt.imshow(dense,cmap='hot',interpolation='nearest')
    plt.savefig('./plots/fig.jpeg')
    print('File save as fig.jpeg')