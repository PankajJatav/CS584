import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def get_data(cross_val=False):

    test_data = pd.read_csv("./data/train/test.dat", sep=" ")
    train_data = pd.read_csv("./data/train/train.dat", sep=" ")
    genre_arr = pd.read_csv("./data/train/movie_genres.dat", sep="\t", skiprows=1, names=['movieID', 'genre'])
    movie_tag_arr = pd.read_csv("./data/train/movie_tags.dat", sep="\t")
    actor_arr = pd.read_csv("./data/train/movie_actors.dat", encoding='iso-8859-1', sep="\t")
    actor_arr = np.delete(actor_arr.values, 2, 1)
    dir_arr = pd.read_csv("./data/train/movie_directors.dat", encoding='iso-8859-1', sep="\t")

    if (cross_val == True):
        train_data, test_data = train_test_split(train_data, test_size=0.2)


    users = np.unique(np.concatenate((train_data.userID.values, test_data.userID.values)))
    users = dict(zip(users, np.arange(users.size)))

    movies = np.unique(np.concatenate((train_data.movieID.values, test_data.values[:, 1], movie_tag_arr.movieID.values, genre_arr.movieID.values)))
    movies = dict(zip(movies, np.arange(movies.size)))

    genres = np.unique(genre_arr.values[:, 1])
    genres = dict(zip(genres, np.arange(genres.size)))

    actors = np.unique(actor_arr[:, 1])
    actors = dict(zip(actors, np.arange(actors.size)))
    directors = np.unique(dir_arr.values[:, 1])
    directors = dict(zip(directors, np.arange(directors.size)))

    # print(genre_arr)

    return train_data, test_data, users, movies, movie_tag_arr, actors, genres, directors, genre_arr, actor_arr, dir_arr


def to_sparse(records, movie_dic=None, features_dict=None, bool=False):
    data = []
    i = []
    j = []

    for record in records:
        if movie_dic==None:
            i.append(record[0])
        elif record[0] in movie_dic:
            i.append(movie_dic[record[0]])
        else:
            i.append(record[0])

        if features_dict==None:
            j.append(record[1])
        elif record[1] in features_dict:
            j.append(features_dict[record[1]])
        else:
            j.append(record[1])

        if bool==False:
            data.append(record[2])
        else:
            data.append(1)

    return sp.coo_matrix((data, (i,j)), shape = [np.amax(i)+1, np.amax(j)+1]).tocsr()



def rec_knn(users, movies, train_sparse, test_data, feature_similarities, k=20):
    test_y = []

    user_cos_similarity = cosine_similarity(train_sparse, train_sparse,
                                            dense_output=False)
    movie_cos_similarity = cosine_similarity(train_sparse.transpose(),
                                             train_sparse.transpose(),
                                             dense_output=False)

    for index, line in enumerate(test_data):

        user = users[line[0]]
        movie = movies[line[1]]
        user_neighbor_indices = user_cos_similarity[user].todense()
        user_neighbor_indices = np.array(user_neighbor_indices).flatten()
        user_neighbor_indices = user_neighbor_indices.argpartition(-k - 1)[-k - 1:]
        user_neighbor_indices = user_neighbor_indices[np.where(user_neighbor_indices != user)]

        user_ratings = []
        for userID in user_neighbor_indices:
            user_ratings.append(train_sparse[userID, movie])

        user_ratings = np.array(user_ratings)
        user_ratings = user_ratings[np.nonzero(user_ratings)]

        if user_ratings.size == 0:
            user_ratings = train_sparse.tocsc()[user, :].data
            try:
                user_rating = user_ratings.mean()
            except:
                user_rating = 0
        else:
            user_rating = user_ratings.mean()
        movie_neighbor_indices = movie_cos_similarity[movie].todense()
        movie_neighbor_indices = np.array(movie_neighbor_indices).flatten()
        movie_neighbor_indices = movie_neighbor_indices.argpartition(-k - 1)[-k - 1:]
        movie_neighbor_indices = movie_neighbor_indices[np.where(movie_neighbor_indices != movie)]
        movie_ratings = []
        for movieID in movie_neighbor_indices:
            movie_ratings.append(train_sparse[user, movieID])

        movie_ratings = np.array(movie_ratings)
        movie_ratings = movie_ratings[np.nonzero(movie_ratings)]

        if movie_ratings.size == 0:

            movie_ratings = train_sparse[:, movie].data
            movie_rating = movie_ratings.mean()

            if (np.isnan(movie_rating)):
                movie_rating = 0
        else:
            movie_rating = movie_ratings.mean()

        rating = [user_rating, movie_rating]
        for feature_similarity in feature_similarities:

            feature_neighbor_indices = feature_similarity[movie].todense()
            feature_neighbor_indices = np.array(feature_neighbor_indices).flatten()
            feature_neighbor_indices = feature_neighbor_indices.argpartition(-k - 1)[-k - 1:]
            feature_neighbor_indices = feature_neighbor_indices[np.where(feature_neighbor_indices != movie)]
            new_ratings = train_sparse[:, feature_neighbor_indices]
            new_ratings = np.true_divide(new_ratings.sum(0), (new_ratings != 0).sum(0))
            new_ratings = np.nanmean(new_ratings)
            rating.append(new_ratings)

        ratings = np.array(rating)

        ratings = ratings[np.nonzero(ratings)]

        if ratings.size == 0:
            new_rating = 3.5
        else:
            new_rating = round(np.mean(ratings), 1)

        test_y.append(new_rating)

    return np.array(test_y)