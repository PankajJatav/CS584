import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_data, to_sparse, rec_knn

def main():
    train_data, test_data, users, movies, movie_tag_arr, actors, genres, directors, genre_arr, actor_arr, dir_arr = get_data()
    train_data = to_sparse(train_data.values, users, movies)
    genre_arr = to_sparse(genre_arr.values, movies, genres, bool=True)
    movie_tag_arr = to_sparse(movie_tag_arr.values, movies, None)
    actor_arr = to_sparse(actor_arr, movies, actors)
    dir_arr = to_sparse(dir_arr.values, movies, directors, bool=True)

    tfidf = TfidfTransformer()
    movie_tag_arr = tfidf.fit_transform(movie_tag_arr)

    tag_cos_similarity = cosine_similarity(movie_tag_arr, movie_tag_arr, dense_output=False)
    submission = rec_knn(users, movies, train_data, test_data.values, [tag_cos_similarity] , k=351)
    np.savetxt("./output/tag-based.dat", submission, fmt='%s')
    print("File save at ./output/tag-based.dat")

    genre_cos_similarity = cosine_similarity(genre_arr, genre_arr, dense_output=False)
    submission = rec_knn(users, movies, train_data, test_data.values, [genre_cos_similarity] , k=351)
    np.savetxt("./output/genre-based.dat", submission, fmt='%s')
    print("File save at ./output/genre-based.dat")

    #86
    actor_cos_similarity = cosine_similarity(actor_arr, actor_arr, dense_output=False)
    submission = rec_knn(users, movies, train_data, test_data.values, [actor_cos_similarity] , k=351)
    np.savetxt("./output/actor-based.dat", submission, fmt='%s')
    print("File save at ./output/actor-based.dat")


    director_cos_similarity = cosine_similarity(dir_arr, dir_arr, dense_output=False)
    submission = rec_knn(users, movies, train_data, test_data.values, [director_cos_similarity] , k=351)
    np.savetxt("./output/dir-based.dat", submission, fmt='%s')
    print("File save at ./output/dir-based.dat")

    #88
    submission = rec_knn(users, movies, train_data, test_data.values, [tag_cos_similarity, genre_cos_similarity], k=351)
    np.savetxt("./output/tan-gen-based.dat", submission, fmt='%s')
    print("File save at ./output/tan-gen-based.dat")

    # 86
    submission = rec_knn(users, movies, train_data, test_data.values, [actor_cos_similarity, director_cos_similarity], k=351)
    np.savetxt("./output/act-dir-based.dat", submission, fmt='%s')
    print("File save at ./output/act-dir-based.dat")

    #92
    submission = rec_knn(users, movies, train_data, test_data.values, [tag_cos_similarity, genre_cos_similarity, actor_cos_similarity, director_cos_similarity], k=351)
    np.savetxt("./output/all-fea-based.dat", submission, fmt='%s')
    print("File save at ./output/all-fea-based.dat")