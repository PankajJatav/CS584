import numpy as np


class KMeans:

    def __init__(self, k, data, travel_centroids = []):
        self.k = k
        self.centroids = []
        self.travel_centroids = np.array(travel_centroids)
        self.distances = []
        self.data = data
        self.initialize_centroids()
        self.initialize_centroids = self.centroids
        self.calculate_distance_from_centroids()
        for i in range(1000):
            centroids = self.centroids
            self.calculate_distance_from_centroids()
            self.cal_new_centroids()
            if np.array_equal(self.centroids, centroids):
                break
        self.calculate_errors()

    def initialize_centroids(self):
        centroid_min = self.data.min()
        centroid_max = self.data.max()
        n_dims = self.data.shape[1]
        for centroid in range(self.k):
            # np.random.seed(42)
            centroid = np.random.uniform(centroid_min, centroid_max, n_dims)
            while np.round(centroid, decimals=3).tolist() in self.travel_centroids.tolist() or \
                    np.round(centroid, decimals=3).tolist() in np.round(self.centroids, decimals=3).tolist():
                # np.round(k_means.centroids, decimals=3).tolist()
                print("DUP")
                centroid = np.random.uniform(centroid_min, centroid_max, n_dims)
            self.centroids.append(centroid.tolist())

    def calculate_errors(self):
        errors = np.array([])
        for i in range(self.k):
            rows_indexes = np.where(self.clusters == i)
            if(len(rows_indexes[0]) == 0):
                continue
            rows = self.data.iloc[rows_indexes]
            error = np.sum( (rows - self.centroids[i]).values ** 2 )
            errors = np.append(errors, error)
        self.errors = errors

    def calculate_distance_from_centroids(self):
        distances = []
        for centroid in self.centroids:
            distances.append(np.sqrt(np.sum((self.data - centroid) ** 2, axis=1)))
        self.distances = np.array(distances).transpose()
        # print(cosine_similarity(self.data, self.centroids))
        # print(self.distances)
        # self.distances = 2 * (1 - cosine_similarity(self.data, self.centroids))
        self.clusters = np.argmin(self.distances, axis=1)

    def cal_new_centroids(self):
        centroids = []
        for i in range(self.k):
            rows_indexes = np.where( self.clusters == i)
            if ( len(rows_indexes[0]) == 0 ):
                centroids.append(self.centroids[i])
                continue
            rows = self.data.iloc[rows_indexes]
            centroids.append(np.sum(rows).values / len(rows_indexes[0]))
        self.centroids = centroids