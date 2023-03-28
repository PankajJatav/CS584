import numpy as np
from matplotlib import pyplot as plt
from k_means import KMeans


def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def run_k_means(k, df, itr):
    travel_centroids = []
    k_means = KMeans(k, df, travel_centroids)
    travel_centroids = travel_centroids + np.round(k_means.initialize_centroids, decimals=3).tolist()
    for i in range(itr):
        new_k_means = KMeans(k, df, travel_centroids)
        travel_centroids = travel_centroids + np.round(new_k_means.initialize_centroids, decimals=3).tolist()
        if (np.sum(k_means.errors) > np.sum(new_k_means.errors)):
            k_means = new_k_means

    return k_means.clusters

def run_k_means_for_plot(k, df, itr):
    travel_centroids = []
    k_means = KMeans(k, df, travel_centroids)
    travel_centroids = travel_centroids + np.round(k_means.initialize_centroids, decimals=3).tolist()
    for i in range(itr):
        new_k_means = KMeans(k, df, travel_centroids)
        travel_centroids = travel_centroids + np.round(new_k_means.initialize_centroids, decimals=3).tolist()
        if (np.sum(k_means.errors) > np.sum(new_k_means.errors)):
            k_means = new_k_means

    return k_means

def run_k_means_bi(k, df, itr):

    local_df = df
    clusters = []
    for i in range(k-1):
        k_means = run_k_means(2, local_df, itr)
        if len(clusters) == 0:
            clusters = clusters
        else:
            new_cluster = np.where(clusters == 0, ind[0], clusters)
            new_cluster = np.where(new_cluster == 1, i+1, new_cluster)
            np.put(clusters, max_c_index, new_cluster)
        freq = np.array(np.unique(clusters, return_counts=True)).T
        ind = np.unravel_index(np.argmax(freq, axis=None), freq.shape)
        max_c_index = np.where(k_means.clusters == ind[0])[0]
        local_df = df.iloc[max_c_index]
        local_df = local_df / local_df.max()

    return clusters


def plot_graph(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = df['w']
    y = df['x']
    z = df['y']
    c = df['z']

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.savefig('fig.jpeg')

def plot_k_vs_sse(df):
    error = []
    for i in range(2, 21, 2):
        k_means = run_k_means_for_plot(i, df, 1)
        error.append(np.sum(k_means.errors))
        print(i)
        print(np.sum(k_means.errors))

    plt.plot(range(2, 21), error)
    plt.title('K and Sum of error')
    plt.xlabel('K Values')
    plt.ylabel('Sum of error')
    plt.savefig('k-e-rel.jpeg')

def plot_data_graph(df):
    plt.plot(df. iloc[:, 0], df. iloc[:, 1])
    plt.savefig('part-2-data.jpeg')