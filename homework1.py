from itertools import combinations

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np

def elbow_method(data) -> int:
    sum_of_squared = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        sum_of_squared.append((k, kmeans.inertia_))

    d_function = []

    for k in range(2, 9):
        numerator = abs(sum_of_squared[k][1] - sum_of_squared[k + 1][1])
        denominator = abs(sum_of_squared[k - 1][1] - sum_of_squared[k][1])
        d_function.append((sum_of_squared[k][0], numerator / denominator))

    min_value = 999999999
    output_index = -1

    for index, num in d_function:
        if min_value > num:
            min_value = num
            output_index = index

    return output_index


def distance(dot1, dot2):
    return np.sqrt(np.sum((dot1 - dot2) ** 2))


def find_centroids(data, num_of_clusters):
    centroids_indexes = np.random.choice(data.shape[0], num_of_clusters, replace=False)
    return data[centroids_indexes]


def visualisation(data, centroids, clusters, step):
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', s=50, label='Точки')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Центроиды')
    plt.title(f'Шаг {step} (2D проекция)')
    plt.xlabel('Признак 1 (Длина чашелистика)')
    plt.ylabel('Признак 2 (Ширина чашелистика)')
    plt.legend()
    plt.show()


def update_centroids(data, clusters, num_of_centroids, old_centroids):
    centroids = []
    for i in range(num_of_centroids):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = old_centroids[i]
        centroids.append(new_centroid)

    return np.array(centroids)


def k_means(data, num_of_clusters, max_step=100):
    centroids = find_centroids(data, num_of_clusters)
    for i in range(max_step):
        clusters = []
        for point in data:
            distances = [distance(point, centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            clusters.append(cluster)

        visualisation(data, centroids, clusters, i)

        new_centroid = update_centroids(data, np.array(clusters), 3, centroids)
        if np.all(centroids == new_centroid):
            print(f"алгоритм закончил работу на {i} шаге")
            break
        centroids = new_centroid

    return centroids, clusters


def plot_all_projections(data, clusters, centroids, feature_names):
    features = range(data.shape[1])

    for x_index, y_index in combinations(features, 2):
        plt.scatter(data[:, x_index], data[:, y_index], c=clusters, cmap='viridis', s=50, label='Точки')
        plt.scatter(centroids[:, x_index], centroids[:, y_index], c='red', marker='X', s=200, label='Центроиды')
        plt.title(f' проекция {feature_names[x_index]} vs {feature_names[y_index]}')
        plt.xlabel(feature_names[x_index])
        plt.ylabel(feature_names[y_index])
        plt.legend()
        plt.show()

def main():
    irises = load_iris()
    data = irises.data
    centroids, clusters = k_means(data, elbow_method(data))
    feature_names = irises['feature_names']
    plot_all_projections(data, clusters, centroids, feature_names)


if __name__ == "__main__":
    main()