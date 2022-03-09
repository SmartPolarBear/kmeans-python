import argparse
from typing import Final

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


class KMeans:
    def __init__(self, feats: pd.DataFrame, k: int):
        self.tries = 0
        self.feats = feats
        self.k = k

    def _wcss(self, centroids, cluster) -> float:
        ret = 0.0
        for i, val in enumerate(self.feats.values):
            ret += np.sqrt(
                (centroids[int(cluster[i]), 0] - val[0]) ** 2 + (centroids[int(cluster[i]), 1] - val[1]) ** 2)
        return ret

    def cluster(self, max_tries: int = 32767):
        self.tries = 0

        cluster = np.zeros(self.feats.shape[0])
        centroids = self.feats.sample(n=self.k).values

        while self.tries < max_tries:
            self.tries += 1

            for id, row in enumerate(self.feats.values):
                min_dist = float('inf')
                for cid, centroid in enumerate(centroids):
                    dist = np.sqrt((centroid[0] - row[0]) ** 2 + (centroid[1] - row[1]) ** 2)
                    if dist < min_dist:
                        min_dist, cluster[id] = dist, cid
            clustered_centroids = self.feats.copy().groupby(by=cluster).mean().values
            if np.count_nonzero(centroids - clustered_centroids) == 0:
                break
            else:
                centroids = clustered_centroids
        return centroids, cluster, self._wcss(centroids, cluster)

    def get_tries(self) -> int:
        return self.tries


def clustering(path: str):
    raw = pd.read_csv(path)

    data = raw.copy()
    data.insert(1, "SepalSquare", data["SepalLengthCm"] * data["SepalWidthCm"])
    data.insert(1, "PetalSquare", data["PetalLengthCm"] * data["PetalWidthCm"])
    # print(data.describe(), data)

    # sb.pairplot(data, vars=data.columns[1:7], hue="Species")
    # plt.show()

    feats = data.iloc[:, [1, 2]]
    # print(feats)

    km: KMeans = KMeans(feats=feats, k=3)
    cen, clu, cost = km.cluster()
    result = raw.copy()
    result.insert(1, "Class", clu)
    # print(result)

    sb.scatterplot(data.iloc[:, 1], data.iloc[:, 2], hue=clu)
    sb.scatterplot(cen[:, 0], cen[:, 1], s=100, color='b', marker='X')

    plt.xlabel("PetalSquare")
    plt.ylabel("SepalSquare")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMeans Clustering")
    parser.add_argument("--dataset", help="The path to the dataset file in CSV format")

    args = parser.parse_args()

    dataset: Final = str(args.dataset)

    print("Dataset: {}".format(dataset))

    clustering(dataset)
