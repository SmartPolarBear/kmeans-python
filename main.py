import argparse
from typing import Final

# 进行数值计算
import numpy as np
# 用于读取csv文件和方便地进行方差、均值计算
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# 使用方法: main.py --dataset <数据集路径>


class KMeans:
    def __init__(self, feats: pd.DataFrame, k: int):
        self.tries = 0
        self.feats = feats
        self.k = k

    def _wcss(self, centroids, cluster) -> float:
        """
        计算准则函数J
        :param centroids:类的中心
        :param cluster:分类内容
        :return: J函数值
        """
        ret = 0.0
        for i, val in enumerate(self.feats.values):
            ret += np.sqrt(
                (centroids[int(cluster[i]), 0] - val[0]) ** 2 + (centroids[int(cluster[i]), 1] - val[1]) ** 2)
        return ret

    def cluster(self, max_tries: int = 32767):
        """
        进行k-means分类
        :param max_tries: 最大尝试次数
        :return: 聚类中心在源数据集中的索引、聚类中心、聚类结果、J函数值
        """
        self.tries = 0

        # 将聚类结果初始化为0
        cluster = np.zeros(self.feats.shape[0])

        # 随机抽样k个点作为初始的聚类中心
        centroid_indexes, centroids = self.feats.sample(n=self.k).index, self.feats.sample(n=self.k).values

        while self.tries < max_tries:
            self.tries += 1

            # 对每个点计算
            for id, row in enumerate(self.feats.values):
                min_dist = float('inf')
                for cid, centroid in enumerate(centroids):
                    # 计算距离，与距离最小的中心分为一类
                    dist = np.sqrt((centroid[0] - row[0]) ** 2 + (centroid[1] - row[1]) ** 2)
                    if dist < min_dist:
                        min_dist, cluster[id] = dist, cid

            # 对每一类计算平均数，作为新的聚类中心
            clustered_centroids = self.feats.copy().groupby(by=cluster).mean().values

            # 如果新的中心与旧中心重合，聚类就结束了
            if np.count_nonzero(centroids - clustered_centroids) == 0:
                break
            else:
                centroids = clustered_centroids
        return centroid_indexes, centroids, cluster, self._wcss(centroids, cluster)

    def get_tries(self) -> int:
        return self.tries


def clustering(path: str):
    """
    调用聚类函数、可视化结果
    :param path: 数据集的路径
    :return:
    """
    raw = pd.read_csv(path)

    data = raw.copy()
    data.insert(1, "SepalSquare", data["SepalLengthCm"] * data["SepalWidthCm"])
    data.insert(1, "PetalSquare", data["PetalLengthCm"] * data["PetalWidthCm"])
    # print(data.describe(), data)

    sb.pairplot(data, vars=data.columns[1:7], hue="Species")
    plt.show()

    plt.clf()

    # 取出选择的特征
    feats = data.iloc[:, [1, 2]]
    # print(feats)

    # 调用聚类函数
    km: KMeans = KMeans(feats=feats, k=3)
    cid, cen, clu, cost = km.cluster()
    result = raw.copy()
    result.insert(1, "Class", clu)
    # print(result)

    class_map = dict()
    species_names = list()
    specie_count = dict()
    class_count = dict()
    error_count = dict()
    species = pd.DataFrame(result.iloc[:, [1, 6]].copy().groupby(by=clu).agg(pd.Series.mode))
    for i, row in species.iterrows():
        species_names.append(row["Species"])
        class_map[row["Class"]] = row["Species"]
        class_count[row["Species"]] = \
            result.copy().loc[result["Class"] == row["Class"]].shape[0]
        specie_count[row["Species"]] = \
            result.copy().loc[result["Species"] == row["Species"]].shape[0]
        error_count[row["Species"]] = \
            result.copy().loc[(result["Class"] == row["Class"]) & (result["Species"] != row["Species"])].shape[0]

    pretty_clu = list([class_map[c] for c in clu])

    samples = pd.DataFrame(data.copy().iloc[cid.values])

    sb.scatterplot(data.iloc[:, 1], data.iloc[:, 2], hue=pretty_clu)
    sb.scatterplot(cen[:, 0], cen[:, 1], s=100, color='b', marker='X', label="Center")
    sb.scatterplot(samples["PetalSquare"], samples["SepalSquare"], s=50, color='r', marker='s',
                   label="Initial Elements Chosen")

    plt.xlabel("PetalSquare")
    plt.ylabel("SepalSquare")

    plt.title("Clustering the Iris Dataset")

    plt.show()

    plt.clf()

    error = pd.DataFrame()
    error["Species"] = species_names
    error["Actual"] = [specie_count[s] for s in species_names]
    error["Clustered"] = [class_count[s] for s in species_names]
    error["Error"] = [error_count[s] for s in species_names]
    error["ErrorRate"] = [round(float(error_count[s]) / float(class_count[s]), 2) for s in species_names]

    print(error)

    bar_data = pd.DataFrame(columns=["Species", "Count", "Type"])

    for s in species_names:
        bar_data = pd.concat([bar_data,
                              pd.DataFrame({"Species": [s], "Count": [specie_count[s]], "Value": ["Actual"]}),
                              pd.DataFrame({"Species": [s], "Count": [class_count[s]], "Value": ["Clustered"]}),
                              pd.DataFrame({"Species": [s], "Count": [error_count[s]], "Value": ["Error"]})]
                             , ignore_index=True, axis=0)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sb.lineplot(data=error["ErrorRate"], marker="s", sort=False, ax=ax2, label="Error Rate", color='r')
    sb.barplot(x="Species", y="Count", data=bar_data, hue="Value", ax=ax1)

    plt.title("Results and Errors")
    plt.show()


if __name__ == "__main__":
    # 使用方法: main.py --dataset <数据集路径>
    parser = argparse.ArgumentParser(description="KMeans Clustering")
    parser.add_argument("--dataset", help="The path to the dataset file in CSV format")

    args = parser.parse_args()

    dataset: Final = str(args.dataset)

    print("Dataset: {}".format(dataset))

    clustering(dataset)
