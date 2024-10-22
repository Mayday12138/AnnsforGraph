from collections import deque
import os
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

start = time.time()

# 三部分：下载数据，预处理数据，执行nsw，调整参数

# part-1 数据拉取
import sys

def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        while True:
            # 读取向量的维度（整数）
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if not dim: break
            # 根据给定的维度读取向量
            vec = np.fromfile(f, dtype=np.float32, count=dim[0])
            yield vec

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

class NSW:
    def __init__(self, data, M, ef_construction, queries, ground_truth):
        self.data = data
        self.M = M
        self.ef_construction = ef_construction
        self.graph = []  # 存储图结构
        self.queries = queries
        self.ground_truth = ground_truth

    def build(self):
        # 构建图结构
        for i, point in enumerate(self.data):
            self._connect_points(i)

    def _connect_points(self, point_idx):
        # 使用最近邻算法找到 M 个最近的邻居
        nn = NearestNeighbors(n_neighbors=self.M + 1, metric='euclidean').fit(self.data)
        distances, indices = nn.kneighbors([self.data[point_idx]])
        distances = distances.flatten()
        indices = indices.flatten()

        # 选择最近的 M 个邻居，排除自身
        nearest_neighbors = indices[1:self.M + 1]

        # 将这些邻居添加到图中
        self.graph.append(nearest_neighbors.tolist())

    def load_smallsift(self, filepath):
        # smallsift 数据集通常是二进制格式，这里假设每个特征向量是 128 维的
        self.data = np.fromfile(filepath, dtype=np.float32).reshape(-1, 128)

    def load_query_and_ground_truth(self, query_filepath, ground_truth_filepath):
        self.queries = np.fromfile(query_filepath, dtype=np.float32).reshape(-1, 128)
        self.ground_truth = {}
        with open(ground_truth_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                query_index = int(parts[0])
                neighbors = [int(part) for part in parts[1:]]
                self.ground_truth[query_index] = neighbors

    def search(self, query, ef_search):
        """
        执行NSW搜索算法。

        :param query: 查询向量，一维数组。
        :param ef_search: 搜索时考虑的候选点数。
        :return: 找到的邻居索引列表。
        """
        # 确保query是一个二维数组
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # 使用 sklearn 的 NearestNeighbors 来找到与查询向量最近的点
        nn = NearestNeighbors(n_neighbors=ef_search, metric='euclidean').fit(self.data)
        distances, indices = nn.kneighbors(query)

        # 返回找到的邻居索引列表
        return indices.flatten().tolist()

    def calculate_recall(self, ef_search):
        true_positives = 0
        total_neighbors = 0
        for query_idx, query_vector in enumerate(self.queries):  # 使用查询向量
            neighbors_indices = self.search(query_vector, ef_search)  # 传入查询向量
            true_positives += len(set(neighbors_indices) & set(self.ground_truth[query_idx]))
            total_neighbors += len(self.ground_truth[query_idx])
        recall = true_positives / total_neighbors if total_neighbors > 0 else 0
        return recall

# 指定 smallsift 数据集、查询集和基准最近邻的路径
project_root = 'E:\yanyi\近邻搜索的高性能构建机制研究\同意框架思路\hnswFirst\siftsmall\siftsmall'
smallsift_path = os.path.join(project_root, 'siftsmall_base.fvecs')
query_path = os.path.join(project_root, 'siftsmall_query.fvecs')
ground_truth_path = os.path.join(project_root, 'siftsmall_groundtruth.ivecs')

# 读取数据
start_time = time.time()
data = fvecs_read(smallsift_path)
# 检查数据加载
print(data.shape)  # 确保数据加载了正确的形状
print(f"Data loaded in {time.time() - start_time} seconds")

# 读取查询和基准最近邻
start_time = time.time()
queries = fvecs_read(query_path)
ground_truth = ivecs_read(ground_truth_path)
# 检查查询和基准最近邻是否加载
print(queries.shape)  # 确保查询向量加载了正确的形状
print(len(ground_truth))  # 确保基准最近邻加载了正确的长度
print(f"Queries and ground truth loaded in {time.time() - start_time} seconds")

# 初始化 NSW
M = 50  # 每层的出边数 有效测试（1）M=100，ef_search = 500，ef_construction = 550
ef_construction = 450  # 图构建时的候选点数
ef_search = 10  # 搜索时考虑的候选点数

nsw = NSW(data, M, ef_construction, queries, ground_truth)
nsw.data = data
start_time = time.time()
nsw.build()
# 检查图是否为空
for node in nsw.graph:
    print(node)  # 期望看到每个节点的邻居列表，不应该为空

print(f"NSW built in {time.time() - start_time} seconds")

# 检查搜索算法
query_vector = data[0]  # 取第一个向量作为查询
search_results = nsw.search(np.array([query_vector]), ef_search)
print("检查搜索算法")
print(search_results)  # 期望看到非空的邻居索引列表
# 计算召回率
ef_search = 200  # 选择一个ef_search值
recall = nsw.calculate_recall(ef_search)
print(f"Recall with ef_search={ef_search}: {recall}")

# 假设你想要更多的ef_search值来测试，可以扩展这个列表
ef_search_values = [50, 100, 200, 300, 400, 500, 550, 600, 700, 800, 900, 950]
recalls = []
for ef_search in ef_search_values:
    start_time = time.time()
    recall = nsw.calculate_recall(ef_search)
    recalls.append(recall)
    print(f"Recall with ef_search={ef_search}: {recall}")
    print(f"Search for ef_search={ef_search} took {time.time() - start_time} seconds")

# 现在x轴和y轴的数据点数量匹配
x_values = np.arange(len(ef_search_values))  # 使用数组的索引作为x轴的值
plt.plot(x_values, recalls, marker='o')  # 设置marker选项
plt.xlabel('ef_search Parameter')
plt.ylabel('Recall')
plt.title('Recall vs. ef_search Parameter')
plt.xticks(x_values)  # 确保x轴的刻度与ef_search_values对应
plt.show()