import numpy as np
import os
from scipy.spatial import cKDTree
from collections import defaultdict
import time

def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        while True:
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if not dim: break
            vec = np.fromfile(f, dtype=np.float32, count=dim[0])
            yield vec

def load_fvecs(filename):
    data = np.vstack(list(read_fvecs(filename)))
    print("Data shape:", data.shape)  # 调试信息
    return data

def load_ivecs(filename):
    with open(filename, 'rb') as f:
        count = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.int32, count=count)
        data = data.reshape(-1, 1)
    print("Ground truth shape:", data.shape)  # 调试信息
    return data

class SSG:
    def __init__(self, data, alpha, k=10):
        self.data = np.array(data)
        self.alpha = np.deg2rad(alpha)  # 转换为弧度
        self.kd_tree = cKDTree(self.data)  # 构建KD树以加速邻居搜索
        self.graph = defaultdict(list)
        self.buildssg(k)

    def buildssg(self, k):
        num_points = self.data.shape[0]
        for i in range(num_points):
            dists, indices = self.kd_tree.query(self.data[i], k=k+1)  # 使用 k+1 个邻居，包括自己
            for j in range(1, len(indices)):  # 从 1 开始，跳过自己
                if self._is_valid_edge(i, indices[j], dists, indices):
                    self.graph[i].append(indices[j])

    def _is_valid_edge(self, i, j, dists, indices):
        if i == j:
            return False
        point_i = self.data[i]
        point_j = self.data[j]
        direction = (point_j - point_i) / np.linalg.norm(point_j - point_i)
        for k in indices:
            if k != i:
                point_k = self.data[k]
                direction_k = (point_k - point_i) / np.linalg.norm(point_k - point_i)
                angle = np.arccos(np.clip(np.dot(direction, direction_k), -1.0, 1.0))
                if angle <= self.alpha:
                    return True
        return False

    def search(self, query, ef_search):
        results = []
        for q in query:
            distances, indices = self.kd_tree.query(q, k=ef_search)
            results.append(indices)
        return results

    def calculate_recall(self, query_results, ground_truth):
        recalls = []
        for query_idx, query_result in enumerate(query_results):
            query_neighbors = set(query_result)
            true_neighbors = set(ground_truth[query_idx])
            true_positives = len(query_neighbors & true_neighbors)
            recall = true_positives / len(true_neighbors) if true_neighbors else 0
            recalls.append(recall)
        average_recall = np.mean(recalls)
        return average_recall

# 设置文件路径
project_root = 'E:\yanyi\近邻搜索的高性能构建机制研究\同意框架思路\hnswFirst\siftsmall\siftsmall'
data_path = os.path.join(project_root, 'siftsmall_base.fvecs')
query_path = os.path.join(project_root, 'siftsmall_query.fvecs')
ground_truth_path = os.path.join(project_root, 'siftsmall_groundtruth.ivecs')

# 检查文件是否存在
if not os.path.exists(data_path) or not os.path.exists(query_path) or not os.path.exists(ground_truth_path):
    print("One or more files do not exist.")
else:
    data = load_fvecs(data_path)
    queries = load_fvecs(query_path)
    ground_truth = load_ivecs(ground_truth_path)

    # 初始化 SSG
    alpha = 30  # 角度阈值
    k = 20  # 每个点的邻居数
    ssg = SSG(data, alpha, k)
    ssg.buildssg(k)

    # 执行搜索
    ef_search = 200  # 搜索时考虑的候选点数
    query_results = ssg.search(queries, ef_search)

    # 计算召回率
    recall = ssg.calculate_recall(query_results, ground_truth)
    print("Query Results:", query_results)
    print("Average Recall:", recall)