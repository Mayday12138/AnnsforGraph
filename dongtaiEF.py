from collections import deque
import os
from sklearn.metrics import confusion_matrix
from concurrent.futures import ThreadPoolExecutor
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
start=time.time()
# 三部分：下载数据，预处理数据，执行hnsw，调整参数
#part-1 数据拉取
import sys

def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        while True:
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if not dim: break
            vec = np.fromfile(f, dtype=np.float32, count=dim[0])
            yield vec  # 使用生成器来逐个产生向量，减少内存占用


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # print(d,"<------这是d的数值\n")
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


class HNSW:
    def __init__(self, data, M, num_levels, ef_construction, queries, ground_truth, ef_construction_factor=2.0):
        self.data = data
        self.M = M
        self.num_levels = num_levels
        self.ef_construction_base = ef_construction
        self.ef_construction_factor = ef_construction_factor
        # 使用字典来存储每个点的邻居，优化数据结构
        self.graphs = [{} for _ in range(num_levels)]
        self.queries = queries
        self.ground_truth = ground_truth


    def _connect_points(self, level, point_idx, M, ef_construction_level):
        # 使用多线程进行邻居搜索
        neighbors_indices = self.search_optimized(self.data[point_idx], M + 1, ef_construction_level)
        # 添加邻居到图结构中，排除自身
        self.graphs[level][point_idx] = [idx for idx in neighbors_indices if idx != point_idx]

    def search_optimized(self, query, k, ef_search):
        # 使用 sklearn 的 NearestNeighbors 来找到最近邻
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto', n_jobs=-1)
        nn.fit(self.data)
        distances, indices = nn.kneighbors([query])
        return indices.flatten()

    def _build_level(self, level):
        # 为特定层次构建图
        ef_construction_level = int(self.ef_construction_base / (self.ef_construction_factor ** (self.num_levels - level - 1)))
        M_level = self.M // (2 ** (self.num_levels - level))
        with ThreadPoolExecutor(max_workers=4) as executor:  # 根据系统CPU核心数调整线程数
            # 将点的索引分块，以便并行处理
            futures = [executor.submit(self._connect_points, level, i, M_level, ef_construction_level) for i in range(len(self.data))]
            for future in futures:
                future.result()  # 确保任务完成

    def build(self):
        # 按层级构建图
        for level in range(self.num_levels):
            self._build_level(level)



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

    def _find_neighbors(self, point_idx, M_level, level):
        # 根据层次和M_level找到M_level个最近邻居
        # 这里需要实现一个更高效的最近邻居搜索算法
        # 例如，使用近似最近邻搜索算法
        pass

    def _search_level(self, query, ef_search, level, visited):
        # 找到与查询向量最近的点作为起始点
        start_point_index = self._find_start_point(query)

        # 初始化 BFS 队列
        queue = deque([(start_point_index, 0)])  # (点的索引, 当前层级的深度)
        visited.add(start_point_index)  # 将起始点添加到已访问集合

        while queue:
            current_point_index, depth = queue.popleft()
            if depth > ef_search:
                break  # 超过候选数限制则停止搜索

            # 获取当前点在该层的所有邻居索引
            neighbors_indices = self.graphs[level][current_point_index]
            for neighbor_index in neighbors_indices:
                if neighbor_index not in visited:
                    visited.add(neighbor_index)
                    queue.append((neighbor_index, depth + 1))
                    if len(visited) >= ef_search:
                        break  # 如果已找到足够的候选点，则停止搜索

        return visited

    def _find_start_point(self, query):
        # 确保 query 是一个二维数组
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # 使用 sklearn 的 NearestNeighbors 来找到与查询向量最近的点
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(self.data)
        distances, indices = nn.kneighbors(query)

        # 返回最近邻居的索引
        return indices.flatten()[0]

    def search(self, query, ef_search):
        """
        执行HNSW搜索算法。

        :param query: 查询向量，一维数组。
        :param ef_search: 搜索时考虑的候选点数。
        :return: 找到的邻居索引列表。
        """
        # 确保query是一个二维数组
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # 找到搜索的起始点
        start_point_index = self._find_start_point(query)

        # 初始化 BFS 队列
        visited = set()  # 存储已访问的点的索引
        queue = deque([(start_point_index, 0)])  # (当前点索引, 当前层级)

        while queue and len(visited) < ef_search:
            current_point_index, current_level = queue.popleft()

            # 在当前层执行搜索
            current_set = self._search_level(query, ef_search, current_level, visited)

            # 如果在当前层找到了新的邻居，并且还有更细的层次
            if current_level > 0 and len(current_set) > 0:
                # 选择一个邻居作为下一层的起始点
                next_point_index = list(current_set)[0]
                # 将新的邻居和下一层的层级添加到队列中
                queue.append((next_point_index, current_level - 1))

            # 更新已访问集合
            visited.update(current_set)

            # 如果已找到足够的候选点，则停止搜索
            if len(visited) >= ef_search:
                break

        # 返回找到的邻居索引列表
        return list(visited)

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
project_root = 'E:\yanyi\近邻搜索的高性能构建机制研究\同意框架思路\hnswFirst'
#smallsift_path = os.path.join(project_root, '\siftsmall\siftsmall\siftsmall_base.fvecs')
smallsift_path = 'E:\yanyi\近邻搜索的高性能构建机制研究\同意框架思路\hnswFirst\siftsmall\siftsmall\siftsmall_base.fvecs'
#query_path = os.path.join(project_root, '\siftsmall\siftsmall\siftsmall_query.fvecs')
query_path = 'E:\yanyi\近邻搜索的高性能构建机制研究\同意框架思路\hnswFirst\siftsmall\siftsmall\siftsmall_query.fvecs'
#ground_truth_path = os.path.join(project_root, '\siftsmall\siftsmall\siftsmall_groundtruth.ivecs')
ground_truth_path = 'E:\yanyi\近邻搜索的高性能构建机制研究\同意框架思路\hnswFirst\siftsmall\siftsmall\siftsmall_groundtruth.ivecs'
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

# 初始化 HNSW
M = 100  # 每层的出边数
ef_construction_base = 100  # 基础候选点数
ef_construction_factor = 2.0  # 候选点数递减因子
num_levels = 4  # 层次数
ef_search = 300  # 搜索时考虑的候选点数

hnsw = HNSW(data, M, num_levels, ef_construction_base, queries, ground_truth, ef_construction_factor)
hnsw.data = data
start_time = time.time()
hnsw.build()
# 检查图是否为空
for level in hnsw.graphs:
    for node in level:
        print(node)  # 期望看到每个节点的邻居列表，不应该为空

print(f"HNSW built in {time.time() - start_time} seconds")

# 检查搜索算法
query_vector = data[0]  # 取第一个向量作为查询
search_results = hnsw.search(np.array([query_vector]), ef_search)
search_results2 = hnsw.search(np.array([query_vector]), ef_search)
print("检查搜索算法")
print(search_results)  # 期望看到非空的邻居索引列表
# 计算召回率
ef_search = 200  # 选择一个ef_search值
recall = hnsw.calculate_recall(ef_search)
print(f"Recall with ef_search={ef_search}: {recall}")


# 假设你想要更多的ef_search值来测试，可以扩展这个列表
ef_search_values = [ 50,100, 200, 300, 400, 500,550, 600, 700, 800, 900,950]
recall2 = 0
recalls = []
recalls2 = []
ef_search2 = 50
search_results2 = hnsw.search(np.array([query_vector]), ef_search2)
for ef_search in ef_search_values:
    start_time = time.time()
    true_positives = 0
    total_neighbors = 0
    for query_idx, query in enumerate(queries):
        query_vector = data[query_idx]
        # 假设 ground_truth[query_idx] 是一个数组或列表
        ground_truth_list = ground_truth[query_idx].tolist() if isinstance(ground_truth[query_idx], np.ndarray) else [
            ground_truth[query_idx]]

        # 执行搜索
        search_results = hnsw.search(np.array([query_vector]), ef_search)
        search_results2 = hnsw.search(np.array([query_vector]), ef_search)
        # 确保 search_results 是一个列表
        if isinstance(search_results, np.ndarray):
            neighbors_indices_list = search_results.tolist()
        else:
            neighbors_indices_list = search_results

        # 确保 neighbors_indices_list 是一个列表
        if isinstance(neighbors_indices_list, int):
            neighbors_indices_list = [neighbors_indices_list]

        # 现在我们可以计算 true_positives
        true_positives += len(set(neighbors_indices_list) & set(ground_truth_list))
        total_neighbors += len(ground_truth_list)
    recall = true_positives / total_neighbors if total_neighbors > 0 else 0
    recall2 = hnsw.calculate_recall(ef_search)
    recalls2.append(recall2)
    recalls.append(recall)
    print(f"Recall with ef_search={ef_search}: {recall}")
    print(f"Search for ef_search={ef_search} took {time.time() - start_time} seconds")

# 现在x轴和y轴的数据点数量匹配
print("recall2:")
print(recalls2)
x_values = np.arange(len(ef_search_values))  # 使用数组的索引作为x轴的值
plt.plot(x_values, recalls, marker='o')  # 设置marker选项
plt.xlabel('ef_search Parameter')
plt.ylabel('Recall')
plt.title('Recall vs. ef_search Parameter')
plt.xticks(x_values)  # 确保x轴的刻度与ef_search_values对应
plt.show()
quit(0)
