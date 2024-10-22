from collections import deque
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys

class HNSW:
    def __init__(self, data, M, num_levels, ef_construction, ef_search):
        self.data = data
        self.M = M
        self.num_levels = num_levels
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.graphs = [dict() for _ in range(num_levels)]
        self.entry_points = {level: self._init_level(data, level) for level in range(num_levels)}

    def _init_level(self, data, level):
        # 为每个层次选择一个随机的起点
        return np.random.choice(range(len(data)))

    def build(self):
        # 初始化层次结构
        for level in range(self.num_levels):
            self._build_level(level)

    def _build_level(self, level):
        M_level = self.M // (2 ** (self.num_levels - level))
        for point_idx in self.data:
            self._insert_point(level, point_idx, M_level)

    def _insert_point(self, level, point_idx, M_level):
        ep = self.entry_points[level]
        neighbors = self._search_layer(point_idx, ep, 1, level)
        neighbors = list(neighbors)[:M_level]
        self.graphs[level][point_idx] = neighbors
        for neighbor in neighbors:
            if point_idx not in self.graphs[level][neighbor]:
                self.graphs[level][neighbor].append(point_idx)

    def _search_layer(self, query_idx, ep, ef, level):
        # 这里需要根据伪代码实现SEARCH_LAYER算法
        # 由于篇幅限制，这里只提供框架，具体实现需要根据HNSW算法细节填充
        W = set()
        W.add(ep)
        visited = set()
        visited.add(ep)
        C = deque([ep])
        while C and len(W) < ef:
            c = C.popleft()
            # 这里需要实现邻居的搜索逻辑
            # ...
        return W

    def search(self, query, k):
        W = set()
        ep = self.entry_points[0]
        L = self.num_levels - 1
        for lc in range(L, -1, -1):
            query_idx = np.where(self.data == query)[0][0]  # 假设query存在于data中
            W = self._search_layer(query_idx, ep, 1, lc)
            ep = self._closest_point(query, W)
        return self._retrieve_k_nearest(W, query, k)

    def _closest_point(self, query, W):
        # 找到W中与query最近的点
        distances = [np.linalg.norm(query - self.data[idx]) for idx in W]
        return list(W)[np.argmin(distances)]

    def _retrieve_k_nearest(self, W, query, k):
        # 从W中检索k个最近点
        distances = [np.linalg.norm(query - self.data[idx]) for idx in W]
        sorted_indices = np.argsort(distances)
        return [list(W)[i] for i in sorted_indices[:k]]


# 使用HNSW类
data = np.random.rand(1000, 128)  # 示例数据
query = data[0]  # 示例查询
k = 10  # 想要检索的最近邻个数

M = 16  # 每层的出边数
num_levels = 5  # 层次数
ef_construction = 100  # 图构建时的候选点数
ef_search = 50  # 搜索时考虑的候选点数

hnsw = HNSW(data, M, num_levels, ef_construction, ef_search)
hnsw.build()

neighbors = hnsw.search(query, k)
print("最近邻索引:", neighbors)