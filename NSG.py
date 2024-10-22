from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义读取 .fvecs 文件的函数
def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        while True:
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if not dim.size:  # 检查数组是否为空
                break
            vec = np.fromfile(f, dtype=np.float32, count=dim[0])
            yield np.array(vec)

# 定义读取 .ivecs 文件的函数
def ivecs_read(file_path):
    with open(file_path, 'rb') as f:
        a = np.fromfile(f, dtype='int32')
        d = a[0]  # 第一个数是向量的维度
        return a.reshape(-1, d)[:, 1:] - 1  # 减1以适应0-based索引

class NSG:
    def __init__(self, data, M, ef_construction, ef_search):
        self.data = np.array(data)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.graph = defaultdict(list)
        self.queries = None
        self.ground_truth = None

    def build_graph(self):
        start_time = time.time()  # 开始计时
        nbrs = NearestNeighbors(n_neighbors=self.M, metric='euclidean', algorithm='auto')
        nbrs.fit(self.data)

        def calculate_neighbors(index):
            distances, indices = nbrs.kneighbors([self.data[index]])
            neighbors_list = indices[0].tolist()[1:]  # 排除自身
            self.graph[index].extend(neighbors_list)

        with ThreadPoolExecutor(max_workers=32) as executor:
            list(executor.map(calculate_neighbors, range(len(self.data))))

        print(f"Graph built in {time.time() - start_time:.2f} seconds.")

    def search(self, query, search_k):
        start_time = time.time()  # 开始计时
        visited = set()
        stack = [self._find_start_point(query)]  # 使用栈进行深度优先搜索
        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                for neighbor in self.graph[current_node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
                if len(visited) >= search_k:
                    break
        print(f"Search completed in {time.time() - start_time:.2f} seconds.")
        return list(visited)

    def _find_start_point(self, query):
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(self.data)
        distances, indices = nn.kneighbors([query])
        return indices[0][0]

    def load_ground_truth(self, ground_truth_path):
        self.ground_truth = ivecs_read(ground_truth_path) - 1

    def load_queries(self, queries):
        self.queries = np.array(queries)

    def calculate_recall(self, query_idx, search_k):
        if query_idx >= len(self.queries):
            raise IndexError("Query index out of range")
        query = self.queries[query_idx]
        search_results = self.search(query, search_k)
        gt_neighbors = set(self.ground_truth[query_idx])
        retrieved_neighbors = set(search_results)
        true_positives = len(retrieved_neighbors & gt_neighbors)
        total_neighbors = len(gt_neighbors)
        recall = true_positives / total_neighbors if total_neighbors > 0 else 0
        return recall

# 主执行区块
if __name__ == "__main__":
    project_root = 'E:\yanyi\近邻搜索的高性能构建机制研究\同意框架思路\hnswFirst\siftsmall\siftsmall'  # 替换为包含数据集的实际路径
    data_path = os.path.join(project_root, 'siftsmall_base.fvecs')
    query_path = os.path.join(project_root, 'siftsmall_query.fvecs')
    ground_truth_path = os.path.join(project_root, 'siftsmall_groundtruth.ivecs')

    data_vectors = list(read_fvecs(data_path))
    queries_vectors = list(read_fvecs(query_path))

    nsg = NSG(data_vectors, M=16, ef_construction=450, ef_search=50)
    print("Starting graph construction...")
    nsg.build_graph()  # 构建图并打印所需时间
    nsg.load_ground_truth(ground_truth_path)
    nsg.load_queries(queries_vectors)  # 加载查询向量集

    num_queries = len(nsg.queries)
    search_k = 10000  # 假设我们想要检索的最近邻个数
    recalls = []
    for query_idx in range(num_queries):
        print(f"Starting search for query {query_idx}...")
        recall = nsg.calculate_recall(query_idx, search_k)
        recalls.append(recall)

    print(f"Recall for each query: {recalls}")
    average_recall = np.mean(recalls)  # 计算平均召回率
    print(f"Average Recall: {average_recall * 100:.2f}%")