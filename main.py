import numpy as np
import matplotlib.pyplot as plt

class dbscan:
    # Khởi tạo các biến epsilon và minPts
    def __init__(self, eps, minPts):
        self.eps = eps
        self.minPts = minPts   
        
    # Phân cụm
    def fit(self, X):
        # X là một tập các điểm (x, y)
        self.X = np.array(X)
        # Tạo biến labels có kích thước bằng kích thước X và có giá trị các phần tử ban đầu bằng -1 
        self.labels = np.full(len(X), -1)
        self.cluster_id = 0
        
        for point_id in range(len(X)):
            # Nếu điểm đang xét đã được đi qua thì bỏ qua
            if self.labels[point_id] != -1:
                continue
            
            # Tạo tập neighbors chứa các điểm lân cận với điểm đang xét
            neighbors = self.region_query(point_id)
            # Nếu số lượng điểm lân cận bé hơn minPts thì điểm đó là điểm nhiễu -> labes[] = 0
            if len(neighbors) < self.minPts:
                self.labels[point_id] = 0
            # Nếu số lượng điểm lân cận lớn hơn hoặc bằng minPts thì gọi hàm expland_cluster để mở rộng cụm và tằng cluster_id
            else:
                self.expandCluster(point_id, neighbors)
                self.cluster_id += 1
                
    def region_query(self, point_id):
        # Tính khoảng cách từ điểm đang xét đến các điểm khác
        distances = np.linalg.norm(self.X - self.X[point_id], axis = 1)
        # Trả về mảng các điểm có khoảng cách bé hơn epsilon
        return np.where(distances <= self.eps)[0]
    
    def expandCluster(self, point_id, neighbors):
        # Gán chỉ số cụm cho điểm hiện tại 
        self.labels[point_id] = self.cluster_id
        
        # Duyệt các điểm lân cận điểm hiện tại
        i = 0
        while i < len(neighbors):
            neighbors_point = neighbors[i]
            # Nếu điểm lân cận là điểm nhiễu thì gán chỉ số cụm cho điểm lân cận
            if self.labels[neighbors_point] == 0:
                self.labels[neighbors_point] = self.cluster_id
            
            # Nếu điểm lân cận chưa được gán thì gán chỉ số cụm cho điểm lân cận 
            elif self.labels[neighbors_point] == -1:
                self.labels[neighbors_point] = self.cluster_id
                
                # Tạo một tập new_neighbors của điểm lân cận
                new_neighbors = self.region_query(neighbors_point)
                
                # Nếu số lượng phần tử tập new_neighbors lớn hơn hoặc bằng minPts thì 
                if len(new_neighbors) >= self.minPts:
                    # Nối tập new_neighbors vào tập neighbors 
                    neighbors = np.concatenate((neighbors, new_neighbors))
            i += 1
        
X = [[1, 1], [1, 0], [0, 1], [0, 0], [0.5, 0.5], [0.5, 0.6], 
     [0.6, 0.5], [0.6, 0.6], [10, 10], [11, 11], [10.5, 10.5], 
     [15, 12], [15, 15], [15, 13], [20, 20], [0, 15]]

dbscan = dbscan(0.5, 5)
dbscan.fit(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=dbscan.labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()