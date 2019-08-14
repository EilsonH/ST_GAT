import numpy as np

#定义了图的label策略
class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes #最大邻域范围，一般是1
        dilation (int): controls the spacing between the kernel points #步长，一般都是1

    """

    def __init__(self,
                 layout='openpose',
                 strategy='distance', #, spatial, uniform
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout) #得到边连接信息
        self.hop_dis = get_hop_distance( #根据最大邻域范围max_hop得到每个节点和它的邻域节点的距离信息，也就是距离矩阵
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy) #生成邻接矩阵

    def __str__(self):
        return self.A

    def get_edge(self, layout): #添加了一个edge属性
        if layout == 'openpose': #layout布局
            self.num_node = 18  #默认18，如果要改成14，还需要改其他地方，不然会报错
            self_link = [(i, i) for i in range(self.num_node)] #自环连接
            # 通过openpose估计得到的数据的图结构，这个版本的结构和工程中不一样，具体要看openpose是在哪个数据集上训练的
            # 这里使用的18个关节的版本是在coco数据集上训练的
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]# 尝试把与眼睛有关的直接去掉
            self.edge = self_link + neighbor_link
            self.center = 1

        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    #
    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation) # valid_hop = range(0,2)，是一个range对象，其行为和list很像，但不是list，是一个可迭代对象
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1 #只要是在hop_dis中值不为inf的，在邻接矩阵中都是1
        # normalize_adjacency = normalize_digraph(adjacency) #对邻接矩阵进行归一化处理，通过度矩阵来实现
        normalize_adjacency = normalize_undigraph(adjacency)  #按照无向图的归一化方式，原始代码中用的是有向图归一化方式
        if strategy == 'uniform': #全部共享
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency #没有变化
            self.A = A
        elif strategy == 'distance': #自身享受一个权重，1邻域享受一个权重
            # 生成一个三维数组，也就是多个二维矩阵，数量等同于最大领域+1
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):#enumerate返回可迭代对象的(索引，值)组合，一般用在for循环中
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop] #重新将邻接矩阵拆分为两个矩阵,分别是不同邻域的
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

#该函数返回一个距离矩阵，其中每个元素代表了两个节点之间的距离，而max_hop限制了这个距离的最大值，如果为大于max_hop，则为inf，即无穷大
#默认最大距离为1
def get_hop_distance(num_node, edge, max_hop=1):
    # 直接定义邻接矩阵
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf #np.inf表示无穷大，这里首先初始化了一个N*N的矩阵
    # 对矩阵求N阶幂，这里是求0次和1次幂，0次幂是单位方阵，1次幂就是矩阵本身
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)] #
    #自环处理
    arrive_mat = (np.stack(transfer_mat) > 0) #这里arrive_mat是一个只包含true和false的矩阵
    for d in range(max_hop, -1, -1): #d为1,0
        hop_dis[arrive_mat[d]] = d #
    return hop_dis

#邻接矩阵的归一化
def normalize_digraph(A): #归一化的有向图
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0: # 有可能存在=0的情况
            Dn[i, i] = Dl[i]**(-1) # **表示幂运算，这里是指求-1次幂，也就是倒数
    AD = np.dot(A, Dn) # Dn的对角线元素都是度的倒数，Dn就是公式中的D-1，即度矩阵的逆矩阵
    return AD


def normalize_undigraph(A): #归一化的无向图,然而这个公式没有用到
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

if __name__=="__main__":
    print('now start debuging')
    graph=Graph() #把断点设置在构造函数中的各种方法就可以开始调试了
    print(graph.A.shape)