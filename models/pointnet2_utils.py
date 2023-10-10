import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint): #寻找最远距离点
    """
    Input:
        xyz: pointcloud data, [B, N, 3]  # 输入的
        npoint: number of samples  # 采样的数量
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  #初始化采样点矩阵B*npoint零矩阵，npoint为采样点数
    distance = torch.ones(B, N).to(device) * 1e10 #初始化距离，B*npoints矩阵每个值都是1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) #随机初始化最远点，随机数范围是从0-N，一共是B个，维度是1*B，保证每个B都有一个最远点
    batch_indices = torch.arange(B, dtype=torch.long).to(device) #0~(B-1)的数组
    for i in range(npoint): #寻找并选取空间中每个点距离多个采样点的最短距离，并存储在dist
        centroids[:, i] = farthest #设采样点为farthers点,[:,i]为取所有行的第i个数据
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3) #取中心点也是farthest点
        dist = torch.sum((xyz - centroid) ** 2, -1) #求所有点到farthest点的欧式距离和
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1] #返回最大距离的点
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):  #寻找球半径里面的点，从S个球内采样nsample个点
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape  #原始点云的BNC
    _, S, _ = new_xyz.shape #由index_points得出的S，例如一共有S个球
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) #获取点云的各个点的序列位置
    sqrdists = square_distance(new_xyz, xyz) #计算中心点与所有点的欧式距离
    group_idx[sqrdists > radius ** 2] = N #大于欧氏距离平方的点序列标签设置为N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] #升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample]) # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    mask = group_idx == N #把N点也替换成第一个点的值（是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换）
    group_idx[mask] = group_first[mask] # 找到group_idx中值等于N的点
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: 要找多少个centorid中心点(采样点）
        radius:  查询或者搜索半径，所有的点都是归一化，所以是一个小于1的数字
        nsample: 要区域查询半径内（每个球形区域）要找多少个点
        xyz: input points position data, [B, N, 3] 输入点的位置，可以理解搜索的中心，B个batch，然后N个点，3维
        points: input points data, [B, N, D] 输入点的数据，D特征维度？
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3] 通过最远点采样（FPS）得到的采样点的坐标
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C] # 使用最远点采样（FPS）得到的采样点的坐标id索引
    new_xyz = index_points(xyz, fps_idx) # 使用 `index_points` 函数从原始点云中提取采样点，形成一个新的点集，称为 `new_xyz`
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # 将原始点云分割为npoint个球形区域，每个区域有nsample个采样点的索引
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]  # 从原始点云中提取采样点的坐标（球星区域内）
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  #每个球体区域的点减去中心点，类似归一

    if points is not None:  # 当采样点不为空的话
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]  ，原始特征和球星拼接
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):  #将所有点作为一个group，和上面相同
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint   # 获得的采样点数
        self.radius_list = radius_list   # 半径尺度设置
        self.nsample_list = nsample_list  # 不同尺度采样的数量
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):   # 同一个尺度，使用多个mlp进行
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  # 通过FPS 从1024采样到512个点的索引值，xyz是原始位置，作为采样点
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  # 对每个尺度进行球查询 ballquery 根据某个中心点，聚簇k个邻居点
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)  #获得相对中心的距离
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S] 转置（由[B, D, S,K ]
            for j in range(len(self.conv_blocks[i])):  # 对每个簇内进行学习
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]   通过上面的多层感知机的学习后，经过max pooling
            new_points_list.append(new_points) # 使用列表进行拼接

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):   # in_channel=1280, mlp=[256, 256]
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2): #前面两层的质心和前面两层的输出
        """                                         #利用前一层的点对后面的点进行插值
        Input:
            xyz1: input points position data, [B, C, N]    # l2层输出 xyz
            xyz2: sampled input points position data, [B, C, S] # l3层输出  xyz
            points1: input points data, [B, D, N]   #l2层输出  points
            points2: input points data, [B, D, S]   #l3层输出  points
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)   # 第一次插值 2,3,128 ---> 2,128,3 | 第二次插值时 2,3,512--->2,512,3
        xyz2 = xyz2.permute(0, 2, 1)   # 第一次插值时2,3,1  ---> 2 ,1,3    |  第二次插值时 2,3,128--->2,128,3

        points2 = points2.permute(0, 2, 1) #  第一次插值时2,1021,1  --->2,1,1024  最后低维信息，压缩成一个点了  这个点有1024个特征
                                           # 第二次插值 2，256，128 --->2,128,256
        B, N, C = xyz1.shape    # N = 128   低维特征的点云数  （其数量大于高维特征）
        _, S, _ = xyz2.shape   # s = 1   高维特征的点云数

        if S == 1: # 如果最后只有一个点，就将S直复制N份后与与低维信息进行拼接
            interpolated_points = points2.repeat(1, N, 1)
        else: # 如果不是一个点 则插值放大倍数 128个点---->512个点；计算出的两层之间任意两点距离是一个矩阵 512x128 也就是512个低维点与128个高维点 两两之间的距离
            dists = square_distance(xyz1, xyz2)  # 第二次插值前 先计算高维与低维的距离 2,512,128
            dists, idx = dists.sort(dim=-1) # 2,512,128 在最后一个维度进行排序 默认进行升序排序，也就是越靠前的位置说明 xyz1离xyz2距离较近
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3] 找到距离最近的三个邻居，这里的idx：2,512,3的含义就是512个点与128个距离最近的前三个点的索引， 例如第一行就是：对应128个点中那三个与512中第一个点距离最近

            dist_recip = 1.0 / (dists + 1e-8) # 求距离的倒数 2,512,3 对应论文中的 Wi(x)
            norm = torch.sum(dist_recip, dim=2, keepdim=True) # 将距离最近的三个邻居的距离加起来  此时对应论文中公式的分母部分
            weight = dist_recip / norm  # 2,512,3   每个距离占总和的比重 也就是weight计算权重；index_points(points2, idx)-->2,512,3,256
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            #  ref：https://blog.csdn.net/weixin_47142735/article/details/121213984
            #  points2: 2,128,256 (128个点 256个特征)   idx 2,512,3 （512个点中与128个点距离最近的三个点的索引）
            #  index_points(points2, idx) 从高维特征（128个点）中找到对应低维特征（512个点） 对应距离最小的三个点的特征 2,512,3,256
            #  这个索引的含义比较重要，可以再看一下idx参数的解释，其实2,512,3,256中的512都是高维特征128个点组成的。
            #  例如 512中的第一个点 可能是由128中的第 1 2 3 组成的；第二个点可能是由2 3 4 三个点组成的


        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

