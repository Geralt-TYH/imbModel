import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from layers import GraphConvolution
from collections import Counter
from math import ceil
from layers import InnerProductDecoder


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x

class GCIM(nn.Module):
    def __init__(self, nfeat, nhid, nclass,dropout):
        super(GCIM, self).__init__()
        #保存聚类中心的映射表，是一个元组的列表，元组中是聚类中心的特征向量和对饮的类标签
        self.class_centers_map = None
        #聚类中心列表，保存所有的聚类中心
        self.clusting_center_list = None
        #输入向量的维度
        self.nfeat = nfeat
        #隐层向量的维度
        self.nhid = nhid
        #输出向量的维度，因为要使用softmax，与class的数目一致
        self.nclass = nclass
        
        #解码器
        self.decoder = InnerProductDecoder()
        #随机失活率
        self.dropout = dropout
        #编码器
        self.encoder = GCN(nfeat=self.nfeat,nhid=self.nhid,nclass=self.nclass,dropout=self.dropout)#全连接层->待删除
        self.fc = nn.Linear(nclass, nclass)

    def forward(self, input, adj,labels):
        #得到中间特征向量
        z = self.encoder.forward(input,adj)
        # #生成映射表
        self.class_centers_map = self._generate_classes_map(z,labels)
        self.clusting_center_list = self._generate_clusting_center_list()
        output = self.fc(z)
        return F.log_softmax(output, dim=1)

    def kl_clusting_loss(self,input,adj,labels):
        with torch.no_grad():
            z = self.encoder(input,adj)
            #TODO 改成用前向传播中生成的聚类中心表
            q = self.soft_assignment(z, labels)
            p = self.target_distribution(q)
            KLDivLoss = nn.KLDivLoss(size_average=False)
            kl_loss_p_q = KLDivLoss(q.log(), p) / q.shape[0]
        return kl_loss_p_q

    def _generate_classes_map(self,z,labels):
        labels_counter = Counter(labels.numpy())
        # 得到所有的类别
        classes = labels_counter.keys()
        #最小实例数量的类别
        minority_class =min(labels_counter,key=labels_counter.get)

        #计算不平衡率,键为标签，值为不平衡率
        imbalace_rate = {}
        for item in labels_counter.items():
            imbalace_rate.update({item[0]:ceil(item[1]/labels_counter.get(minority_class))})

        #生成每个类的聚类中心,并且保存映射表中
        class_centers_map = []
        for class_ in classes:
            cluster = KMeans(n_clusters=imbalace_rate[class_])
            cluster.fit(z[torch.where(labels == class_)].detach().numpy())
            for center in cluster.cluster_centers_:
                class_centers_map.append((center,class_))
        return class_centers_map
    def _generate_clusting_center_list(self):
        clusting_center_list = []
        for i in self.class_centers_map:
            clusting,_ = i
            clusting_center_list.append(clusting)
        return clusting_center_list
    def soft_assignment(self,z,labels):
        norm_squared = torch.sum((z.unsqueeze(1) - torch.Tensor(self.clusting_center_list)) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared))

        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q

    def target_distribution(self,batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def recon_loss(self,adj,z:torch.Tensor):
        return torch.sum(self.decoder(z)-adj,dim=1).mean()/adj.size()[0]
