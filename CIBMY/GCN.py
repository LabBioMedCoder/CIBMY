# 单通道
# 步骤4：定义三层GCN+MLP模型
class ThreeLayerGCN_MLP(nn.Module):
    def __init__(self, input_dim, gcn_hidden=64, mlp_hidden=32, num_classes=2):
        super(ThreeLayerGCN_MLP, self).__init__()

        # 三层GCN特征提取
        self.gcn1 = GCNConv(input_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.gcn3 = GCNConv(gcn_hidden, gcn_hidden)

        # MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(gcn_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, num_classes)  # 二分类输出
        )

        # BAT模块
        self.bat = BAT(num_classes=num_classes)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        for conv in [self.gcn1, self.gcn2, self.gcn3]:
            conv.reset_parameters()
        for layer in self.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data, apply_bat=False):
        x, edge_index = data.x, data.edge_index

        # 第一层GCN
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层GCN
        x = F.relu(self.gcn2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # 第三层GCN
        x = F.relu(self.gcn3(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # 分类预测
        out = self.classifier(x)
        # 分类预测
        out = self.classifier(x)
        log_probs = F.log_softmax(out, dim=1)

        # 应用BAT拓扑增强
        if apply_bat and self.training:
            with torch.no_grad():
                # 计算预测概率
                probs = torch.exp(log_probs)
                pred_labels = probs.argmax(dim=1)

                # 计算不确定性和风险
                uncertainty = self.bat.calculate_uncertainty(probs)
                risks = self.bat.calculate_risk(uncertainty, pred_labels)

                # 计算后验似然(使用一阶方法)
                posterior = self.bat.first_order_posterior(probs, pred_labels, edge_index)

                # 执行拓扑增强
                augmented_data = self.bat.augment_topology(data, risks, posterior)

                # 使用增强后的数据重新计算特征
                x_aug = F.relu(self.gcn1(augmented_data.x, augmented_data.edge_index))
                x_aug = F.dropout(x_aug, p=0.5, training=self.training)
                x_aug = F.relu(self.gcn2(x_aug, augmented_data.edge_index))
                x_aug = F.dropout(x_aug, p=0.5, training=self.training)
                x_aug = F.relu(self.gcn3(x_aug, augmented_data.edge_index))

                # 只保留原始节点的特征
                x = x_aug[:x.size(0)]

        return log_probs