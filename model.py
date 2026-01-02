# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

from bat_module import BAT

class DualChannelGCNWithBAT(nn.Module):
    def __init__(self, input_dim, gcn_hidden=64, att_hidden=32, mlp_hidden=16, num_classes=2):
        super(DualChannelGCNWithBAT, self).__init__()

        self.num_classes = num_classes

        self.channel1_gcn1 = GCNConv(input_dim, gcn_hidden)
        self.channel1_gcn2 = GCNConv(gcn_hidden, gcn_hidden)

        self.channel2_gcn1 = GCNConv(input_dim, gcn_hidden)
        self.channel2_gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.channel2_bat = BAT(num_classes=num_classes)

        self.attention = nn.Sequential(
            nn.Linear(gcn_hidden * 2, att_hidden),
            nn.Tanh(),
            nn.Linear(att_hidden, 2)
        )

        print("gcn_hidden:", gcn_hidden)
        print("mlp_hidden:", mlp_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(gcn_hidden * 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, num_classes)
        )

        self.temperature = 0.1
        self.lambda_cl = 0.1

        self.reset_parameters()

    def reset_parameters(self):
        for conv in [self.channel1_gcn1, self.channel1_gcn2,
                     self.channel2_gcn1, self.channel2_gcn2]:
            conv.reset_parameters()
        for layer in self.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.attention:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data1, data2, apply_bat=False):

        x1, edge_index1 = data1.x, data1.edge_index
        x1 = F.relu(self.channel1_gcn1(x1, edge_index1))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = F.relu(self.channel1_gcn2(x1, edge_index1))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2, edge_index2 = data2.x, data2.edge_index
        x2 = F.relu(self.channel2_gcn1(x2, edge_index2))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = F.relu(self.channel2_gcn2(x2, edge_index2))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        if apply_bat and self.training:
            with torch.no_grad():
                combined = torch.cat([x1, x2], dim=1)
                temp_out = self.classifier(combined)
                log_probs = F.log_softmax(temp_out, dim=1)
                probs = torch.exp(log_probs)
                pred_labels = probs.argmax(dim=1)

                uncertainty2 = self.channel2_bat.calculate_uncertainty(probs)
                risks2 = self.channel2_bat.calculate_risk(uncertainty2, pred_labels)
                posterior2 = self.channel2_bat.first_order_posterior(probs, pred_labels, edge_index2)
                augmented_data2 = self.channel2_bat.augment_topology(data2, risks2, posterior2)

                x2_aug = F.relu(self.channel2_gcn1(augmented_data2.x, augmented_data2.edge_index))
                x2_aug = F.dropout(x2_aug, p=0.5, training=self.training)
                x2_aug = F.relu(self.channel2_gcn2(x2_aug, augmented_data2.edge_index))
                x2 = x2_aug[:x2.size(0)]

        combined = torch.cat([x1, x2], dim=1)
        attention_weights = torch.softmax(self.attention(combined), dim=1)
        x1_weighted = x1 * attention_weights[:, 0:1]
        x2_weighted = x2 * attention_weights[:, 1:2]
        fused = torch.cat([x1_weighted, x2_weighted], dim=1)

        out = self.classifier(fused)

        loss_cl = self.contrastive_loss(x1, x2, data1.y)

        return F.log_softmax(out, dim=1), loss_cl

    def contrastive_loss(self, x1, x2, labels):
        pos_sim = F.cosine_similarity(x1, x2, dim=1) / self.temperature
        neg_sim = torch.mm(x1, x2.t()) / self.temperature
        neg_sim = neg_sim - torch.diag_embed(torch.diag(neg_sim))
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        pos_weights = 1 / (class_counts[labels] + 1e-10)
        neg_weights = torch.zeros_like(neg_sim)
        for c in range(self.num_classes):
            class_mask = (labels == c)
            neg_weights[class_mask, :] = 1 / (class_counts[c] + 1e-10)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        weights = torch.cat([pos_weights.unsqueeze(1), neg_weights], dim=1)
        weighted_logits = logits * weights
        labels = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
        loss = F.cross_entropy(weighted_logits, labels)
        return loss * self.lambda_cl

def calculate_metrics(y_true, y_pred, y_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUROC': roc_auc_score(y_true, y_proba),
        'AUPRC': average_precision_score(y_true, y_proba)
    }
    return metrics

def get_class_specific_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        pos_acc = tp / (tp + fn)
        neg_acc = tn / (tn + fp)
        return pos_acc, neg_acc
    return 0, 0