import torch
from torch_geometric.data import Data

class BAT:
    def __init__(self, num_classes, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.virtual_nodes = None
        self.virtual_features = None
        self.virtual_labels = None

    def calculate_uncertainty(self, pred_probs):
        pred_labels = pred_probs.argmax(dim=1)
        one_hot = torch.zeros_like(pred_probs)
        one_hot.scatter_(1, pred_labels.unsqueeze(1), 1)
        uncertainty = 0.5 * torch.sum(torch.abs(pred_probs - one_hot), dim=1)
        return uncertainty

    def calculate_risk(self, uncertainty, predicted_labels, true_labels=None):

        if true_labels is not None:
            class_counts = torch.bincount(true_labels, minlength=self.num_classes)
            label_rates = class_counts.float() / len(true_labels)
        else:
            class_counts = torch.bincount(predicted_labels, minlength=self.num_classes)
            label_rates = class_counts.float() / len(predicted_labels)

        max_label_rate = label_rates.max()
        node_label_rates = label_rates[predicted_labels]
        calibration_factors = max_label_rate / node_label_rates

        risks = uncertainty * calibration_factors
        return risks

    def zeroth_order_posterior(self, pred_probs, predicted_labels):
        posterior = torch.zeros_like(pred_probs)
        for j in range(self.num_classes):
            if j != predicted_labels:
                mask = (predicted_labels != j)
                posterior[:, j] = pred_probs[:, j] / (1 - pred_probs[:, predicted_labels] + 1e-10)
        return posterior

    def first_order_posterior(self, pred_probs, predicted_labels, edge_index):
        num_nodes = pred_probs.shape[0]
        posterior = torch.zeros((num_nodes, self.num_classes), device=self.device)

        row, col = edge_index
        adj = torch.sparse_coo_tensor(edge_index,
                                      torch.ones_like(row, dtype=torch.float),
                                      size=(num_nodes, num_nodes))

        for i in range(num_nodes):
            neighbors = adj[i].coalesce().indices()[0]
            if len(neighbors) == 0:
                continue

            neighbor_preds = predicted_labels[neighbors]
            total_neighbors = len(neighbors)
            same_class = (neighbor_preds == predicted_labels[i]).sum().item()

            for j in range(self.num_classes):
                if j != predicted_labels[i]:
                    class_count = (neighbor_preds == j).sum().item()
                    denominator = total_neighbors - same_class
                    if denominator > 0:
                        posterior[i, j] = class_count / denominator

        return posterior

    def create_virtual_nodes(self, features, labels):
        self.virtual_nodes = torch.arange(self.num_classes, device=self.device) + features.shape[0]
        self.virtual_features = torch.zeros((self.num_classes, features.shape[1]), device=self.device)
        self.virtual_labels = torch.arange(self.num_classes, device=self.device)

        for j in range(self.num_classes):
            class_mask = (labels == j)
            if class_mask.any():
                self.virtual_features[j] = features[class_mask].mean(dim=0)

    def augment_topology(self, data, risks, posterior, risk_threshold=0.5):
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]

        if self.virtual_nodes is None:
            self.create_virtual_nodes(data.x, data.y)

        class_avg_risk = torch.zeros(self.num_classes, device=self.device)
        class_counts = torch.zeros(self.num_classes, device=self.device)

        for j in range(self.num_classes):
            class_mask = (data.y == j)
            if class_mask.any():
                class_avg_risk[j] = risks[class_mask].mean()
                class_counts[j] = class_mask.sum()

        new_edges = []
        new_edge_weights = []

        for i in range(num_nodes):
            predicted_class = data.y[i] if hasattr(data, 'y') else posterior[i].argmax()
            gamma = max(risks[i] - class_avg_risk[predicted_class], 0)

            if gamma > 0 and risks[i] > risk_threshold:
                for j in range(self.num_classes):
                    if j != predicted_class:
                        q = gamma * posterior[i, j]
                        if torch.rand(1).item() < q:
                            new_edges.append([i, self.virtual_nodes[j].item()])
                            new_edge_weights.append(q.item())

        if not new_edges:
            return data

        new_edges = torch.tensor(new_edges, dtype=torch.long, device=self.device).t()
        all_edges = torch.cat([data.edge_index, new_edges], dim=1)

        if hasattr(data, 'edge_weight'):
            new_edge_weights = torch.tensor(new_edge_weights, dtype=torch.float, device=self.device)
            all_weights = torch.cat([data.edge_weight, new_edge_weights])
        else:
            all_weights = torch.ones(all_edges.shape[1], device=self.device)

        all_features = torch.cat([data.x, self.virtual_features])
        all_labels = torch.cat([data.y, self.virtual_labels]) if hasattr(data, 'y') else None

        augmented_data = Data(
            x=all_features,
            edge_index=all_edges,
            edge_weight=all_weights,
            y=all_labels
        )

        return augmented_data