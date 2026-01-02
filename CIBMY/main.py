import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from graph_construction import load_and_preprocess_data, construct_population_graph1, construct_population_graph2, \
    create_graph_data
from model import DualChannelGCNWithBAT, calculate_metrics, get_class_specific_metrics


def cross_validate(X, y, n_splits, epochs, use_bat):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        A1 = construct_population_graph1(X, y)
        A2 = construct_population_graph2(X)
        data1 = create_graph_data(X, A1, y)
        data2 = create_graph_data(X, A2, y)
        train_mask = torch.zeros(data1.x.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(data1.x.shape[0], dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True


        model = DualChannelGCNWithBAT(input_dim=X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.NLLLoss()


        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out, loss_cl = model(data1, data2, apply_bat=use_bat)
            loss_bce = criterion(out[train_mask], data1.y[train_mask])
            loss = loss_cl + loss_bce
            loss.backward()
            optimizer.step()


        model.eval()
        with torch.no_grad():
            logits, _ = model(data1, data2, apply_bat=False)
            y_pred = logits.argmax(dim=1).numpy()
            y_proba = torch.exp(logits)[:, 1].numpy()

            metrics = calculate_metrics(data1.y[test_mask].numpy(),
                                        y_pred[test_mask],
                                        y_proba[test_mask])

            pos_acc, neg_acc = get_class_specific_metrics(data1.y[test_mask].numpy(), y_pred[test_mask])
            metrics['Positive Accuracy'] = pos_acc
            metrics['Negative Accuracy'] = neg_acc

            all_metrics.append(metrics)

            print(f"Fold {fold + 1} Metrics:")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")


    final_results = {}
    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        final_results[metric] = f"{mean:.3f}±{std:.3f}"
    return final_results

def main():
    try:
        file_path = "data.xlsx"

        X, y, _, _ = load_and_preprocess_data(file_path)
        results_with_bat = cross_validate(X, y, n_splits=5, epochs=48, use_bat=True)
        print("\n=== 最终评估结果 ===")
        for metric, value in results_with_bat.items():
            print(f"{metric}: {value}")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()