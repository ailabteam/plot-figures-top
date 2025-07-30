import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

print("Đang tạo Mẫu 4: Federated Learning Analysis (4 subplots)...")

# --- Dữ liệu giả định ---
# (a) Global Accuracy
rounds = np.arange(1, 101)
fedavg_acc = 0.92 - 0.5 * np.exp(-rounds / 25)
fedfair_acc = 0.95 - 0.6 * np.exp(-rounds / 18)
# (b) Communication Cost
costs = {'FedAvg': 1.5, 'FedFair (Ours)': 0.9} # MB per round
# (c) Per-Client Accuracy Distribution (100 clients)
np.random.seed(1)
fedavg_client_acc = np.random.normal(loc=82, scale=8, size=100)
fedfair_client_acc = np.random.normal(loc=91, scale=3, size=100)
client_df = pd.DataFrame({
    'Algorithm': np.repeat(['FedAvg', 'FedFair (Ours)'], 100),
    'Client Accuracy (%)': np.concatenate([fedavg_client_acc, fedfair_client_acc])
})
# (d) Non-IID Data Distribution (10 clients, 5 classes)
clients = [f'Client {i+1}' for i in range(10)]
classes = [f'Class {j+1}' for j in range(5)]
# Dùng Dirichlet để tạo phân bố Non-IID một cách "pro"
dist = np.random.dirichlet(np.ones(5)*0.5, 10) * 1000

# --- Vẽ đồ thị ---
plt.style.use(['science', 'grid', 'muted', 'no-latex'])
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# --- Subplot (a): Global Accuracy vs. Rounds ---
axes[0, 0].plot(rounds, fedavg_acc, '--', label='FedAvg')
axes[0, 0].plot(rounds, fedfair_acc, '-', label='FedFair (Ours)')
axes[0, 0].set_xlabel('Communication Rounds')
axes[0, 0].set_ylabel('Global Model Accuracy')
axes[0, 0].set_title('Global Performance')
axes[0, 0].legend()

# --- Subplot (b): Communication Cost ---
axes[0, 1].bar(costs.keys(), costs.values(), color=['C0', 'C1'])
axes[0, 1].set_ylabel('Cost per Round (MB)')
axes[0, 1].set_title('Communication Overhead')

# --- Subplot (c): Client Fairness ---
sns.violinplot(data=client_df, x='Algorithm', y='Client Accuracy (%)', ax=axes[1, 0])
axes[1, 0].set_title('Per-Client Accuracy Distribution')
axes[1, 0].set_xlabel('')

# --- Subplot (d): Data Heterogeneity ---
im = axes[1, 1].imshow(dist.T, cmap='viridis', aspect='auto')
axes[1, 1].set_xticks(np.arange(len(clients)), labels=clients, rotation=45, ha="right")
axes[1, 1].set_yticks(np.arange(len(classes)), labels=classes)
axes[1, 1].set_title('Non-IID Data Distribution')
fig.colorbar(im, ax=axes[1, 1], label='Number of Samples')

# Thêm nhãn (a), (b), (c), (d)
subplot_labels = ['(a)', '(b)', '(c)', '(d)']
for ax, label in zip(axes.flatten(), subplot_labels):
    ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=14, fontweight='bold')

plt.tight_layout(pad=1.5)
plt.savefig('figure_template_4_federated_learning.pdf')
print("Đã lưu 'figure_template_4_federated_learning.pdf'")
