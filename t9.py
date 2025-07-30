import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

print("Đang tạo Mẫu 9: AI Model Reliability and Robustness (3 subplots)...")

# --- Dữ liệu giả định ---
# (a) Robustness to Noise
acc_data = {
    'Condition': ['Clean Data', 'Noisy Data', 'Clean Data', 'Noisy Data'],
    'Accuracy (%)': [92.5, 75.3, 94.1, 88.9],
    'Model': ['Baseline', 'Baseline', 'Calib-Aug (Ours)', 'Calib-Aug (Ours)']
}
acc_df = pd.DataFrame(acc_data)
# (b) Calibration Curve
baseline_conf = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
baseline_acc = np.array([0.1, 0.2, 0.4, 0.5, 0.6]) # Under confident at low conf, over confident at high
calibaug_conf = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
calibaug_acc = np.array([0.1, 0.3, 0.52, 0.71, 0.88]) # Much closer to ideal
# (c) t-SNE Visualization
features, labels = make_blobs(n_samples=300, centers=4, cluster_std=2.5, random_state=42)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(features)

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig = plt.figure(figsize=(16, 7))
gs = GridSpec(2, 2, width_ratios=[1.2, 1], hspace=0.35)

# --- Subplot (a): Robustness Bar Plot (chiếm cả cột trái) ---
ax1 = fig.add_subplot(gs[:, 0])
sns.barplot(data=acc_df, x='Condition', y='Accuracy (%)', hue='Model', ax=ax1)
ax1.set_title('Robustness to Noisy Data')
ax1.set_ylim(70, 100)
ax1.grid(axis='y', linestyle='--')

# --- Subplot (b): Calibration Plot ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
ax2.plot(baseline_conf, baseline_acc, 'o--', label='Baseline')
ax2.plot(calibaug_conf, calibaug_acc, 's-', label='Calib-Aug (Ours)')
ax2.set_xlabel('Mean Predicted Confidence')
ax2.set_ylabel('Fraction of Positives')
ax2.set_title('Model Calibration')
ax2.legend()

# --- Subplot (c): t-SNE Feature Visualization ---
ax3 = fig.add_subplot(gs[1, 1])
scatter = ax3.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
ax3.set_title('t-SNE of Learned Feature Space')
ax3.set_xticks([])
ax3.set_yticks([])
legend1 = ax3.legend(*scatter.legend_elements(), title="Classes")
ax3.add_artist(legend1)

# Thêm nhãn
ax1.text(-0.1, 1.02, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold')
ax2.text(-0.25, 1.1, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')
ax3.text(-0.25, 1.1, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figure_template_9_ai_reliability.pdf')
print("Đã lưu 'figure_template_9_ai_reliability.pdf'")
