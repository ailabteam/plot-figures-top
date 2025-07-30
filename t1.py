import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

print("Đang tạo Mẫu 1: AI Classification Analysis (4 subplots)...")

# --- Dữ liệu giả định ---
# (a) Training History
epochs = np.arange(1, 31)
acc = 0.95 - 0.3 * np.exp(-epochs/5)
loss = 0.5 * np.exp(-epochs/4) + 0.05
# (b) Confusion Matrix
conf_matrix = np.array([[450, 15, 5], [10, 480, 10], [8, 2, 490]])
# (c) ROC Curve data
baseline_fpr, baseline_tpr = np.array([0, 0.1, 0.4, 1]), np.array([0, 0.7, 0.9, 1])
our_fpr, our_tpr = np.array([0, 0.05, 0.2, 1]), np.array([0, 0.85, 0.98, 1])
# (d) Feature/Block Importance
blocks = ['Conv Block 1', 'Conv Block 2', 'Attention Block', 'FC Layer']
importance = [0.25, 0.35, 0.85, 0.5] # Attention là quan trọng nhất

# --- Vẽ đồ thị ---
plt.style.use(['science', 'grid', 'no-latex'])
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Subplot (a): Training History ---
ax1_twin = axes[0, 0].twinx() # Tạo trục y thứ hai
p1, = axes[0, 0].plot(epochs, acc, 'C0-', label='Accuracy')
p2, = ax1_twin.plot(epochs, loss, 'C1--', label='Loss')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Accuracy', color='C0')
ax1_twin.set_ylabel('Loss', color='C1')
axes[0, 0].set_title('Training Process')
axes[0, 0].legend(handles=[p1, p2])

# --- Subplot (b): Confusion Matrix ---
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Confusion Matrix (ClassifyNet)')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_ylabel('True Label')

# --- Subplot (c): ROC Curve ---
axes[1, 0].plot(baseline_fpr, baseline_tpr, 'C2--', label='Baseline (AUC = 0.75)')
axes[1, 0].plot(our_fpr, our_tpr, 'C3-', label='ClassifyNet (AUC = 0.92)')
axes[1, 0].plot([0, 1], [0, 1], 'k:', label='Random Chance')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve Comparison')
axes[1, 0].legend()

# --- Subplot (d): Block Importance ---
axes[1, 1].barh(blocks, importance, color='C4')
axes[1, 1].set_xlabel('Importance Score')
axes[1, 1].set_title('Model Block Importance')

# Thêm nhãn (a), (b), (c), (d)
subplot_labels = ['(a)', '(b)', '(c)', '(d)']
for ax, label in zip(axes.flatten(), subplot_labels):
    ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figure_template_1_classification.pdf')
print("Đã lưu 'figure_template_1_classification.pdf'")
