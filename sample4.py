import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

print("\nĐang tạo Ví dụ 4: Ablation Study...")

# --- Dữ liệu giả định ---
# (a) Ablation results
models = ['Baseline', 'w/ Attention', 'w/ Residual', 'OurModel (Full)']
accuracy = [85.2, 89.5, 88.1, 92.3]
# (b) Confusion Matrix for the best model (OurModel)
# Rows: True Label, Cols: Predicted Label
conf_matrix = np.array([
    [98,  1,  0,  1],
    [ 2, 95,  3,  0],
    [ 1,  5, 89,  5], # Mô hình vẫn còn nhầm lẫn một chút ở Class 3
    [ 0,  2,  4, 94]
])
class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig = plt.figure(figsize=(14, 6))

# Sử dụng GridSpec để tạo layout 1 hàng, 3 cột (tỉ lệ 1:2)
gs = GridSpec(1, 3, figure=fig)

# Subplot (a) chiếm cột đầu tiên
ax1 = fig.add_subplot(gs[0, 0])
# Subplot (b) chiếm 2 cột còn lại
ax2 = fig.add_subplot(gs[0, 1:])

# --- Subplot (a): Bar Plot for Ablation ---
colors = ['#AAAAAA', '#88CCEE', '#DDCC77', '#332288']
ax1.barh(models, accuracy, color=colors, edgecolor='black') # barh cho dễ đọc tên dài
ax1.set_xlabel('Accuracy (%)')
ax1.set_title('Ablation Study Results')
ax1.set_xlim(84, 94)
ax1.invert_yaxis() # Đảo ngược trục y để Baseline ở trên cùng
ax1.grid(axis='x', linestyle='--', alpha=0.7)
ax1.text(-0.25, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold')

# --- Subplot (b): Confusion Matrix ---
# Chuẩn hóa ma trận theo hàng (theo tổng số lượng thực tế của mỗi lớp)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_matrix_percent, 
            annot=True, fmt='.2%', cmap='Blues', ax=ax2,
            xticklabels=class_names, yticklabels=class_names)
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')
ax2.set_title('Confusion Matrix for OurModel (Full)')
ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('example_4_ablation_study.pdf')
print("Đã lưu 'example_4_ablation_study.pdf'")
