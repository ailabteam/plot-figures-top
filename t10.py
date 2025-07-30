import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

print("Đang tạo Mẫu 10: t-SNE Evolution Analysis (1x4 subplots)...")

# --- Dữ liệu giả định ---
# Chúng ta sẽ mô phỏng việc các cụm ngày càng tách biệt hơn
n_samples = 500
n_classes = 5
epochs_to_show = [1, 20, 50, 100]
all_tsne_results = []
# Độ phân tán của cụm giảm dần theo thời gian
cluster_stds = [5.0, 2.5, 1.0, 0.5] 

for std in cluster_stds:
    # Tạo dữ liệu đặc trưng và nhãn
    features, labels = make_blobs(n_samples=n_samples, centers=n_classes, 
                                  cluster_std=std, random_state=42)
    # Áp dụng t-SNE (ĐÃ SỬA LỖI)
    # Xóa tham số 'n_iter' không còn được hỗ trợ
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    all_tsne_results.append((tsne_results, labels))

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
# Tạo figure rất rộng, chia sẻ cả 2 trục
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)

# Lặp qua các kết quả và vẽ
for i, ax in enumerate(axes):
    tsne_results, labels = all_tsne_results[i]
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, 
                         cmap='viridis', alpha=0.7, s=20)
    ax.set_title(f'Epoch {epochs_to_show[i]}')
    ax.set_xticks([]) # Ẩn các vạch chia
    ax.set_yticks([]) # Ẩn các vạch chia

# Thêm một chú thích chung cho toàn bộ figure
handles, legend_labels = scatter.legend_elements()
fig.legend(handles, [f'Class {i}' for i in range(n_classes)], 
           loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=n_classes)

fig.suptitle("Evolution of Feature Space During Training", fontsize=16)
plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Dành không gian cho legend và suptitle
plt.savefig('figure_template_10_tsne_evolution.pdf')
print("Đã lưu 'figure_template_10_tsne_evolution.pdf'")
