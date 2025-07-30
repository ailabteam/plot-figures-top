import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# --- Dữ liệu giả định ---
np.random.seed(0)
# Kích thước mô hình (triệu tham số) và độ chính xác
model_size = np.random.normal(50, 15, 200).clip(5, 100)
accuracy = 0.85 + (model_size / 200) - ((model_size - 50) / 100)**2 + np.random.normal(0, 0.02, 200)
accuracy = accuracy.clip(0.6, 1.0)
#--------------------------------

# Sử dụng style cơ bản, sạch sẽ
plt.style.use(['science', 'grid'])

fig = plt.figure(figsize=(7, 7))
gs = GridSpec(4, 4)

# Định nghĩa các subplot
ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_x = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
ax_hist_y = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)

# --- Vẽ các đồ thị ---

# 1. Scatter plot ở trung tâm
ax_scatter.scatter(model_size, accuracy, alpha=0.6, s=20, edgecolors='k', linewidths=0.5)

# 2. Histogram phân phối của X ở trên
ax_hist_x.hist(model_size, bins=30, alpha=0.7)

# 3. Histogram phân phối của Y ở bên phải
ax_hist_y.hist(accuracy, bins=30, orientation='horizontal', alpha=0.7)

# --- Dọn dẹp & Thêm nhãn ---

# Ẩn các tick label không cần thiết để tránh rối
ax_hist_x.tick_params(axis="x", labelbottom=False)
ax_hist_y.tick_params(axis="y", labelleft=False)

# Nhãn cho các trục chính
ax_scatter.set_xlabel("Model Size (Million Parameters)")
ax_scatter.set_ylabel("Accuracy")

# Nhãn cho các histogram
ax_hist_x.set_ylabel("Count")
ax_hist_y.set_xlabel("Count")


fig.suptitle("Model Size vs. Accuracy with Marginal Distributions", y=0.95)

# Lưu lại file PDF
print("Đang lưu figure 'scatter_marginal_example.pdf'...")
plt.savefig('scatter_marginal_example.pdf', format='pdf')
print("Đã lưu xong.")
