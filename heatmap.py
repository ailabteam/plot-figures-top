import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

print("\nĐang tạo Figure 3: Heatmap for Correlation Matrix...")

# --- Dữ liệu giả định ---
# Giả sử chúng ta khảo sát 5 tham số trong một thuật toán tối ưu
np.random.seed(42)
data = {
    'Learning Rate': np.random.rand(100) * 0.01,
    'Batch Size': np.random.randint(16, 128, 100),
    'Momentum': 0.9 + np.random.rand(100) * 0.09,
    'Convergence Time': 100 - np.random.rand(100) * 30,
    'Final Error': 0.05 + np.random.rand(100) * 0.1
}
df = pd.DataFrame(data)
# Tạo mối tương quan nhân tạo
df['Convergence Time'] -= df['Learning Rate'] * 2000
df['Final Error'] -= df['Momentum'] * 0.1

# Tính toán ma trận tương quan
corr_matrix = df.corr()

# --- Vẽ đồ thị ---
# Heatmap thường không cần style của SciencePlots vì Seaborn đã rất đẹp,
# nhưng chúng ta vẫn dùng để đồng bộ font chữ.
plt.style.use(['science', 'no-latex']) 
fig, ax = plt.subplots(figsize=(8, 7))

# Sử dụng Seaborn để vẽ heatmap
sns.heatmap(corr_matrix, 
            annot=True,      # Hiển thị giá trị số trong mỗi ô
            fmt=".2f",       # Định dạng số với 2 chữ số thập phân
            cmap='viridis',  # Chọn một bảng màu đẹp (colormap)
            linewidths=.5,   # Thêm đường kẻ giữa các ô
            ax=ax)

# --- Tinh chỉnh và chi tiết ---
ax.set_title("Correlation Matrix of Optimization Hyperparameters")
# Xoay nhãn để dễ đọc hơn
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('figure_3_heatmap_optimization.pdf')
print("Đã lưu 'figure_3_heatmap_optimization.pdf'")
