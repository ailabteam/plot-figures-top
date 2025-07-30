import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Dữ liệu giả định ---
# Tạo dữ liệu cho 3 thuật toán, mỗi thuật toán chạy 50 lần
np.random.seed(42)
data = {
    'Algorithm': np.repeat(['FedAvg', 'FedProx', 'Scaffold (Ours)'], 50),
    'Final Accuracy': np.concatenate([
        np.random.normal(loc=0.88, scale=0.03, size=50),
        np.random.normal(loc=0.90, scale=0.025, size=50),
        np.random.normal(loc=0.92, scale=0.02, size=50)
    ])
}
df = pd.DataFrame(data)
#--------------------------------

# Áp dụng style IEEE
plt.style.use(['science', 'ieee'])

fig, ax = plt.subplots(figsize=(7, 5))

# 1. Vẽ Boxplot trước làm nền
sns.boxplot(x='Algorithm', y='Final Accuracy', data=df, ax=ax,
            boxprops=dict(alpha=0.6)) # Làm cho boxplot hơi trong suốt

# 2. Vẽ Swarmplot đè lên trên để thấy từng điểm dữ liệu
sns.swarmplot(x='Algorithm', y='Final Accuracy', data=df, ax=ax,
              color='k', # Màu đen
              s=3)      # Kích thước điểm nhỏ

ax.set_xlabel("Federated Learning Algorithm")
ax.set_ylabel("Final Model Accuracy")
ax.set_title("Accuracy Distribution across 50 Runs")
ax.grid(axis='y') # Chỉ kẻ lưới ngang cho dễ so sánh

plt.tight_layout()

# Lưu lại file PDF
print("Đang lưu figure 'boxplot_swarm_example.pdf'...")
plt.savefig('boxplot_swarm_example.pdf', format='pdf')
print("Đã lưu xong.")
