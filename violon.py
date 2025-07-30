import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print("\nĐang tạo Figure 4: Violin & Box Plot for Optimization Results...")

# --- Dữ liệu giả định ---
# So sánh chất lượng giải pháp cuối cùng của 3 thuật toán qua 100 lần chạy
np.random.seed(0)
data = {
    'Algorithm': np.repeat(['Genetic Algorithm', 'Particle Swarm', 'Our Method'], 100),
    'Solution Quality': np.concatenate([
        np.random.normal(loc=150, scale=20, size=100), # GA có phương sai lớn
        np.random.normal(loc=130, scale=10, size=100), # PSO ổn định hơn
        np.random.normal(loc=120, scale=8, size=100)   # Phương pháp của ta tốt nhất
    ])
}
df = pd.DataFrame(data)

# --- Vẽ đồ thị ---
plt.style.use(['science', 'grid', 'no-latex'])
fig, ax = plt.subplots(figsize=(9, 6))

# Vẽ Violin Plot
sns.violinplot(x='Algorithm', y='Solution Quality', data=df, ax=ax,
               inner=None, # Tắt các đường mặc định bên trong violin
               palette='muted') # Dùng bộ màu của style muted

# Vẽ Box Plot chồng lên trên
# Chỉnh độ rộng và độ trong suốt để nó nằm gọn bên trong violin
sns.boxplot(x='Algorithm', y='Solution Quality', data=df, ax=ax,
            width=0.2, boxprops={'facecolor':'white', 'zorder': 2},
            whiskerprops={'zorder': 2}, capprops={'zorder': 2})


# --- Tinh chỉnh và chi tiết ---
ax.set_ylabel("Objective Function Value (Lower is Better)")
ax.set_title("Distribution of Final Solutions from Optimization Algorithms")

plt.tight_layout()
plt.savefig('figure_4_violin_optimization.pdf')
print("Đã lưu 'figure_4_violin_optimization.pdf'")
