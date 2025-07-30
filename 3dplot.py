import matplotlib.pyplot as plt
import numpy as np

print("\nĐang tạo Figure 7: 3D Surface for Pareto Front...")

# --- Dữ liệu giả định ---
# Giả sử 3 mục tiêu: Throughput (càng cao càng tốt), Latency (càng thấp càng tốt),
# và Power Consumption (càng thấp càng tốt).
# Chúng ta sẽ vẽ 1/Latency để tất cả các trục đều theo hướng "càng cao càng tốt".
np.random.seed(1)
# Tạo một lưới cho 2 mục tiêu đầu
latency_inv = np.linspace(1/100, 1/20, 50) # 1/Latency (ms)
throughput = np.linspace(50, 200, 50) # Mbps
X, Y = np.meshgrid(latency_inv, throughput)

# Mục tiêu thứ 3 (Power) là một hàm của 2 mục tiêu kia
# Càng muốn latency thấp và throughput cao thì càng tốn năng lượng
# Chúng ta vẽ 1/Power cho đồng nhất
Z = 1 / ( (1/X)*0.5 + Y*0.2 + np.random.rand(50, 50)*10 )

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
# Để vẽ 3D, chúng ta cần tạo một subplot với projection='3d'
fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": "3d"})

# Vẽ bề mặt
# rstride và cstride kiểm soát độ "thưa" của lưới, giúp đồ thị nhẹ hơn
surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                       linewidth=0, antialiased=False, alpha=0.8)

# --- Tinh chỉnh và chi tiết ---
ax.set_xlabel('1 / Latency (ms$^{-1}$)')
ax.set_ylabel('Throughput (Mbps)')
ax.set_zlabel('1 / Power (W$^{-1}$)')
ax.set_title('Pareto Front for Satellite Network Design')

# Thêm thanh màu
fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

# Thay đổi góc nhìn cho đẹp hơn
ax.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.savefig('figure_7_3dplot_pareto.pdf')
print("Đã lưu 'figure_7_3dplot_pareto.pdf'")
