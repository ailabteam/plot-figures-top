import matplotlib.pyplot as plt
import numpy as np

print("Đang tạo Mẫu 12: Satellite Resource Allocation Comparison (1x4 subplots)...")

# --- Dữ liệu giả định ---
# Tạo một "bản đồ" nhu cầu (demand map) có 2 trung tâm thành phố
grid_size = 50
demand_map = np.zeros((grid_size, grid_size))
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
# Thành phố 1 (lớn)
demand_map += 1.0 * np.exp(-((x - 15)**2 + (y - 15)**2) / 50)
# Thành phố 2 (nhỏ)
demand_map += 0.6 * np.exp(-((x - 38)**2 + (y - 35)**2) / 30)

# Tạo kết quả phân bổ của 4 thuật toán
# (a) Static: Phân bổ đều
static_alloc = np.ones((grid_size, grid_size))
# (b) Greedy: Chỉ tập trung vào điểm nóng nhất
greedy_alloc = np.zeros_like(demand_map)
greedy_alloc[10:20, 10:20] = 2.5
# (c) Round-Robin: Phân bổ theo các dải không hiệu quả
rr_alloc = np.sin(x/3) * np.cos(y/3) + 1.5
# (d) Our Method: Bám sát bản đồ nhu cầu
our_alloc = demand_map * 1.8 + 0.2

allocations = [static_alloc, greedy_alloc, rr_alloc, our_alloc]
titles = ['(a) Static RA', '(b) Greedy RA', '(c) Round-Robin RA', '(d) Demand-Aware RA (Ours)']

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

# Lặp qua các thuật toán và vẽ heatmap
for i, ax in enumerate(axes):
    # ĐÃ SỬA LỖI: Thay thế 'rocket' bằng một colormap có sẵn như 'inferno'
    im = ax.imshow(allocations[i], cmap='inferno', vmin=0, vmax=2.5)
    ax.set_title(titles[i])
    ax.set_xticks([])
    ax.set_yticks([])

# Thêm một colorbar chung cho cả figure
fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', 
             label='Allocated Bandwidth (Gbps)', shrink=0.8, pad=0.02)

fig.suptitle("Comparison of Satellite Resource Allocation Strategies", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figure_template_12_satellite_allocation.pdf')
print("Đã lưu 'figure_template_12_satellite_allocation.pdf'")
