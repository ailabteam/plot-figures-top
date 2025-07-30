import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle

print("\nĐang tạo Mẫu 2: UAV Path Optimization (3 subplots)...")

# --- Dữ liệu giả định ---
# (a) Path Data
start, end = (5, 50), (95, 50)
obstacles = [Rectangle((20, 30), 15, 40, color='gray'), Circle((70, 60), 10, color='gray')]
pso_path_x, pso_path_y = np.array([5, 30, 45, 60, 80, 95]), np.array([50, 75, 50, 25, 40, 50])
our_path_x, our_path_y = np.array([5, 25, 50, 75, 95]), np.array([50, 72, 50, 28, 50])
# (b) Convergence Data
iterations = np.arange(1, 101)
pso_cost = 150 * np.exp(-iterations/40) + 120
our_cost = 200 * np.exp(-iterations/15) + 105
# (c) Final Metrics
metrics = ['Path Length (m)', 'Energy (J)']
pso_values = [122, 550]
our_values = [104, 440]

# --- Vẽ đồ thị ---
plt.style.use(['science', 'ieee', 'no-latex'])
fig = plt.figure(figsize=(15, 7))
gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.4, wspace=0.3)

# --- Subplot (a): Path Visualization (chiếm cả cột trái) ---
ax_map = fig.add_subplot(gs[:, 0])
ax_map.plot(pso_path_x, pso_path_y, 'o--', color='C2', label='PSO Path')
ax_map.plot(our_path_x, our_path_y, 's-', color='C3', label='SmartAnt-PSO Path (Ours)')
for patch in obstacles: ax_map.add_patch(patch)
ax_map.plot(start[0], start[1], 'g^', markersize=12, label='Start')
ax_map.plot(end[0], end[1], 'rX', markersize=12, label='End')
ax_map.set_xlabel('X Coordinate (m)')
ax_map.set_ylabel('Y Coordinate (m)')
ax_map.set_title('UAV Path Planning in Obstacle Field')
ax_map.legend()
ax_map.set_xlim(0, 100)
ax_map.set_ylim(0, 100)
ax_map.set_aspect('equal', adjustable='box')

# --- Subplot (b): Convergence Curve ---
ax_conv = fig.add_subplot(gs[0, 1])
ax_conv.plot(iterations, pso_cost, 'C2--', label='PSO')
ax_conv.plot(iterations, our_cost, 'C3-', label='SmartAnt-PSO')
ax_conv.set_xlabel('Iterations')
ax_conv.set_ylabel('Path Cost')
ax_conv.set_title('Convergence Speed')
ax_conv.legend()

# --- Subplot (c): Final Metrics (Grouped Bar Chart) ---
ax_bar = fig.add_subplot(gs[1, 1])
x = np.arange(len(metrics))
width = 0.35
rect1 = ax_bar.bar(x - width/2, pso_values, width, label='PSO', color='C2')
rect2 = ax_bar.bar(x + width/2, our_values, width, label='SmartAnt-PSO', color='C3')
ax_bar.set_ylabel('Value')
ax_bar.set_title('Final Solution Quality')
ax_bar.set_xticks(x, metrics)
ax_bar.legend()
ax_bar.bar_label(rect1, padding=3)
ax_bar.bar_label(rect2, padding=3)

# Thêm nhãn
ax_map.text(-0.05, 1.02, '(a)', transform=ax_map.transAxes, fontsize=14, fontweight='bold')
ax_conv.text(-0.2, 1.1, '(b)', transform=ax_conv.transAxes, fontsize=14, fontweight='bold')
ax_bar.text(-0.2, 1.1, '(c)', transform=ax_bar.transAxes, fontsize=14, fontweight='bold')

plt.savefig('figure_template_2_uav_optimization.pdf')
print("Đã lưu 'figure_template_2_uav_optimization.pdf'")
