import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

print("Đang tạo Mẫu 8: Satellite Multi-Objective Analysis (3 subplots)...")

# --- Dữ liệu giả định ---
users = np.array([10, 20, 30, 40, 50])
# (a) Per-User Throughput
static_thr = 50 - 0.5 * users
dynamic_thr = 48 - 0.2 * users
# (b) Jain's Fairness Index
static_fair = 0.95 - 0.01 * users
dynamic_fair = np.full_like(users, 0.98)
# (c) 3D Pareto Front data
static_points = np.random.rand(20, 3) * np.array([20, 0.8, 0.1]) + np.array([20, 0.1, 0.85])
dynamic_points = np.random.rand(20, 3) * np.array([20, 0.8, 0.05]) + np.array([30, 0.1, 0.95])

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig = plt.figure(figsize=(15, 7))
gs = GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1], hspace=0.4, wspace=0.3)

# --- Subplot (a): Throughput vs. Users ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(users, static_thr, 'o--', label='Static-RA')
ax1.plot(users, dynamic_thr, 's-', label='Dynamic-RA (Ours)')
ax1.set_xlabel('Number of Users')
ax1.set_ylabel('Avg. Throughput (Mbps)')
ax1.set_title('Scalability')
ax1.legend()
ax1.grid(True)

# --- Subplot (b): Fairness vs. Users ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(users, static_fair, 'o--', label='Static-RA')
ax2.plot(users, dynamic_fair, 's-', label='Dynamic-RA (Ours)')
ax2.set_xlabel('Number of Users')
ax2.set_ylabel("Jain's Fairness Index")
ax2.set_title('User Fairness')
ax2.legend()
ax2.set_ylim(0.7, 1.0)
ax2.grid(True)

# --- Subplot (c): 3D Pareto Front (chiếm cả cột phải) ---
ax3 = fig.add_subplot(gs[:, 1], projection='3d')
ax3.scatter(static_points[:, 0], static_points[:, 1], static_points[:, 2], 
            c='C0', marker='o', label='Static-RA Solutions')
ax3.scatter(dynamic_points[:, 0], dynamic_points[:, 1], dynamic_points[:, 2], 
            c='C1', marker='s', label='Dynamic-RA Solutions')
ax3.set_xlabel('Throughput')
ax3.set_ylabel('1 / Latency')
ax3.set_zlabel('Fairness')
ax3.set_title('Multi-Objective Trade-off Space')
ax3.legend()
ax3.view_init(elev=20, azim=45)

# Thêm nhãn
ax1.text(-0.25, 1.1, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold')
ax2.text(-0.25, 1.1, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')
ax3.text2D(-0.1, 1.05, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold')

plt.savefig('figure_template_8_satellite_tradeoff.pdf')
print("Đã lưu 'figure_template_8_satellite_tradeoff.pdf'")
