import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

print("Đang tạo Mẫu 5: UAV-DRL Control System (3 subplots)...")

# --- Dữ liệu giả định ---
# (a) UAV Trajectory
sensors = np.random.rand(10, 2) * 100
path = np.cumsum(np.random.randn(200, 2), axis=0) + 50
path = np.clip(path, 0, 100)
# (b) Reward Curve
episodes = np.arange(1, 501)
reward = 200 * (1 - np.exp(-episodes/150)) + np.random.normal(0,10,500)
# (c) Action Distribution
actions = ['Forward', 'Left', 'Right', 'Hover']
# Ban đầu: random; Sau đó: ưu tiên Forward
dist_start = np.array([0.25, 0.25, 0.25, 0.25])
dist_mid = np.array([0.4, 0.2, 0.2, 0.2])
dist_end = np.array([0.7, 0.15, 0.1, 0.05])
action_history = np.array([dist_start, dist_mid, dist_end]).T

# --- Vẽ đồ thị ---
plt.style.use(['science', 'ieee', 'no-latex'])
fig = plt.figure(figsize=(15, 7))
gs = GridSpec(2, 2, width_ratios=[2, 1.2], hspace=0.3, wspace=0.35)

# --- Subplot (a): Mission Environment & Trajectory (chiếm cả cột trái) ---
ax_map = fig.add_subplot(gs[:, 0])
ax_map.plot(path[:, 0], path[:, 1], '-', color='C1', label='UAV Trajectory')
ax_map.scatter(sensors[:, 0], sensors[:, 1], c='C2', marker='x', s=100, label='Ground Sensors')
ax_map.plot(path[0, 0], path[0, 1], 'g>', markersize=12, label='Start')
ax_map.set_xlabel('X Coordinate (m)')
ax_map.set_ylabel('Y Coordinate (m)')
ax_map.set_title('UAV Data Collection Mission')
ax_map.legend()
ax_map.set_aspect('equal', adjustable='box')

# --- Subplot (b): DRL Reward Curve ---
ax_reward = fig.add_subplot(gs[0, 1])
ax_reward.plot(episodes, reward, color='C3')
ax_reward.set_xlabel('Training Episodes')
ax_reward.set_ylabel('Average Reward')
ax_reward.set_title('Learning Curve')

# --- Subplot (c): Action Distribution (Stacked Bar) ---
ax_action = fig.add_subplot(gs[1, 1])
bottom = np.zeros(3)
x_labels = ['Early (Ep 1-50)', 'Mid (Ep 200-250)', 'Late (Ep 450-500)']
for i, row in enumerate(action_history):
    ax_action.bar(x_labels, row, bottom=bottom, label=actions[i])
    bottom += row
ax_action.set_ylabel('Action Probability')
ax_action.set_title('Learned Policy Evolution')
ax_action.legend(title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')

# Thêm nhãn
ax_map.text(-0.05, 1.02, '(a)', transform=ax_map.transAxes, fontsize=14, fontweight='bold')
ax_reward.text(-0.25, 1.1, '(b)', transform=ax_reward.transAxes, fontsize=14, fontweight='bold')
ax_action.text(-0.25, 1.1, '(c)', transform=ax_action.transAxes, fontsize=14, fontweight='bold')

plt.savefig('figure_template_5_uav_drl.pdf')
print("Đã lưu 'figure_template_5_uav_drl.pdf'")
