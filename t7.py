import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

print("Đang tạo Mẫu 7: Deep Dive into DRL Exploration (4 subplots)...")

# --- Dữ liệu giả định ---
timesteps = np.linspace(0, 1e6, 200)
# (a) Reward Curve
ppo_reward = 300 * (1 - np.exp(-timesteps / 3e5))
eppo_reward = 350 * (1 - np.exp(-timesteps / 4e5)) + 10*np.sin(timesteps/1e5)
# (b) Policy Entropy (thước đo sự ngẫu nhiên/khám phá)
ppo_entropy = 1.0 * np.exp(-timesteps / 1.5e5) + 0.1
eppo_entropy = 1.2 * np.exp(-timesteps / 2.5e5) + 0.15 # Giảm chậm hơn
# (c) State Visitation (Bản đồ nhiệt 20x20)
ppo_visitation = np.zeros((20, 20))
ppo_visitation[5:15, 5:10] = np.random.rand(10, 5) # Bị kẹt ở một vùng
eppo_visitation = np.random.rand(20, 20) * 0.7 # Khám phá rộng hơn
eppo_visitation[10:18, 10:18] += 0.3 # Vẫn có vùng tập trung
# (d) Final Success Rate
success_rates = {'PPO': 78, 'E-PPO (Ours)': 95}

# --- Vẽ đồ thị ---
plt.style.use(['science', 'ieee', 'no-latex'])
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Subplot (a): Reward Comparison ---
axes[0, 0].plot(timesteps, ppo_reward, '--', label='PPO')
axes[0, 0].plot(timesteps, eppo_reward, '-', label='E-PPO (Ours)')
axes[0, 0].set_xlabel('Timesteps')
axes[0, 0].set_ylabel('Average Reward')
axes[0, 0].set_title('Overall Performance')
axes[0, 0].legend()
axes[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# --- Subplot (b): Policy Entropy ---
axes[0, 1].plot(timesteps, ppo_entropy, '--', label='PPO')
axes[0, 1].plot(timesteps, eppo_entropy, '-', label='E-PPO (Ours)')
axes[0, 1].set_xlabel('Timesteps')
axes[0, 1].set_ylabel('Policy Entropy')
axes[0, 1].set_title('Exploration Strategy')
axes[0, 1].legend()
axes[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# --- Subplot (c): State Visitation Heatmap ---
# Cần tạo một GridSpec con để có không gian cho colorbar
gs_c = axes[1, 0].get_subplotspec().subgridspec(1, 2, width_ratios=[10, 1])
ax_heatmap = fig.add_subplot(gs_c[0, 0])
ax_cbar = fig.add_subplot(gs_c[0, 1])
im = ax_heatmap.imshow(eppo_visitation, cmap='inferno')
fig.colorbar(im, cax=ax_cbar)
ax_heatmap.set_title('State Visitation Frequency (E-PPO)')
ax_heatmap.set_xticks([])
ax_heatmap.set_yticks([])

# --- Subplot (d): Final Success Rate ---
axes[1, 1].bar(success_rates.keys(), success_rates.values(), color=['C0', 'C1'])
axes[1, 1].set_ylabel('Success Rate (%)')
axes[1, 1].set_title('Task Completion (Final 100 Episodes)')
axes[1, 1].set_ylim(70, 100)

# Thêm nhãn
subplot_labels = ['(a)', '(b)', '(c)', '(d)']
axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
for ax, label in zip(axes_flat, subplot_labels):
    ax.text(-0.15, 1.1, label, transform=ax.transAxes, fontsize=14, fontweight='bold')

plt.tight_layout(pad=1.5)
plt.savefig('figure_template_7_drl_exploration.pdf')
print("Đã lưu 'figure_template_7_drl_exploration.pdf'")
