import matplotlib.pyplot as plt
import numpy as np

print("Đang tạo Ví dụ 1: DRL Training Dynamics...")

# --- Dữ liệu giả định ---
timesteps = np.linspace(0, 500_000, 100)
# (a) Dữ liệu Reward
dqn_reward_mean = 80 + 150 * (1 - np.exp(-timesteps / 200_000)) + np.random.normal(0, 5, 100)
ourdqn_reward_mean = 100 + 180 * (1 - np.exp(-timesteps / 150_000)) + np.random.normal(0, 3, 100)
# (b) Dữ liệu Loss
dqn_loss = 1.5 * np.exp(-timesteps / 100_000) + 0.1 + np.random.normal(0, 0.05, 100)
ourdqn_loss = 1.2 * np.exp(-timesteps / 80_000) + 0.05 + np.random.normal(0, 0.02, 100)

# --- Vẽ đồ thị ---
plt.style.use(['science', 'grid', 'muted', 'no-latex'])
# Tạo figure với 2 hàng, 1 cột, chia sẻ trục x
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

# --- Subplot (a): Reward Curve ---
axes[0].plot(timesteps, dqn_reward_mean, label='DQN (Baseline)')
axes[0].plot(timesteps, ourdqn_reward_mean, label='OurDQN (Ours)')
axes[0].set_ylabel("Episodic Reward")
axes[0].legend()
axes[0].set_title("DRL Agent Performance Comparison")
axes[0].text(-0.1, 1.1, '(a)', transform=axes[0].transAxes, 
             fontsize=12, fontweight='bold', va='top', ha='right')

# --- Subplot (b): Loss Curve ---
axes[1].plot(timesteps, dqn_loss, label='DQN Loss')
axes[1].plot(timesteps, ourdqn_loss, label='OurDQN Loss')
axes[1].set_xlabel("Timesteps")
axes[1].set_ylabel("Q-Network Loss")
axes[1].legend()
axes[1].set_yscale('log') # Thường dùng thang log cho loss để thấy rõ sự thay đổi ban đầu
axes[1].text(-0.1, 1.1, '(b)', transform=axes[1].transAxes, 
             fontsize=12, fontweight='bold', va='top', ha='right')

# Định dạng trục x chung
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda x, p: f'{int(x/1e3)}k')
axes[1].xaxis.set_major_formatter(formatter)

plt.tight_layout(h_pad=0.5) # Điều chỉnh khoảng cách dọc
plt.savefig('example_1_drl_dynamics.pdf')
print("Đã lưu 'example_1_drl_dynamics.pdf'")
