import matplotlib.pyplot as plt
import numpy as np

print("Đang tạo Figure 1: DRL Performance Comparison...")

# --- Dữ liệu giả định ---
# Thể hiện kết quả qua 1 triệu timesteps, trung bình trên 5 lần chạy
timesteps = np.linspace(0, 1_000_000, 100)

# Thuật toán PPO (ổn định, học nhanh ban đầu)
ppo_mean = 250 * (1 - np.exp(-timesteps / 400_000)) + np.random.randn(100) * 5
ppo_std = 20 + 15 * np.exp(-timesteps / 500_000)

# Thuật toán SAC (học chậm hơn nhưng đạt reward cao hơn)
sac_mean = 300 * (1 - np.exp(-timesteps / 600_000)) + np.random.randn(100) * 8
sac_std = 30 + 10 * np.exp(-timesteps / 700_000)

# Thuật toán của chúng ta (OurDRL - hội tụ nhanh và tốt hơn)
ourdrl_mean = 320 * (1 - np.exp(-timesteps / 350_000))
ourdrl_std = 15 + 10 * np.exp(-timesteps / 400_000)

# --- Vẽ đồ thị ---
# Style 'muted' cung cấp các màu đẹp, dịu mắt
plt.style.use(['science', 'grid', 'muted', 'no-latex'])
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Vẽ PPO
ax.plot(timesteps, ppo_mean, label='PPO')
ax.fill_between(timesteps, ppo_mean - ppo_std, ppo_mean + ppo_std, alpha=0.2)

# Vẽ SAC
ax.plot(timesteps, sac_mean, label='SAC')
ax.fill_between(timesteps, sac_mean - sac_std, sac_mean + sac_std, alpha=0.2)

# Vẽ thuật toán của chúng ta
ax.plot(timesteps, ourdrl_mean, label='OurDRL (Ours)')
ax.fill_between(timesteps, ourdrl_mean - ourdrl_std, ourdrl_mean + ourdrl_std, alpha=0.2)

# --- Tinh chỉnh và chi tiết ---
ax.set_xlabel("Timesteps")
ax.set_ylabel("Average Episodic Reward")
ax.set_title("Performance on Satellite Control Environment")
ax.legend(title='Algorithm', loc='lower right')
ax.set_xlim(0, 1_000_000)
ax.set_ylim(0, 400)

# Định dạng lại trục x cho dễ đọc
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda x, p: f'{int(x/1e6)}M' if x > 0 else '0')
ax.xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig('figure_1_lineplot_drl.pdf')
print("Đã lưu 'figure_1_lineplot_drl.pdf'")
