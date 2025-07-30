import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

print("\nĐang tạo Mẫu 3: Beamforming Performance Analysis (2 subplots)...")

# --- Dữ liệu giả định ---
# (a) Radiation Pattern
theta = np.linspace(0, 2 * np.pi, 360)
target_angle = np.deg2rad(30)
# Static beam (rộng hơn, gain thấp hơn, có sidelobe)
static_gain = 10 * np.sinc(5 * (theta - target_angle))**2 + 0.5 * np.sin(10 * theta)**2
static_gain = np.maximum(static_gain, 0.1) # Gain không thể âm
# AdaBeam (hẹp, gain cao, sidelobe rất thấp)
adabeam_gain = 15 * np.exp(-((theta - target_angle) / np.deg2rad(10))**2) + 0.1
# (b) Throughput Distribution
np.random.seed(42)
data = {
    'Algorithm': np.repeat(['Static Beam', 'AdaBeam (Ours)'], 100),
    'System Throughput (Mbps)': np.concatenate([
        np.random.normal(loc=120, scale=30, size=100),
        np.random.normal(loc=180, scale=15, size=100)
    ])
}
df = pd.DataFrame(data)

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot (a): Radiation Pattern (Polar Plot) ---
# Tạo subplot (a) với projection là polar
axes[0] = fig.add_subplot(1, 2, 1, projection='polar')
axes[0].plot(theta, static_gain, '--', label='Static Beam')
axes[0].plot(theta, adabeam_gain, '-', label='AdaBeam (Ours)')
axes[0].set_theta_zero_location('N') # Đặt 0 độ ở phía trên
axes[0].set_thetamin(0)
axes[0].set_thetamax(360)
axes[0].set_title('Radiation Pattern (Gain)')
axes[0].legend()

# --- Subplot (b): Throughput Distribution ---
# Do subplot (a) được tạo lại, ta cần tạo lại subplot (b)
axes[1] = fig.add_subplot(1, 2, 2)
sns.boxplot(x='Algorithm', y='System Throughput (Mbps)', data=df, ax=axes[1])
axes[1].set_title('System Throughput Distribution')
axes[1].grid(axis='y', linestyle='--')

# Thêm nhãn
axes[0].text(-0.3, 1.1, '(a)', transform=axes[0].transAxes, fontsize=14, fontweight='bold')
axes[1].text(-0.15, 1.1, '(b)', transform=axes[1].transAxes, fontsize=14, fontweight='bold')

plt.tight_layout(w_pad=3)
plt.savefig('figure_template_3_beamforming.pdf')
print("Đã lưu 'figure_template_3_beamforming.pdf'")
