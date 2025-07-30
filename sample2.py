import matplotlib.pyplot as plt
import numpy as np

print("\nĐang tạo Ví dụ 2: Holistic Optimization Performance...")

# --- Dữ liệu giả định ---
algorithms = ['Simulated Annealing', 'Genetic Algorithm', 'Our-Opt']
# (a) Solution Quality (càng cao càng tốt)
quality_mean = [18.5, 22.1, 25.3]
quality_std = [2.1, 1.5, 0.8]
# (b) Convergence Time (càng thấp càng tốt)
time_mean = [350, 280, 150]
time_std = [40, 30, 20]
# (c) CPU Usage (càng thấp càng tốt)
cpu_mean = [75, 90, 45]
cpu_std = [8, 12, 5]

# --- Vẽ đồ thị ---
plt.style.use(['science', 'ieee', 'no-latex'])
fig, axes = plt.subplots(1, 3, figsize=(15, 4)) # Figure rộng hơn
colors = ['#88CCEE', '#DDCC77', '#332288']

# --- Subplot (a): Solution Quality ---
axes[0].bar(algorithms, quality_mean, yerr=quality_std, color=colors, capsize=4)
axes[0].set_ylabel("Solution Quality Score")
axes[0].set_title("Final Solution")
axes[0].text(-0.1, 1.1, '(a)', transform=axes[0].transAxes, fontsize=12, fontweight='bold')

# --- Subplot (b): Convergence Time ---
axes[1].bar(algorithms, time_mean, yerr=time_std, color=colors, capsize=4)
axes[1].set_ylabel("Time (seconds)")
axes[1].set_title("Convergence Speed")
axes[1].text(-0.1, 1.1, '(b)', transform=axes[1].transAxes, fontsize=12, fontweight='bold')

# --- Subplot (c): CPU Usage ---
axes[2].bar(algorithms, cpu_mean, yerr=cpu_std, color=colors, capsize=4)
axes[2].set_ylabel("CPU Usage (%)")
axes[2].set_title("Computational Cost")
axes[2].text(-0.1, 1.1, '(c)', transform=axes[2].transAxes, fontsize=12, fontweight='bold')

# Xoay nhãn cho tất cả các subplot
for ax in axes:
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.suptitle("Overall Performance Comparison of Optimization Algorithms", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Điều chỉnh để title không chồng lên subplot
plt.savefig('example_2_optimization_holistic.pdf')
print("Đã lưu 'example_2_optimization_holistic.pdf'")
