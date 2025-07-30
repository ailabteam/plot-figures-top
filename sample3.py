import matplotlib.pyplot as plt
import numpy as np

print("\nĐang tạo Ví dụ 3: Satellite Network Latency Analysis...")

# --- Dữ liệu giả định ---
protocols = ['Baseline', 'LEO-Handoff', 'Our-Routing']
# (a) Latency Components (ms)
propagation = np.array([40, 40, 40]) # Giống nhau
processing = np.array([10, 8, 5])
queuing = np.array([50, 25, 10]) # Khác biệt lớn nhất ở đây
labels = ['Propagation', 'Processing', 'Queuing']
# (b) Latency vs. Load
network_load = np.linspace(0.2, 1.0, 8) # Từ 20% đến 100%
baseline_latency = 100 + 80 * network_load**2
our_latency = 55 + 25 * network_load

# --- Vẽ đồ thị ---
plt.style.use(['science', 'grid', 'no-latex'])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Subplot (a): Latency Breakdown (Stacked Bar Chart) ---
bottom = np.zeros(len(protocols))
colors = plt.get_cmap('viridis')(np.linspace(0, 1, 3))
for i, component in enumerate([propagation, processing, queuing]):
    p = axes[0].bar(protocols, component, label=labels[i], bottom=bottom, color=colors[i])
    bottom += component
    axes[0].bar_label(p, label_type='center', color='white', weight='bold')

axes[0].set_ylabel('Latency Component (ms)')
axes[0].set_title('End-to-End Latency Breakdown')
axes[0].legend(title='Component')
axes[0].text(-0.1, 1.1, '(a)', transform=axes[0].transAxes, fontsize=12, fontweight='bold')

# --- Subplot (b): Latency Scaling ---
axes[1].plot(network_load * 100, baseline_latency, 'o-', label='Baseline')
axes[1].plot(network_load * 100, our_latency, 's-', label='Our-Routing')
axes[1].set_xlabel('Network Load (%)')
axes[1].set_ylabel('Total E2E Latency (ms)')
axes[1].set_title('Latency vs. Network Load')
axes[1].legend()
axes[1].text(-0.1, 1.1, '(b)', transform=axes[1].transAxes, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('example_3_satellite_latency.pdf')
print("Đã lưu 'example_3_satellite_latency.pdf'")
