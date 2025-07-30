import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

print("\nĐang tạo Ví dụ 6 (Nâng cao): Satellite Handoff Analysis...")

# --- Dữ liệu giả định ---
time = np.linspace(0, 100, 200) # 100 giây quan sát
# (a) Dữ liệu không gian
sat_ground_track_x = time
sat_ground_track_y = np.ones_like(time) * 5
gs_a_pos = (10, 2)
gs_b_pos = (90, 2)
# (b) Dữ liệu tín hiệu (SNR - Signal-to-Noise Ratio)
# SNR từ GS-A giảm dần
snr_a = 25 - 0.4 * time + np.random.normal(0, 0.3, 200)
# SNR từ GS-B tăng dần
snr_b = -15 + 0.4 * time + np.random.normal(0, 0.3, 200)
# Thời điểm chuyển giao
baseline_handoff_time = 75 # Baseline chờ SNR của A quá thấp mới chuyển
predictive_handoff_time = 50 # Predictive chuyển giao đúng lúc SNR của A và B giao nhau

# --- Vẽ đồ thị ---
plt.style.use(['science', 'grid', 'no-latex'])
fig = plt.figure(figsize=(10, 8))
# Tạo layout 2x1 với tỉ lệ chiều cao 1:2
gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.1)

# --- Subplot (a): Spatial Context (Bản đồ) ---
ax_map = fig.add_subplot(gs[0, 0])
# Vẽ đường đi của vệ tinh
ax_map.plot(sat_ground_track_x, sat_ground_track_y, 'k--', label='Satellite Ground Track')
# Vẽ các trạm mặt đất
ax_map.plot(gs_a_pos[0], gs_a_pos[1], 'p', markersize=12, label='Ground Station A')
ax_map.plot(gs_b_pos[0], gs_b_pos[1], 'h', markersize=12, label='Ground Station B')
# Đánh dấu vị trí xảy ra handoff
ax_map.scatter(baseline_handoff_time, 5, s=100, facecolors='none', edgecolors='r', linewidth=2, label='Baseline HO Point')
ax_map.scatter(predictive_handoff_time, 5, s=100, c='g', marker='*', label='Predictive HO Point')

ax_map.set_title('Spatial Context of Satellite Handoff')
ax_map.set_ylabel('Y-Coord (km)')
ax_map.set_ylim(0, 8)
ax_map.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize='small')
ax_map.tick_params(axis='x', labelbottom=False) # Ẩn nhãn trục x vì nó dùng chung với subplot dưới
ax_map.text(-0.1, 1.1, '(a)', transform=ax_map.transAxes, fontsize=12, fontweight='bold')

# --- Subplot (b): Temporal Signal Analysis ---
ax_snr = fig.add_subplot(gs[1, 0], sharex=ax_map) # sharex để 2 trục x đồng bộ
# Vẽ đường SNR
ax_snr.plot(time, snr_a, label='SNR from GS-A')
ax_snr.plot(time, snr_b, label='SNR from GS-B')

# Đánh dấu thời điểm chuyển giao bằng đường thẳng đứng
ax_snr.axvline(x=baseline_handoff_time, color='r', linestyle='--', linewidth=2, label='Baseline Handoff')
ax_snr.axvline(x=predictive_handoff_time, color='g', linestyle='-', linewidth=2, label='Predictive Handoff (Ours)')

# Tô bóng vùng tín hiệu kém của Baseline
# Tìm thời điểm SNR_A cắt một ngưỡng thấp, ví dụ 5 dB
poor_signal_threshold = 5
start_poor_signal = time[np.where(snr_a < poor_signal_threshold)[0][0]]
ax_snr.axvspan(start_poor_signal, baseline_handoff_time, color='red', alpha=0.15, label='Poor Connection Period')

# Thêm chú thích chi tiết (annotation)
ax_snr.annotate('Late Handoff!\n(Signal drop)', 
                xy=(baseline_handoff_time, snr_a[np.argmin(np.abs(time - baseline_handoff_time))]), 
                xytext=(baseline_handoff_time + 5, -5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                ha='left', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.8))

ax_snr.annotate('Optimal Handoff', 
                xy=(predictive_handoff_time, 5), 
                xytext=(predictive_handoff_time - 25, 15),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                ha='center', va='center', fontsize=10)

ax_snr.set_xlabel('Time (s)')
ax_snr.set_ylabel('Signal-to-Noise Ratio (dB)')
ax_snr.legend(loc='center left', fontsize='small')
ax_snr.set_ylim(-20, 30)
ax_snr.text(-0.1, 1.05, '(b)', transform=ax_snr.transAxes, fontsize=12, fontweight='bold')

# --- Lưu Figure ---
plt.tight_layout(h_pad=0) # Điều chỉnh khoảng cách dọc
plt.savefig('example_6_satellite_handoff.pdf')
print("Đã lưu 'example_6_satellite_handoff.pdf'")
