import matplotlib.pyplot as plt
import numpy as np

print("\nĐang tạo Figure 2: Bar Plot for Network Latency...")

# --- Dữ liệu giả định ---
protocols = ['LEO-Routing v1', 'LEO-Routing v2', 'Ground-Relay', 'Our-Protocol']
# Độ trễ trung bình (ms)
mean_latency = [85, 72, 110, 65]
# Độ lệch chuẩn
std_latency = [8, 6, 12, 5]

# --- Vẽ đồ thị ---
plt.style.use(['science', 'ieee', 'no-latex'])
fig, ax = plt.subplots(figsize=(7, 5))

bar_colors = ['#88CCEE', '#DDCC77', '#AA4499', '#332288'] # Sử dụng màu tùy chỉnh

# Vẽ các cột với thanh lỗi
bars = ax.bar(protocols, mean_latency, yerr=std_latency, 
              capsize=5, # Kích thước của "mũ" trên thanh lỗi
              color=bar_colors,
              edgecolor='black')

# --- Tinh chỉnh và chi tiết ---
ax.set_ylabel("End-to-End Latency (ms)")
ax.set_title("Latency in LEO Satellite Constellation")
ax.set_ylim(0, 140)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Thêm nhãn giá trị trên mỗi cột cho rõ ràng
ax.bar_label(bars, fmt='%.1f', padding=3)

# Xoay nhãn trục x nếu chúng quá dài và chồng chéo
plt.xticks(rotation=15, ha="right")

plt.tight_layout()
plt.savefig('figure_2_barplot_network.pdf')
print("Đã lưu 'figure_2_barplot_network.pdf'")
