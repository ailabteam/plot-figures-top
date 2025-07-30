import matplotlib.pyplot as plt
import numpy as np

print("\nĐang tạo Figure 6: Donut Chart for Traffic Distribution...")

# --- Dữ liệu giả định ---
# Phân bổ các loại lưu lượng trong một mạng di động 5G
labels = ['Video Streaming', 'Web Browsing', 'Online Gaming', 'VoIP', 'Others']
sizes = [45, 25, 15, 10, 5]  # Tỷ lệ phần trăm
# Màu sắc tương ứng, có thể lấy từ các bộ màu có sẵn
colors = plt.get_cmap('Paired').colors

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig, ax = plt.subplots(figsize=(7, 7))

# Vẽ Pie Chart
# wedgeprops tạo ra các đường viền trắng giữa các miếng bánh
# pctdistance điều chỉnh vị trí của nhãn phần trăm
# autopct định dạng chuỗi hiển thị phần trăm
wedges, texts, autotexts = ax.pie(sizes, 
                                  autopct='%1.1f%%', 
                                  startangle=90,
                                  colors=colors,
                                  pctdistance=0.85,
                                  wedgeprops=dict(width=0.4, edgecolor='w'))

# --- Tinh chỉnh và chi tiết ---
# Thêm một vòng tròn trắng ở giữa để tạo thành Donut Chart
centre_circle = plt.Circle((0,0), 0.60, fc='white')
fig.gca().add_artist(centre_circle)

# Căn chỉnh lại các nhãn cho đẹp
plt.setp(autotexts, size=10, weight="bold", color='white')

# Thêm legend (chú thích) thay vì nhãn trực tiếp trên các miếng bánh
ax.legend(wedges, labels,
          title="Traffic Types",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

ax.set_title("5G Network Traffic Distribution by Type")

# Equal aspect ratio đảm bảo đồ thị là hình tròn.
ax.axis('equal')  

plt.tight_layout()
plt.savefig('figure_6_donutchart_traffic.pdf')
print("Đã lưu 'figure_6_donutchart_traffic.pdf'")
