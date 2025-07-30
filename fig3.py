import matplotlib.pyplot as plt
import numpy as np

# --- Dữ liệu giả định ---
models = ['ResNet18', 'MobileNetV2', 'EfficientNet-B0']
# Thời gian thực thi trung bình (giây)
mean_times = np.array([0.05, 0.02, 0.035])
# Độ lệch chuẩn
std_times = np.array([0.008, 0.003, 0.005])
#--------------------------------

# Sử dụng style của Nature
plt.style.use(['science', 'nature'])

fig, ax = plt.subplots(figsize=(6, 5))

# Vẽ đồ thị cột
# yerr=std_times sẽ tự động vẽ thanh lỗi
# capsize thêm các "mũ" ở đầu thanh lỗi cho đẹp hơn
ax.bar(models, mean_times, yerr=std_times, capsize=5,
       align='center', alpha=0.8, ecolor='black')

ax.set_ylabel("Average Inference Time (s)")
ax.set_title("Model Inference Speed Comparison")
# Đặt giới hạn trục y bắt đầu từ 0
ax.set_ylim(0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Xoay nhãn trục x nếu chúng quá dài
# plt.xticks(rotation=15)

plt.tight_layout()

# Lưu lại file PDF
print("Đang lưu figure 'barplot_errorbar_example.pdf'...")
plt.savefig('barplot_errorbar_example.pdf', format='pdf')
print("Đã lưu xong.")
