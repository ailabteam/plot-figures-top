import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

print("Đang tạo Mẫu 13: AI Image Restoration Pipeline (1x4 subplots)...")

# --- Dữ liệu giả định ---
# Tạo một ảnh "Ground Truth" gốc
grid_size = 64
gt_image = np.zeros((grid_size, grid_size))
gt_image[10:25, 10:25] = 1 # Hình vuông
gt_image[35:55, 40:60] = 0.7 # Hình chữ nhật mờ hơn
gt_image[28:38, 28:38] = 0.3 # Vùng trung tâm

# (a) Input Image: Thêm nhiễu Gauss
noisy_image = gt_image + np.random.normal(0, 0.2, gt_image.shape)
noisy_image = np.clip(noisy_image, 0, 1)

# (b) Gaussian Filter: Làm mờ
gaussian_result = gaussian_filter(noisy_image, sigma=1.5)

# (c) Baseline CNN: Gần giống GT nhưng có lỗi
baseline_cnn_result = gt_image.copy()
baseline_cnn_result[50:55, 10:20] = 0.5 # Thêm artifact

# (d) Our Model: Gần như hoàn hảo
our_result = gt_image.copy()

images = [noisy_image, gaussian_result, baseline_cnn_result, our_result]
titles = ['(a) Noisy Input', '(b) Gaussian Filter', '(c) Baseline CNN', '(d) AI-Restore (Ours)']

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(titles[i])
    ax.set_axis_off() # Rất quan trọng cho việc hiển thị ảnh

fig.suptitle("Qualitative Comparison of Image Denoising Results", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figure_template_13_image_restoration.pdf')
print("Đã lưu 'figure_template_13_image_restoration.pdf'")
