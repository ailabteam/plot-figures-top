import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec # Dòng import đã được thêm vào

print("\nĐang tạo Ví dụ 5: Hyperparameter Sensitivity Analysis...")

# --- Dữ liệu giả định ---
# (a) Sensitivity to Learning Rate (khi batch_size = 64)
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
acc_vs_lr = [88.1, 90.5, 91.8, 92.1, 90.2, 85.4]

# (b) Sensitivity to Batch Size (khi lr = 1e-4)
batch_sizes = [8, 16, 32, 64, 128, 256]
acc_vs_bs = [86.7, 89.1, 90.9, 91.8, 91.5, 89.9]

# (c) 2D Performance Surface
# Tạo lưới tọa độ cho các trục
lr_grid_log = np.linspace(-5, -2.5, 50) # log10(learning_rate)
bs_grid_log = np.linspace(3, 8, 50)    # log2(batch_size)
LR_mesh, BS_log_mesh = np.meshgrid(10**lr_grid_log, bs_grid_log)
BS_mesh = 2**BS_log_mesh

# Hàm toán học mô phỏng bề mặt accuracy, có đỉnh ở lr=10^-3.5, bs=2^6.5
ACC_SURFACE = 92.5 - 2 * (np.log10(LR_mesh) - (-3.5))**2 - 0.5 * (np.log2(BS_mesh) - 6.5)**2 - np.abs(np.log2(BS_mesh)-6.5)*(np.log10(LR_mesh)-(-3.5))

# --- Vẽ đồ thị ---
plt.style.use(['science', 'grid', 'no-latex'])
fig = plt.figure(figsize=(12, 10))
# Sử dụng GridSpec để tạo layout 2 hàng 2 cột
gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

# --- Subplot (a): Learning Rate ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(learning_rates, acc_vs_lr, 'o-')
ax1.set_xscale('log') # Learning rate thường được vẽ trên thang log
ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('Final Accuracy (%)')
ax1.set_title('Sensitivity to Learning Rate')
ax1.text(-0.25, 1.1, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold')

# --- Subplot (b): Batch Size ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(batch_sizes, acc_vs_bs, 's-')
ax2.set_xscale('log', base=2) # Batch size cũng thường là log base 2
ax2.set_xlabel('Batch Size')
# ax2.set_ylabel('Final Accuracy (%)') # Có thể ẩn đi vì giống subplot (a)
ax2.set_title('Sensitivity to Batch Size')
ax2.text(-0.25, 1.1, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold')

# --- Subplot (c): 2D Surface ---
ax3 = fig.add_subplot(gs[1, :]) # Chiếm cả hàng dưới
contour = ax3.contourf(LR_mesh, BS_mesh, ACC_SURFACE, cmap='cividis', levels=15)
ax3.set_xscale('log')
ax3.set_yscale('log', base=2)
ax3.set_xlabel('Learning Rate')
ax3.set_ylabel('Batch Size')
ax3.set_title('Performance Surface')
cbar = fig.colorbar(contour, ax=ax3)
cbar.set_label('Accuracy (%)')
ax3.text(-0.15, 1.1, '(c)', transform=ax3.transAxes, fontsize=12, fontweight='bold')

# Tinh chỉnh lại layout một chút để tránh chồng chéo
# tight_layout không hoạt động tốt với colorbar của subplot lớn, 
# nên ta dùng GridSpec's hspace và wspace
# plt.tight_layout() # Có thể không cần khi đã dùng hspace, wspace

plt.savefig('example_5_sensitivity_analysis.pdf')
print("Đã lưu 'example_5_sensitivity_analysis.pdf'")
