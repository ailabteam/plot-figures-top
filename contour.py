import matplotlib.pyplot as plt
import numpy as np

print("\nĐang tạo Figure 5: Contour Plot for Loss Landscape...")

# --- Dữ liệu giả định ---
# Tạo một lưới (grid) các giá trị cho 2 trọng số w1 và w2
w1 = np.linspace(-2.0, 2.0, 100)
w2 = np.linspace(-2.0, 2.0, 100)
W1, W2 = np.meshgrid(w1, w2)

# Định nghĩa một hàm mất mát (loss function) đơn giản có nhiều local minima
# Ví dụ: Hàm Rastrigin - một hàm benchmark phổ biến trong tối ưu
Loss = (W1**2 - 10 * np.cos(2 * np.pi * W1)) + \
       (W2**2 - 10 * np.cos(2 * np.pi * W2)) + 20

# Giả định đường đi của thuật toán tối ưu (ví dụ: Gradient Descent)
path_w1 = np.array([1.8, 1.5, 1.1, 0.6, 0.2, 0.05, 0.01])
path_w2 = np.array([-1.7, -1.2, -0.8, -0.3, -0.1, -0.02, 0.0])

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig, ax = plt.subplots(figsize=(7, 6))

# Vẽ các đường đồng mức được tô màu (filled contour)
# 20 là số lượng mức, cmap là bảng màu
contourf = ax.contourf(W1, W2, Loss, levels=20, cmap='plasma')

# Vẽ các đường viền của đường đồng mức để làm rõ hơn
contour_lines = ax.contour(W1, W2, Loss, levels=20, colors='white', linewidths=0.5, alpha=0.7)

# Thêm thanh màu (colorbar) để chú giải giá trị
cbar = fig.colorbar(contourf, ax=ax)
cbar.set_label('Loss Value')

# Vẽ điểm bắt đầu và đường đi của thuật toán tối ưu
ax.plot(path_w1, path_w2, 'o-', color='black', linewidth=2, 
        markersize=5, label='Optimization Path')
# Đánh dấu điểm global minimum
ax.plot(0, 0, '*', color='red', markersize=15, label='Global Minimum')


# --- Tinh chỉnh và chi tiết ---
ax.set_xlabel('Weight 1 ($w_1$)')
ax.set_ylabel('Weight 2 ($w_2$)')
ax.set_title('Loss Landscape and Optimization Trajectory')
ax.legend(loc='upper right')
ax.set_aspect('equal', adjustable='box') # Giữ tỉ lệ 1:1 cho các trục

plt.tight_layout()
plt.savefig('figure_5_contour_loss.pdf')
print("Đã lưu 'figure_5_contour_loss.pdf'")
