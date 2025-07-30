import matplotlib.pyplot as plt
import numpy as np

# --- Dữ liệu giả định ---
epochs = np.arange(1, 101)
# Dữ liệu cho subplot (a) - Accuracy
acc = 0.98 - 0.4 * np.exp(-epochs / 20)
# Dữ liệu cho subplot (b) - Loss
loss = 0.05 + 0.8 * np.exp(-epochs / 15)
#--------------------------------

# Áp dụng style ACM cho khác biệt
plt.style.use(['science', 'acm'])

# Tạo một figure chứa 1 hàng, 2 cột subplot
# figsize được điều chỉnh cho 2 subplot
fig, axes = plt.subplots(1, 2, figsize=(10, 4)) 

# --- Subplot (a): Accuracy ---
axes[0].plot(epochs, acc, color='C0')
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].set_ylim(0.5, 1.0)
axes[0].grid(True)
# Thêm nhãn (a) cho chuyên nghiệp
axes[0].text(-0.1, 1.1, '(a)', transform=axes[0].transAxes, 
            fontsize=12, fontweight='bold', va='top', ha='right')


# --- Subplot (b): Loss ---
axes[1].plot(epochs, loss, color='C1')
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Loss")
axes[1].set_title("Model Loss")
axes[1].set_ylim(0, 1.0)
axes[1].grid(True)
# Thêm nhãn (b)
axes[1].text(-0.1, 1.1, '(b)', transform=axes[1].transAxes, 
            fontsize=12, fontweight='bold', va='top', ha='right')


# Tự động điều chỉnh khoảng cách giữa các subplot
plt.tight_layout()

# Lưu lại dưới dạng file vector PDF
print("Đang lưu figure 'multi_subplot_example.pdf'...")
plt.savefig('multi_subplot_example.pdf', format='pdf')
print("Đã lưu xong.")
