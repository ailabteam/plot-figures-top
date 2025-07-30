import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("Đang tạo Mẫu 6: Deep Dive into Multi-Class Learning...")

# --- Dữ liệu giả định ---
epochs = np.arange(1, 51)
num_classes = 10
class_names = [f'Class {i}' for i in range(num_classes)]
class_names[0], class_names[1], class_names[8], class_names[9] = 'Cat', 'Dog', 'Car', 'Truck' # Ví dụ lớp dễ và khó

# (a) Overall Accuracy
overall_accuracy = 0.9 - 0.7 * np.exp(-epochs / 10)

# (b) Class-wise accuracy history (Heatmap data)
class_acc_history = np.zeros((num_classes, len(epochs)))
# Tạo ra các tốc độ học khác nhau cho các lớp khác nhau
learning_speeds = np.linspace(5, 20, num_classes)
np.random.shuffle(learning_speeds)
for i in range(num_classes):
    class_acc_history[i, :] = 0.95 * (1 - np.exp(-epochs / learning_speeds[i]))

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.5]})

# --- Subplot (a): Overall Training Accuracy ---
axes[0].plot(epochs, overall_accuracy, 'C0-')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Overall Accuracy')
axes[0].set_title('Global Model Performance')
axes[0].grid(True)
axes[0].set_ylim(0, 1)

# --- Subplot (b): Class-wise Learning Trajectory ---
im = sns.heatmap(class_acc_history, cmap='magma', ax=axes[1], cbar_kws={'label': 'Per-Class Accuracy'})
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Class')
axes[1].set_yticklabels(class_names, rotation=0)
axes[1].set_title('Class-wise Learning Dynamics')

# Thêm nhãn
axes[0].text(-0.15, 1.05, '(a)', transform=axes[0].transAxes, fontsize=14, fontweight='bold')
axes[1].text(-0.1, 1.05, '(b)', transform=axes[1].transAxes, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figure_template_6_multiclass_dynamics.pdf')
print("Đã lưu 'figure_template_6_multiclass_dynamics.pdf'")
