import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Ellipse

print("Đang tạo Mẫu 11: Qualitative Prediction Results (2x6 subplots)...")

# --- Dữ liệu giả định ---
# Hàm tạo ra một "cảnh" (scene) giả định
def create_scene(scene_id):
    np.random.seed(scene_id)
    scene = np.zeros((100, 100)) # Nền đen
    # Thêm các đối tượng ngẫu nhiên
    for _ in range(np.random.randint(2, 5)):
        x, y, w, h = np.random.randint(10, 80, 4)
        scene[y:y+h, x:x+w] = 1 # Tòa nhà
    for _ in range(np.random.randint(1, 3)):
        x, y, w, h = np.random.randint(10, 80, 4)
        scene[y:y+h, x:x+w] = 2 # Cây cối
    return scene

# --- Vẽ đồ thị ---
plt.style.use(['science', 'no-latex'])
fig = plt.figure(figsize=(24, 7)) # Figure cực rộng
gs = GridSpec(2, 6, hspace=0.1, wspace=0.1) # Khoảng cách nhỏ

# Lặp qua 6 kịch bản
for i in range(6):
    # Tạo subplot cho Ground Truth
    ax_gt = fig.add_subplot(gs[0, i])
    # Tạo subplot cho Prediction
    ax_pred = fig.add_subplot(gs[1, i])
    
    # Tạo dữ liệu ảnh
    ground_truth = create_scene(i)
    prediction = ground_truth.copy()
    # Thêm một chút lỗi vào prediction cho thực tế
    if np.random.rand() > 0.5:
        x, y = np.random.randint(0, 90, 2)
        prediction[y:y+10, x:x+10] = 1 - prediction[y:y+10, x:x+10]

    # Vẽ ảnh
    ax_gt.imshow(ground_truth, cmap='terrain')
    ax_pred.imshow(prediction, cmap='terrain')
    
    # Dọn dẹp đồ thị
    ax_gt.set_axis_off()
    ax_pred.set_axis_off()
    
    # Thêm tiêu đề cho từng cột
    ax_gt.set_title(f'Scenario {i+1}')
    
    # Chỉ thêm nhãn hàng ở cột đầu tiên
    if i == 0:
        ax_gt.set_ylabel('Ground Truth', fontsize=14, labelpad=20)
        ax_pred.set_ylabel('Prediction (Ours)', fontsize=14, labelpad=20)

fig.suptitle("Qualitative Comparison of UAV Scene Prediction", fontsize=20, y=0.98)
plt.savefig('figure_template_11_qualitative_results.pdf', bbox_inches='tight')
print("Đã lưu 'figure_template_11_qualitative_results.pdf'")
