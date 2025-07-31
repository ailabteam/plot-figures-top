import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

print("Đang tạo Mẫu 14: Prediction Error Distribution Analysis (1x4 subplots)...")

# --- Dữ liệu giả định ---
np.random.seed(0)
# (a) ARIMA: Lỗi có phương sai lớn
arima_errors = np.random.normal(0, 5, 1000)
# (b) LSTM: Lỗi bị lệch (dự đoán thấp hơn -> lỗi dương)
lstm_errors = np.random.normal(2, 2.5, 1000)
# (c) GRU: Phương sai khá cao
gru_errors = np.random.normal(0, 4, 1000)
# (d) Our Model: Phân phối chuẩn, phương sai thấp
our_errors = np.random.normal(0, 1.5, 1000)

all_errors = [arima_errors, lstm_errors, gru_errors, our_errors]
titles = ['(a) ARIMA', '(b) LSTM (Baseline)', '(c) GRU (Baseline)', '(d) ChronoNet (Ours)']
colors = ['C0', 'C1', 'C2', 'C3']

# --- Vẽ đồ thị ---
plt.style.use(['science', 'grid', 'no-latex'])
# Chia sẻ trục y để dễ so sánh chiều cao của các phân phối
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, ax in enumerate(axes):
    # Sử dụng Seaborn để vẽ histogram và KDE một cách dễ dàng
    sns.histplot(all_errors[i], ax=ax, kde=True, color=colors[i], stat='density')
    ax.axvline(x=0, color='k', linestyle='--') # Thêm đường tại lỗi bằng 0
    mean_error = np.mean(all_errors[i])
    ax.axvline(x=mean_error, color='r', linestyle=':', label=f'Mean = {mean_error:.2f}')
    ax.set_title(titles[i])
    ax.set_xlabel('Prediction Error')
    ax.legend(fontsize='small')
    ax.set_xlim(-15, 15)

# Chỉ hiển thị nhãn trục y ở subplot đầu tiên
axes[0].set_ylabel('Density')

fig.suptitle("Comparison of Prediction Error Distributions", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figure_template_14_error_distribution.pdf')
print("Đã lưu 'figure_template_14_error_distribution.pdf'")
