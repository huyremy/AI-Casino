import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# Đường dẫn đến file CSV
csv_file = '/content/data3.csv'

# Đọc dữ liệu từ file CSV
data = pd.read_csv(csv_file)

# Chia dữ liệu thành features (X) và target (y)
X = data[['Date']]  # Lấy cột 'Date' làm feature
y = data[['Value']]   # Lấy cột 'Value' làm target

# Chuyển đổi cột 'Date' thành dạng số nguyên sử dụng hàm rank()
X['Date'] = X['Date'].rank(method='dense').astype(int).copy()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Random Forest Classifier
model = RandomForestClassifier()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán giá trị trong date tiếp theo
next_date = X['Date'].max() + 1
next_value = model.predict([[next_date]])

# In kết quả dự đoán
print(f"Predicted value for next date (Date {next_date}): {next_value[0]}")
