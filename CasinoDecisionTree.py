import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Đọc file CSV
data = pd.read_csv('arimax2.csv')

# Chia dữ liệu thành features (X) và labels (y)
X = data[['Date']]
y = data['Value']

# Chuyển đổi dữ liệu ngày thành số nguyên
X['Date'] = pd.to_datetime(X['Date'])
X['Date'] = X['Date'].map(pd.Timestamp.to_julian_date)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Dự đoán giá trị trong Date tiếp theo
next_date = pd.to_datetime('2023-08-15').to_julian_date()
next_value = model.predict([[next_date]])

print('Dự đoán giá trị trong Date tiếp theo:', next_value)
