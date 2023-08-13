import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Dự đoán  dùng mạng NEURAL nhân tạo ANN
# huynq@isi.com.vn - https://ai.matilda.vn

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data2.csv')

# Chuyển đổi cột "Date" thành dữ liệu thời gian (ví dụ: ngày tháng đếm từ một ngày gốc)
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = (data['Date'] - data['Date'].min()).dt.days

# Tách dữ liệu thành features (X) và labels (y)
X = data['Date'].values.reshape(-1, 1)
y = data['Value'].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Xây dựng mạng nơ-ron nhân tạo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Dự đoán giá trị cho ngày tiếp theo
next_date_value = data['Date'].max() + 1
next_date_value_scaled = scaler.transform(np.array([[next_date_value]]))
predicted_value = model.predict(next_date_value_scaled)[0][0]
print(f'Predicted Value for the next date: {predicted_value:.2f}')

if (predicted_value >= 0.5): 
    print('Dự đoán ván tiếp theo: 1')
else: 
    print('Dự đoán ván tiếp theo: 0')
