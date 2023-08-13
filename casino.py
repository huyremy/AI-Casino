import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
# Dự đoán casino by huynq@isi.com.vn
# Đọc dữ liệu từ file CSV
df = pd.read_csv('arimax2.csv')

# Chuyển cột 'Date' thành kiểu dữ liệu ngày giờ
df['Date'] = pd.to_datetime(df['Date'])

# Sắp xếp dữ liệu theo thời gian
df.sort_values('Date', inplace=True)

# Chuẩn bị dữ liệu
data = df['Value'].values
target = np.roll(data, -1)

# Chia dữ liệu thành tập train và tập test
train_data = data[:-1]
train_target = target[:-1]
test_data = data[-1:]
test_target = target[-1:]

# Xây dựng mô hình Seq2Seq
model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Đào tạo mô hình
model.fit(train_data[:, np.newaxis, np.newaxis], train_target[:, np.newaxis], epochs=10, batch_size=1)

# Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(test_data[:, np.newaxis, np.newaxis], test_target[:, np.newaxis])
print('Độ chính xác trên tập test:', accuracy)

# Dự đoán giá trị máy chủ trả về
predictions = model.predict(test_data[:, np.newaxis, np.newaxis])
print('Predicted value:', predictions[0][0])

# Áp dụng ngưỡng
threshold = 0.5
binary_prediction = 1 if predictions[0][0] >= threshold else 0

print('Dự đoán ván tiếp theo:', binary_prediction)
