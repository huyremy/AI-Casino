import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Đọc dữ liệu từ file CSV
df = pd.read_csv('path_to_your_csv_file.csv')

# Chuyển cột 'timestamp' thành kiểu dữ liệu ngày giờ
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sắp xếp dữ liệu theo thời gian
df.sort_values('timestamp', inplace=True)

# Chuẩn bị dữ liệu
data = df['value'].values
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
