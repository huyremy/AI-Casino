from sklearn import svm
import numpy as np

# Dữ liệu huấn luyện
X_train = np.array([[2, 1], [1, 1], [6, 1], [4, 4]]) 
# Trong [2, 1] thì 2 là số bàn thắng đội nhà đã ghi trong những lần va chạm trước 1 là số bàn thắng khách ghi
#tương tự với [1, 1], [6, 1] [4,4] là kết quả của những trận đấu trước đó
y_train = np.array(['X', '1', '2', 'X'])

# Tạo mô hình SVM và huấn luyện
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Dữ liệu kiểm tra
X_test = np.array([[5, 1]])
# Trong [5, 1] là tỉ số dự đoán chung cuộc 5 - 1 nghiêng về đội nhà
# Dự đoán kèo 1X2 cho dữ liệu kiểm tra
predictions = model.predict(X_test)

# In kết quả
print(f"Trận đấu {i+1}: {pred}")
    # kết quả 1 = đội chủ nhà thắng với tỉ số 0 - 1
    # kết quả X =  hoà với tỉ số  0 - 1
    # kết quả 2 =  đội khách thắng với tỉ số 0 - 1  
