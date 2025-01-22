import pandas as pd
import numpy as np
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 创建一些模拟数据
def create_data():
    # 生成一些模拟数据，包含上述输入变量
    data = pd.DataFrame({
        '手术方式': np.random.choice([0, 1], 100),
        '手术术式': np.random.choice([1, 2, 3], 100),
        '肿物破裂': np.random.choice([0, 1], 100),
        '全面分期': np.random.choice([0, 1], 100),
        '清大网': np.random.choice([0, 1], 100),
        '清淋巴': np.random.choice([0, 1], 100),
        '分期': np.random.choice([0, 1, 2, 3, 4], 100),
        '单侧/双侧': np.random.choice([0, 1], 100),
        '肿瘤直径': np.random.choice([0, 1], 100)
    })
    data['是否生育'] = np.random.choice([0, 1], 100)  # 输出变量：是否生育
    return data

# 数据预处理
def preprocess_data(data):
    X = data.drop('是否生育', axis=1)
    y = data['是否生育']
    return X, y

# 训练SVM模型
def train_svm(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, scaler, accuracy

# Streamlit界面
def predict_with_svm(model, scaler, input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

# 主函数
def main():
    st.title("是否生育预测")
    
    # 数据加载和模型训练
    data = create_data()
    X, y = preprocess_data(data)
    model, scaler, accuracy = train_svm(X, y)
    st.write(f"模型训练准确率: {accuracy:.2f}")

    st.subheader("上传数据进行预测")
    
    # 文件上传功能
    uploaded_file = st.file_uploader("选择CSV文件上传", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("上传的数据", input_data)
        
        # 确保输入数据格式正确
        if set(input_data.columns) == set(X.columns):
            prediction = predict_with_svm(model, scaler, input_data)
            result = ["是" if p == 1 else "否" for p in prediction]
            st.write("预测结果:", result)
        else:
            st.error("上传的数据格式错误，请确保列名与训练数据一致。")

# 启动Streamlit
if __name__ == "__main__":
    main()
