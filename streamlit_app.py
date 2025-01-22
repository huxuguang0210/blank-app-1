import pandas as pd
import numpy as np
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 模拟训练数据
def create_model():
    # 示例数据：手术方式、手术术式、肿物破裂等
    X_train = np.array([
        [0, 1, 0, 1, 1, 0, 0, 0, 1],
        [1, 2, 1, 0, 0, 1, 1, 1, 0],
        [0, 3, 0, 1, 1, 0, 2, 0, 1],
        [1, 1, 0, 1, 1, 1, 3, 1, 0],
        [0, 2, 1, 0, 1, 1, 0, 0, 1]
    ])
    y_train = np.array([0, 1, 0, 1, 1])  # 是否生育：0=否, 1=是
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 使用支持向量机（SVM）训练模型
    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# 创建模型和标准化器
model, scaler = create_model()

# Streamlit页面
st.set_page_config(page_title="是否生育预测", page_icon=":female-doctor:", layout="wide")

# 页面头部
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">生育预测系统</h1>
    <p style="text-align: center; font-size: 18px;">根据手术信息预测是否能生育</p>
    <hr style="border: 1px solid #4CAF50;">
""", unsafe_allow_html=True)

# 用户输入
st.sidebar.header('输入数据进行预测')

# 在侧边栏中创建输入框
手术方式 = st.sidebar.selectbox('手术方式', [0, 1])
手术术式 = st.sidebar.selectbox('手术术式', [1, 2, 3])
肿物破裂 = st.sidebar.selectbox('肿物破裂', [0, 1])
全面分期 = st.sidebar.selectbox('全面分期', [0, 1])
清大网 = st.sidebar.selectbox('清大网', [0, 1])
清淋巴 = st.sidebar.selectbox('清淋巴', [0, 1])
分期 = st.sidebar.selectbox('分期', [0, 1, 2, 3, 4])
单侧双侧 = st.sidebar.selectbox('单侧/双侧', [0, 1])
肿瘤直径 = st.sidebar.selectbox('肿瘤直径', [0, 1])

# 获取用户输入的数据并标准化
input_data = np.array([[手术方式, 手术术式, 肿物破裂, 全面分期, 清大网, 清淋巴, 分期, 单侧双侧, 肿瘤直径]])
input_data_scaled = scaler.transform(input_data)

# 使用SVM进行预测
prediction = model.predict(input_data_scaled)

# 显示预测结果
st.subheader('预测结果')
if prediction == 1:
    st.markdown("<h3 style='text-align: center; color: #4CAF50;'>预测结果：可以生育</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='text-align: center; color: #FF6347;'>预测结果：不能生育</h3>", unsafe_allow_html=True)

# 支持批量上传数据
st.subheader('上传CSV文件进行批量预测')

# 文件上传
uploaded_file = st.file_uploader("选择CSV文件上传", type=["csv"])
if uploaded_file is not None:
    # 读取文件
    input_df = pd.read_csv(uploaded_file)
    
    # 确保上传的数据格式正确
    expected_columns = ['手术方式', '手术术式', '肿物破裂', '全面分期', '清大网', '清淋巴', '分期', '单侧双侧', '肿瘤直径']
    if set(input_df.columns) == set(expected_columns):
        # 标准化数据
        input_data_scaled = scaler.transform(input_df)
        
        # 进行预测
        predictions = model.predict(input_data_scaled)
        
        # 将预测结果加入到原始数据
        input_df['是否生育'] = ['是' if p == 1 else '否' for p in predictions]
        
        # 显示预测结果
        st.write("预测结果：", input_df)
    else:
        st.error("上传的文件列名不匹配，请确保列名正确！")

# 页脚
st.markdown("""
    <hr style="border: 1px solid #4CAF50;">
    <footer style="text-align: center;">
        <p>© 2025 生育预测系统 | 由OpenAI提供支持</p>
    </footer>
""", unsafe_allow_html=True)
