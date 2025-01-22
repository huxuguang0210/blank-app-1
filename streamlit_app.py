import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib

# 设置页面配置，必须是第一个命令
st.set_page_config(page_title="生育预测系统", page_icon=":female-doctor:", layout="wide")

# 添加中文支持的CSS
st.markdown("""
    <style>
        body {
            font-family: "Microsoft YaHei", "Arial", "SimHei", "Songti SC", "KaiTi", sans-serif;
        }
        h1, h2, h3, p {
            font-family: "Microsoft YaHei", "Arial", "SimHei", "Songti SC", "KaiTi", sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# 配置matplotlib的中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial', 'SimHei', 'Songti SC', 'KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, y_train

# 创建模型和标准化器
model, scaler, X_train_scaled, y_train = create_model()

# 页面头部
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">生育预测系统</h1>
    <p style="text-align: center; font-size: 18px;">根据手术信息预测生育概率</p>
    <hr style="border: 1px solid #4CAF50;">
""", unsafe_allow_html=True)

# 用户输入
st.sidebar.header('输入数据进行预测')

# 在侧边栏中创建输入框
手术方式 = st.sidebar.selectbox('手术方式', [0, 1])
手术术式 = st.sidebar.selectbox(
    '手术术式',
    [1, 2, 3],
    format_func=lambda x: {
        1: "肿物切",
        2: "一侧附件切",
        3: "一侧+对侧肿物切"
    }.get(x)
)

st.sidebar.markdown("""
    **手术术式说明：**
    - 肿物切 = 1
    - 一侧附件切 = 2
    - 一侧+对侧肿物切 = 3
""")

肿物破裂 = st.sidebar.selectbox('肿物破裂', [0, 1])
全面分期 = st.sidebar.selectbox('全面分期（腹水+大网+腹膜活检）', [0, 1])
清大网 = st.sidebar.selectbox('清大网', [0, 1])
清淋巴 = st.sidebar.selectbox('清淋巴', [0, 1])
分期 = st.sidebar.selectbox('分期', [0, 1, 2, 3, 4])
单侧双侧 = st.sidebar.selectbox('单侧/双侧', [0, 1])
肿瘤直径 = st.sidebar.selectbox('肿瘤直径（直径≥7）', [0, 1])

# 获取用户输入的数据并标准化
input_data = np.array([[手术方式, 手术术式, 肿物破裂, 全面分期, 清大网, 清淋巴, 分期, 单侧双侧, 肿瘤直径]])
input_data_scaled = scaler.transform(input_data)

# 使用SVM进行预测
prediction_prob = model.predict_proba(input_data_scaled)[0][1]  # 获取预测生育的概率

# 显示预测结果：生育与否及对应的概率
生育结果 = "是" if prediction_prob >= 0.5 else "否"
st.subheader('预测结果')
st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>生育结果: {生育结果} (概率: {prediction_prob:.2f})</h3>", unsafe_allow_html=True)

# 生成变量贡献率（Feature Importance）
st.subheader('变量贡献率')

# 计算各个特征的贡献率
result = permutation_importance(model, X_train_scaled, y_train, n_repeats=10, random_state=42)
importance = result.importances_mean

# 绘制贡献率图
features = ['手术方式', '手术术式', '肿物破裂', '全面分期', '清大网', '清淋巴', '分期', '单侧双侧', '肿瘤直径']
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(features, importance, color='skyblue')
ax.set_xlabel('贡献率')
ax.set_title('每个变量对生育概率的贡献率')

# 显示贡献率图
st.pyplot(fig)

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
        predictions_prob = model.predict_proba(input_data_scaled)[:, 1]
        
        # 将预测结果加入到原始数据
        input_df['生育概率'] = predictions_prob
        input_df['生育结果'] = ['是' if prob >= 0.5 else '否' for prob in predictions_prob]
        
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
