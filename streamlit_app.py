import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 设置页面标题和背景颜色
st.set_page_config(page_title="生育概率预测", page_icon="💡", layout="wide")
st.markdown("""
    <style>
        .css-1d391kg {background-color: #f0f8ff;}
        .css-1v3fvcr {color: #444444; font-weight: bold;}
        .css-1lsf32v {background-color: #ffcc00; padding: 10px; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# 样本数据（可以替换为实际数据）
data = {
    '手术方式': [0, 1, 0, 1, 0],
    '手术术式': [1, 2, 1, 3, 2],
    '肿物破裂': [0, 1, 0, 1, 0],
    '全面分期': [1, 0, 1, 0, 1],
    '清大网': [1, 0, 1, 1, 0],
    '清淋巴': [0, 1, 0, 1, 0],
    '分期': [1, 2, 3, 0, 2],
    '单侧/双侧': [0, 1, 0, 1, 1],
    '肿瘤直径': [1, 0, 0, 1, 1],
    '生育概率': [0.7, 0.6, 0.4, 0.9, 0.5]  # 假设为生育概率的标签
}

df = pd.DataFrame(data)

# 准备数据
X = df.drop('生育概率', axis=1)  # 特征
y = df['生育概率']  # 目标变量

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集（用于训练）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用SVM模型
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# 在测试集上进行预测并显示准确度
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit界面
st.title("🌟 生育概率预测 🌟")
st.markdown("<p class='css-1lsf32v'>请输入各项变量以预测生育概率</p>", unsafe_allow_html=True)

st.sidebar.header("🔧 输入变量")
手术方式 = st.sidebar.selectbox('手术方式 (0=0, 1=1)', [0, 1])
手术术式 = st.sidebar.selectbox('手术术式 (肿物切=1, 一侧附件切=2, 一侧+对侧肿物切=3)', [1, 2, 3])
肿物破裂 = st.sidebar.selectbox('肿物破裂 (0=否, 1=是)', [0, 1])
全面分期 = st.sidebar.selectbox('全面分期 (0=否, 1=是)', [0, 1])
清大网 = st.sidebar.selectbox('清大网 (0=否, 1=是)', [0, 1])
清淋巴 = st.sidebar.selectbox('清淋巴 (0=否, 1=是)', [0, 1])
分期 = st.sidebar.selectbox('分期 (IA期=0, IB期=1, IC期=2, II期=3, III期=4)', [0, 1, 2, 3, 4])
单侧双侧 = st.sidebar.selectbox('单侧/双侧 (单侧=0, 双侧=1)', [0, 1])
肿瘤直径 = st.sidebar.selectbox('肿瘤直径 (≥7是1, ＜7是0)', [0, 1])

# 创建用户输入的特征
input_data = np.array([[手术方式, 手术术式, 肿物破裂, 全面分期, 清大网, 清淋巴, 分期, 单侧双侧, 肿瘤直径]])
input_data_scaled = scaler.transform(input_data)

# 进行预测
predicted_probability = model.predict_proba(input_data_scaled)[0][1]

# 显示预测结果
st.subheader("🔮 预测结果")
st.write(f"预测的生育概率为: **{predicted_probability:.2f}**")

# 计算每个变量的贡献率（通过模型系数）
coefficients = model.coef_.flatten()
features = X.columns
contributions = pd.DataFrame({
    '特征': features,
    '系数': coefficients
})

# 绘制列线图
fig, ax = plt.subplots(figsize=(10, 6))
contributions.plot.bar(x='特征', y='系数', ax=ax, color='#1f77b4', legend=False)
ax.set_title('各变量对生育概率的贡献率', fontsize=16)
ax.set_ylabel('系数', fontsize=12)
ax.set_xlabel('特征', fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# 美化图表
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
st.pyplot(fig)

# 显示模型准确率
st.sidebar.markdown(f"**模型的测试集准确率: {accuracy:.2f}**")

# 美化按钮和输入框样式
st.markdown("""
    <style>
        .stButton button {
            background-color: #ffcc00;
            color: black;
            font-size: 16px;
            border-radius: 10px;
            width: 100%;
            height: 50px;
        }
        .stButton button:hover {
            background-color: #ff9900;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 12px;
            color: #888888;
        }
    </style>
""", unsafe_allow_html=True)

# 添加版权信息
st.markdown('<div class="footer">© 2025 版权所有 | 开发者: Your Name</div>', unsafe_allow_html=True)
