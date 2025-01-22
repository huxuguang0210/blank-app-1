import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# 页面配置
st.set_page_config(page_title="癌症复发预测", page_icon="🎯", layout="centered")

# 页面标题和简介
st.markdown('<h1 style="text-align: center; color: #FF6347;">🎯 癌症复发预测 - 动态列线图</h1>', unsafe_allow_html=True)
st.markdown("""
通过输入患者信息，实时训练模型并预测复发概率。  
支持批量预测及特征贡献分析，助力临床决策优化。
""")

# **数据模拟和训练逻辑**
@st.cache_data
def generate_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'Age': np.random.randint(30, 80, 200),
        'Tumor_Size': np.random.uniform(1, 5, 200),
        'Treatment_Type': np.random.choice([0, 1], 200),
        'Gender': np.random.choice([0, 1], 200),
        'Smoking_History': np.random.choice([0, 1], 200),
        'Stage': np.random.choice([1, 2, 3, 4], 200),
        'Family_History': np.random.choice([0, 1], 200),
        'Chemotherapy': np.random.choice([0, 1], 200),
        'Age_at_Diagnosis': np.random.randint(30, 80, 200),
        'Previous_Surgeries': np.random.choice([0, 1], 200),
        'Recurrence': np.random.choice([0, 1], 200)
    })
    return data

@st.cache_data
def train_model(data):
    X = data.drop(columns=['Recurrence'])
    y = data['Recurrence']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = SVC(probability=True, kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model, scaler, X.columns.tolist()

# 数据生成与模型训练
data = generate_data()
svm_model, scaler, features = train_model(data)

# 用户输入区域
st.markdown('<h2 style="color: #4B0082;">📝 输入患者信息</h2>', unsafe_allow_html=True)
with st.form("patient_form", clear_on_submit=True):
    user_input = []
    for feature in features:
        value = st.number_input(f"{feature}:", min_value=0.0, max_value=100.0, step=1.0, key=feature)
        user_input.append(value)
    submit_button = st.form_submit_button(label="提交")

# 单个患者预测
if submit_button:
    if any([v is None for v in user_input]):
        st.warning("请填写所有字段！")
    else:
        user_input_np = np.array(user_input).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input_np)
        probability = svm_model.predict_proba(user_input_scaled)[0][1]

        st.markdown('<h2 style="color: #2E8B57;">📊 预测结果</h2>', unsafe_allow_html=True)
        st.subheader(f"**预测复发概率: {probability * 100:.2f}%**")

        coefficients = svm_model.coef_[0]
        contributions = coefficients * user_input_scaled[0]

        fig = go.Figure(go.Bar(
            x=contributions,
            y=features,
            orientation='h',
            marker=dict(color=contributions, colorscale='Viridis', colorbar=dict(title="贡献值"))
        ))

        fig.update_layout(
            title="特征对复发概率的贡献",
            xaxis_title="贡献值",
            yaxis_title="特征",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig)

# 批量预测
st.markdown('<h2 style="color: #4B0082;">📂 批量预测</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("上传患者数据文件 (CSV)", type=["csv"])
if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    if list(batch_data.columns) != features:
        st.error("文件的列名与模型所需特征不匹配！")
    else:
        batch_scaled = scaler.transform(batch_data)
        batch_predictions = svm_model.predict_proba(batch_scaled)[:, 1]
        batch_data["复发概率 (%)"] = batch_predictions * 100

        st.dataframe(batch_data)
        csv = batch_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="下载预测结果",
            data=csv,
            file_name="batch_predictions.csv",
            mime="text/csv",
        )
