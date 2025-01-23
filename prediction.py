import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import norm

# 加载训练好的 SVM 模型
@st.cache_resource
def load_model():
    return joblib.load("svm_model.joblib")  # 替换为您的模型路径

model = load_model()

# 定义 95% 置信区间计算函数
def calculate_confidence_interval(proba, confidence=0.95):
    z = norm.ppf((1 + confidence) / 2)  # Z-score
    margin = z * np.sqrt((proba * (1 - proba)) / len(proba))
    lower_bound = np.clip(proba - margin, 0, 1)  # 确保概率区间在 [0, 1]
    upper_bound = np.clip(proba + margin, 0, 1)
    return lower_bound, upper_bound

# 单样本预测函数
def predict_single(data):
    proba = model.predict_proba([data])[0][1]  # 返回怀孕概率 (Pregnant Result)
    lower, upper = calculate_confidence_interval(np.array([proba]))
    return lower[0], upper[0]  # 返回区间

# 批量预测函数
def predict_batch(data):
    probabilities = model.predict_proba(data)[:, 1]  # 返回怀孕概率
    lower, upper = calculate_confidence_interval(probabilities)
    return lower, upper

# Streamlit 应用界面
st.title("怀孕结果预测系统")
st.write("基于SVM模型的预测系统，用于评估 `Pregnant Result` 的概率（以概率区间形式展示）。")

# 单样本输入
st.subheader("单样本预测")
with st.form("single_input"):
    col1, col2 = st.columns(2)
    with col1:
        surgical_method = st.selectbox("Surgical Method (0=0, 1=1)", [0, 1])
        surgical_procedure = st.selectbox("Surgical Procedure", [1, 2, 3])
        tumor_rupture = st.selectbox("Tumor Rupture (0=No, 1=Yes)", [0, 1])
        comprehensive_staging = st.selectbox("Comprehensive Staging (0=No, 1=Yes)", [0, 1])
    with col2:
        omentum_resection = st.selectbox("Omentum Resection (0=No, 1=Yes)", [0, 1])
        lymphadenectomy = st.selectbox("Lymphadenectomy (0=No, 1=Yes)", [0, 1])
        staging = st.selectbox("Staging", [0, 1, 2, 3, 4])
        unilateral_bilateral = st.selectbox("Unilateral/Bilateral", [0, 1])
        tumor_diameter = st.selectbox("Tumor Diameter (Diameter ≥7 cm=1, <7 cm=0)", [0, 1])
    
    submit_single = st.form_submit_button("预测")

if submit_single:
    input_data = [
        surgical_method,
        surgical_procedure,
        tumor_rupture,
        comprehensive_staging,
        omentum_resection,
        lymphadenectomy,
        staging,
        unilateral_bilateral,
        tumor_diameter
    ]
    lower, upper = predict_single(input_data)
    st.success(f"怀孕概率预测区间: [{lower:.2%}, {upper:.2%}]")

# 批量文件上传预测
st.subheader("批量预测")
uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("上传的数据：", data.head())

    if st.button("开始批量预测"):
        lower, upper = predict_batch(data.values)
        data['Lower CI'] = lower
        data['Upper CI'] = upper
        data['Predicted Interval'] = data.apply(
            lambda row: f"[{row['Lower CI']:.2%}, {row['Upper CI']:.2%}]", axis=1
        )
        st.write("预测结果：", data[['Predicted Interval']])
        
        # 下载预测结果
        csv = data.to_csv(index=False)
        st.download_button("下载预测结果", csv, "prediction_results.csv", "text/csv")

# 页面底部版权信息
st.markdown("---")
st.markdown("Copyright © Shengjing Hospital of China Medical University")
