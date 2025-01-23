import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load("svm_model.joblib")  # 替换为实际的模型文件路径

model = load_model()

# 主界面
st.title("生育预测模型")
st.write("基于支持向量机 (SVM) 的生育结果预测模型")
st.markdown("---")

# 输入表单
st.sidebar.header("输入参数")
surgical_method = st.sidebar.selectbox("Surgical Method", options=["0", "1"], format_func=lambda x: "0=0" if x == "0" else "1=1")
surgical_procedure = st.sidebar.selectbox(
    "Surgical Procedure", options=["1", "2", "3"],
    format_func=lambda x: "Tumor Resection=1" if x == "1" else ("Unilateral Adnexectomy=2" if x == "2" else "Unilateral + Contralateral Tumor Resection=3")
)
tumor_rupture = st.sidebar.selectbox("Tumor Rupture", options=["0", "1"], format_func=lambda x: "No=0" if x == "0" else "Yes=1")
comprehensive_staging = st.sidebar.selectbox(
    "Comprehensive Staging", options=["0", "1"], format_func=lambda x: "No=0" if x == "0" else "Yes=1"
)
omentum_resection = st.sidebar.selectbox("Omentum Resection", options=["0", "1"], format_func=lambda x: "No=0" if x == "0" else "Yes=1")
lymphadenectomy = st.sidebar.selectbox("Lymphadenectomy", options=["0", "1"], format_func=lambda x: "No=0" if x == "0" else "Yes=1")
staging = st.sidebar.selectbox(
    "Staging", options=["0", "1", "2", "3", "4"],
    format_func=lambda x: ["Stage IA=0", "Stage IB=1", "Stage IC=2", "Stage II=3", "Stage III=4"][int(x)]
)
unilateral_bilateral = st.sidebar.selectbox(
    "Unilateral/Bilateral", options=["0", "1"], format_func=lambda x: "Unilateral=0" if x == "0" else "Bilateral=1"
)
tumor_diameter = st.sidebar.selectbox(
    "Tumor Diameter", options=["0", "1"], format_func=lambda x: "<7 cm=0" if x == "0" else "≥7 cm=1"
)

# 输入参数转为数值
input_data = np.array([
    int(surgical_method), int(surgical_procedure), int(tumor_rupture),
    int(comprehensive_staging), int(omentum_resection), int(lymphadenectomy),
    int(staging), int(unilateral_bilateral), int(tumor_diameter)
]).reshape(1, -1)

# 在线预测
if st.sidebar.button("预测结果"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    result = "Pregnant" if prediction == 1 else "Not Pregnant"
    st.write(f"### 预测结果: {result}")
    st.write(f"怀孕概率: {prediction_proba[1]:.2f}")
    st.write(f"未怀孕概率: {prediction_proba[0]:.2f}")

    # 绘制列线图
    fig, ax = plt.subplots()
    ax.bar(["Not Pregnant", "Pregnant"], prediction_proba, color=["blue", "green"])
    ax.set_ylabel("概率")
    ax.set_title("预测概率")
    st.pyplot(fig)

# 批量文件上传
st.markdown("---")
st.header("批量预测")
uploaded_file = st.file_uploader("上传 CSV 文件", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("上传的数据：")
    st.dataframe(data)

    if st.button("批量预测"):
        predictions = model.predict(data)
        prediction_proba = model.predict_proba(data)
        data["Prediction"] = ["Pregnant" if p == 1 else "Not Pregnant" for p in predictions]
        data["Pregnant Probability"] = prediction_proba[:, 1]
        data["Not Pregnant Probability"] = prediction_proba[:, 0]
        st.write("预测结果：")
        st.dataframe(data)

        # 下载结果文件
        result_csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("下载预测结果", data=result_csv, file_name="predictions.csv", mime="text/csv")

# 版权信息
st.markdown("---")
st.markdown("版权所有 © 中国医科大学附属盛京医院")
