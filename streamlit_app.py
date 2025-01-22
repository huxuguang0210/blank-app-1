import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 加载训练好的模型
model = joblib.load('svm_model.joblib')

# Streamlit 输入界面
st.title("生育结果预测")

# 创建两列布局，将手术术式选择框和提示按钮放在同一行
col1, col2 = st.columns([4, 1])  # 左列为下拉框，右列为按钮

# 手术方式
手术方式 = st.selectbox("手术方式", [0, 1])

# 手术术式选项
手术术式 = col1.selectbox("手术术式", [1, 2, 3])

# 提示按钮，点击后弹出信息
if col2.button("点击查看手术术式说明"):
    st.info("""
    1. 肿物切 = 1
    2. 一侧附件切 = 2
    3. 一侧+对侧肿物切 = 3
    """)

# 其他输入变量
肿物破裂 = st.selectbox("肿物破裂", [0, 1])
全面分期 = st.selectbox("全面分期", [0, 1])
清大网 = st.selectbox("清大网", [0, 1])
清淋巴 = st.selectbox("清淋巴", [0, 1])
分期 = st.selectbox("分期", [0, 1, 2, 3, 4])
单侧双侧 = st.selectbox("单侧/双侧", [0, 1])
肿瘤直径 = st.selectbox("肿瘤直径", [0, 1])

# 合成输入数据
input_data = [[手术方式, 手术术式, 肿物破裂, 全面分期, 清大网, 清淋巴, 分期, 单侧双侧, 肿瘤直径]]

# 预测
if st.button('预测生育结果'):
    prediction_prob = model.predict_proba(input_data)  # 返回预测概率
    prediction = model.predict(input_data)  # 返回预测类别
    fertility_result = '是' if prediction == 1 else '否'
    st.write(f"生育结果：{fertility_result}")
    st.write(f"生育概率：{prediction_prob[0][1]*100:.2f}%")

    # 显示列线图
    fig, ax = plt.subplots()
  
