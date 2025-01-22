import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 加载训练好的模型
model = joblib.load('svm_model.joblib')

# Streamlit 输入界面
st.title("生育结果预测")

# 手术方式输入框
st.write("提示：手术方式：0 = 0，1 = 1")
手术方式 = st.selectbox("手术方式", [0, 1])

# 手术术式输入框
st.write("""
提示：手术术式：
1. 肿物切 = 1
2. 一侧附件切 = 2
3. 一侧+对侧肿物切 = 3
""")
手术术式 = st.selectbox("手术术式", [1, 2, 3])

# 肿物破裂输入框
st.write("提示：肿物破裂：0 = 否，1 = 是")
肿物破裂 = st.selectbox("肿物破裂", [0, 1])

# 全面分期输入框
st.write("提示：全面分期：0 = 否（没有腹水、大网、腹膜活检），1 = 是（有这些检查）")
全面分期 = st.selectbox("全面分期", [0, 1])

# 清大网输入框
st.write("提示：清大网：0 = 否，1 = 是")
清大网 = st.selectbox("清大网", [0, 1])

# 清淋巴输入框
st.write("提示：清淋巴：0 = 否，1 = 是")
清淋巴 = st.selectbox("清淋巴", [0, 1])

# 分期输入框
st.write("""
提示：分期：
0 = IA期，1 = IB期，2 = IC期，3 = II期，4 = III期
""")
分期 = st.selectbox("分期", [0, 1, 2, 3, 4])

# 单侧/双侧输入框
st.write("提示：单侧/双侧：0 = 单侧，1 = 双侧")
单侧双侧 = st.selectbox("单侧/双侧", [0, 1])

# 肿瘤直径输入框
st.write("提示：肿瘤直径：0 = 直径＜7，1 = 直径≥7")
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
    ax.bar([0, 1], prediction_prob[0], tick_label=["否", "是"], color=["red", "green"])
    ax.set_ylabel('预测概率')
    ax.set_title('生育结果预测概率')
    st.pyplot(fig)

# 批量上传文件
st.title("批量上传文件")
uploaded_file = st.file_uploader("上传 CSV 文件", type="csv")
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    predictions = model.predict(batch_data[['手术方式', '手术术式', '肿物破裂', '全面分期', '清大网', '清淋巴', '分期', '单侧双侧', '肿瘤直径']])
    batch_data['生育结果'] = ['是' if x == 1 else '否' for x in predictions]
    st.write(batch_data)

# 添加版权所有信息
st.markdown("版权所有 © 2025")
