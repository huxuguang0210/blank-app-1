import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 加载训练好的模型
def load_model():
    return joblib.load("svm_model.joblib")

# 显示界面
def main():
    st.title("基于 SVM 的生育结果预测系统")

    st.sidebar.header("输入变量")
    手术方式 = st.sidebar.selectbox("手术方式", [0, 1])
    手术术式 = st.sidebar.selectbox("手术术式", [1, 2, 3])
    肿物破裂 = st.sidebar.selectbox("肿物破裂", [0, 1])
    全面分期 = st.sidebar.selectbox("全面分期", [0, 1])
    清大网 = st.sidebar.selectbox("清大网", [0, 1])
    清淋巴 = st.sidebar.selectbox("清淋巴", [0, 1])
    分期 = st.sidebar.selectbox("分期", [0, 1, 2, 3, 4])
    单侧双侧 = st.sidebar.selectbox("单侧/双侧", [0, 1])
    肿瘤直径 = st.sidebar.selectbox("肿瘤直径", [0, 1])

    model = load_model()
    
    st.subheader("模型信息")
    st.write("已加载训练好的 SVM 模型。")

    # 单例预测
    input_data = np.array([[手术方式, 手术术式, 肿物破裂, 全面分期, 清大网, 清淋巴, 分期, 单侧双侧, 肿瘤直径]])
    prediction = model.predict(input_data)[0]
    st.write(f"预测的生育结果: {'成功' if prediction == 1 else '失败'}")

    # 显示特征贡献率
    st.subheader("变量贡献率")
    feature_names = ["手术方式", "手术术式", "肿物破裂", "全面分期", "清大网", "清淋巴", "分期", "单侧/双侧", "肿瘤直径"]
    if hasattr(model, 'coef_'):
        feature_importances = model.coef_.flatten()
        importance_df = pd.DataFrame({"变量": feature_names, "贡献率": feature_importances})
        importance_df = importance_df.sort_values(by="贡献率", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="贡献率", y="变量", data=importance_df, palette="viridis")
        plt.title("特征贡献率")
        st.pyplot(plt)
    else:
        st.write("当前模型不支持特征贡献率显示。")

    # 批量预测
    st.subheader("批量预测")
    uploaded_file = st.file_uploader("上传 CSV 文件", type="csv")
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        predictions = model.predict(batch_data)
        batch_data["预测结果"] = predictions
        st.write(batch_data)
        st.download_button(
            label="下载预测结果", 
            data=batch_data.to_csv(index=False), 
            file_name="预测结果.csv", 
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
