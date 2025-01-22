import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt


def main():
    st.title("生育概率预测")
    # 用户输入组件
    surgery_type = st.selectbox("手术方式", options=[0, 1])
    surgery_method = st.selectbox("手术术式", options=[1, 2, 3])
    rupture = st.selectbox("肿物破裂", options=[0, 1])
    staging = st.selectbox("全面分期", options=[0, 1])
    omentum_clear = st.selectbox("清大网", options=[0, 1])
    lymph_clear = st.selectbox("清淋巴", options=[0, 1])
    stage = st.selectbox("分期", options=[0, 1, 2, 3, 4])
    side = st.selectbox("单侧/双侧", options=[0, 1])
    tumor_diameter = st.selectbox("肿瘤直径", options=[0, 1])

    if st.button("预测生育概率"):
        # 整理输入数据
        input_data = np.array([surgery_type, surgery_method, rupture, staging, omentum_clear, lymph_clear, stage, side, tumor_diameter]).reshape(1, -1)
        # 数据预处理
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)
        # 这里假设我们已经有一个训练好的模型，实际使用中需要先训练模型
        model = load_trained_model()
        # 预测
        prediction = model.predict(input_data)
        st.write(f"预测的生育概率为: {prediction[0]}")

        # 显示列线图和变量贡献率
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(input_data)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig)


def load_trained_model():
    # 这里只是一个示例，实际中需要使用真实数据进行训练
    X = np.random.rand(100, 9)
    y = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    return grid.best_estimator_


if __name__ == "__main__":
    main()
