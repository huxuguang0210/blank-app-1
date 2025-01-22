import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据生成函数
def generate_sample_data():
    np.random.seed(42)
    data = {
        "手术方式": np.random.randint(0, 2, 100),
        "手术术式": np.random.choice([1, 2, 3], 100),
        "肿物破裂": np.random.randint(0, 2, 100),
        "全面分期": np.random.randint(0, 2, 100),
        "清大网": np.random.randint(0, 2, 100),
        "清淋巴": np.random.randint(0, 2, 100),
        "分期": np.random.choice([0, 1, 2, 3, 4], 100),
        "单侧/双侧": np.random.randint(0, 2, 100),
        "肿瘤直径": np.random.randint(0, 2, 100),
        "生育结果": np.random.randint(0, 2, 100),
    }
    return pd.DataFrame(data)

# 训练模型
def train_model(data):
    X = data.drop(columns=["生育结果"])
    y = data["生育结果"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用 SVM 模型
    model = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
    model.fit(X_train, y_train)

    # 模型性能
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, model.named_steps['svc'].coef_.flatten()

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

    sample_data = generate_sample_data()
    model, accuracy, feature_importances = train_model(sample_data)

    st.write(f"模型准确率: {accuracy:.2f}")

    # 单例预测
    input_data = np.array([[手术方式, 手术术式, 肿物破裂, 全面分期, 清大网, 清淋巴, 分期, 单侧双侧, 肿瘤直径]])
    prediction = model.predict(input_data)[0]
    st.write(f"预测的生育结果: {'成功' if prediction == 1 else '失败'}")

    # 显示特征贡献率
    st.subheader("变量贡献率")
    feature_names = ["手术方式", "手术术式", "肿物破裂", "全面分期", "清大网", "清淋巴", "分期", "单侧/双侧", "肿瘤直径"]
    importance_df = pd.DataFrame({"变量": feature_names, "贡献率": feature_importances})
    importance_df = importance_df.sort_values(by="贡献率", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="贡献率", y="变量", data=importance_df, palette="viridis")
    plt.title("特征贡献率")
    st.pyplot(plt)

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
