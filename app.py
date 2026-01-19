import streamlit as st
import time
import pandas as pd

# --- 1. 设置网页为宽屏模式 (大屏体验) ---
st.set_page_config(page_title="Codespaces 控制台", layout="wide")

st.title("🚀 Codespaces 任务控制中心")
st.markdown("---") # 分割线

# --- 2. 左侧栏：参数设置 (可选) ---
with st.sidebar:
    st.header("设置")
    user_input = st.text_input("输入一些参数(例如文件名前缀):", "my_data")

# --- 3. 主区域：运行按钮 ---
st.subheader("1. 执行任务")
st.write("点击下方按钮开始运行服务器端脚本...")

if st.button('▶️ 开始运行代码', type="primary", use_container_width=True):
    
    # 显示加载状态
    with st.spinner('正在 Codespaces 中疯狂计算中...'):
        
        # === 在这里替换为你真实的代码逻辑 ===
        time.sleep(2) # 模拟耗时操作
        
        # 假设我们生成了一些数据 (这里用 DataFrame 举例)
        data = {
            'ID': [1, 2, 3, 4],
            '名称': ['任务A', '任务B', '任务C', '任务D'],
            '结果': ['成功', '成功', '失败', '成功'],
            '备注': [f'来自用户输入: {user_input}'] * 4
        }
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False).encode('utf-8')
        # ==================================

        st.success("✅ 任务执行成功！")

        # --- 4. 显示结果预览 ---
        st.subheader("2. 结果预览")
        st.dataframe(df, use_container_width=True)

        # --- 5. 提供下载功能 ---
        st.subheader("3. 获取文件")
        
        # 核心功能：下载按钮
        st.download_button(
            label="📥 点击下载结果 (result.csv)",
            data=csv_data,
            file_name=f"{user_input}_result.csv",
            mime="text/csv",
            type="primary" # 按钮样式
        )