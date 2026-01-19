import streamlit as st
import time
import pandas as pd

# 1. 设置网页标题
st.set_page_config(page_title="股票数据下载", layout="wide")

st.title("🚀 股票数据下载器")

# --- 核心修改：把输入框放在主界面，而不是侧边栏 ---
st.subheader("第一步：输入参数")

# 这里创建输入框，默认留空
stock_code = st.text_input("请输入股票代码 (例如: 600519, AAPL)", value="")

# --- 运行按钮 ---
# 只有当用户点击按钮时，才去检查有没有输入代码
if st.button('▶️ 开始运行并获取数据', type="primary", use_container_width=True):
    
    # 检查用户是否填了代码
    if not stock_code:
        st.error("❌ 请先输入股票代码，然后再点运行！")
    else:
        with st.spinner(f'正在搜索代码为 [{stock_code}] 的数据...'):
            
            # === 在这里放入你真实的股票爬虫逻辑 ===
            # 例如: df = get_stock_data(stock_code)
            
            # (这里是模拟演示)
            time.sleep(1.5) 
            data = {
                '股票代码': [stock_code, stock_code, stock_code],
                '交易日期': ['2023-10-01', '2023-10-02', '2023-10-03'],
                '收盘价': [100.5, 102.3, 101.8],
                '涨跌幅': ['+0.5%', '+1.8%', '-0.5%']
            }
            df = pd.DataFrame(data)
            csv_data = df.to_csv(index=False).encode('utf-8')
            # ==================================

            st.success(f"✅ [{stock_code}] 数据获取成功！")

            # 显示结果预览
            st.dataframe(df, use_container_width=True)

            # 显示下载按钮
            st.download_button(
                label=f"📥 下载 {stock_code}.csv",
                data=csv_data,
                file_name=f"{stock_code}_data.csv",
                mime="text/csv",
                type="secondary"
            )