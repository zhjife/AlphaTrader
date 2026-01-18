# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd
import numpy as np
from snownlp import SnowNLP
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

class AlphaGalaxyExcelSystem:
    def __init__(self, symbol):
        self.symbol = str(symbol)
        self.data = {}
        self.diagnosis = {"verdict": "观望", "risk_level": "中", "score": 0, "core_logic": []}
        self.metrics_list = []
        self.levels_list = []
        
        # 自动识别指数
        if self.symbol.startswith('6'):
            self.index_id = 'sh000001'; self.index_name = "上证指数"
        elif self.symbol.startswith('8') or self.symbol.startswith('4'):
            self.index_id = 'bj899050'; self.index_name = "北证50"
        else:
            self.index_id = 'sz399001'; self.index_name = "深证成指"

    def _fetch_data(self):
        print(f"🚀 正在提取 {self.symbol} 的全维数据...")
        try:
            spot = ak.stock_zh_a_spot_em()
            target_spot = spot[spot['代码'] == self.symbol]
            if target_spot.empty:
                print(f"❌ 错误：未找到代码 {self.symbol}，请检查代码是否正确。")
                return False
            self.data['spot'] = target_spot.iloc[0]
            self.data['all_spot'] = spot
            
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
            hist = ak.stock_zh_a_hist(symbol=self.symbol, period='daily', start_date=start, end_date=end, adjust='qfq')
            if hist is None or hist.empty:
                print("❌ 错误：无法获取历史K线数据。")
                return False
            hist.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume', '换手率':'turnover'}, inplace=True)
            self.data['hist'] = hist
            
            # --- 资金流容错处理 (关键修改) ---
            try:
                flow = ak.stock_individual_fund_flow(stock=self.symbol, market="sh" if self.symbol.startswith("6") else "sz")
                # 确保获取到了数据且不为空
                if flow is not None and not flow.empty:
                    self.data['flow'] = flow.sort_values('日期').tail(10)
                else:
                    self.data['flow'] = pd.DataFrame() # 给个空表
            except:
                self.data['flow'] = pd.DataFrame() # 报错也给空表
            
            try:
                self.data['news'] = ak.stock_news_em(symbol=self.symbol)
            except:
                self.data['news'] = pd.DataFrame()

            return True
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            return False

    def _add_metric(self, name, value, status, explanation, logic_desc):
        self.metrics_list.append({
            "指标名称": name, "当前数值": value, "状态判定": status,
            "大白话解释 (含义)": explanation, "判断理由 (AI分析)": logic_desc
        })

    def _analyze(self):
        hist = self.data['hist']
        spot = self.data['spot']
        flow = self.data['flow']
        close = hist['close'].iloc[-1]
        
        # 1. 趋势
        ma20 = hist['close'].rolling(20).mean().iloc[-1]
        ma60 = hist['close'].rolling(60).mean().iloc[-1]
        trend_status = "多头" if close > ma20 else "空头"
        trend_desc = "股价在月线之上，短线强势" if close > ma20 else "股价跌破月线，短线走弱"
        if close < ma60: trend_status = "破位"; trend_desc = "有效跌破60日生命线，中期趋势转坏"
        self._add_metric("趋势状态 (MA均线)", f"现价{close} / MA20:{round(ma20,2)}", trend_status, "判断股票是在爬山(多头)还是下山(空头)。", trend_desc)

        # 2. 筹码
        df_chip = hist.tail(120).copy()
        df_chip['avg'] = (df_chip['open'] + df_chip['close'])/2
        winner_vol = df_chip[df_chip['avg'] < close]['volume'].sum()
        total_vol = df_chip['volume'].sum()
        winner_pct = (winner_vol / total_vol * 100) if total_vol > 0 else 0
        chip_status = "中性"
        chip_logic = "多空博弈中，无极端情况"
        if winner_pct > 90: chip_status = "高危预警"; chip_logic = "90%的人都赚钱了，随时可能有人砸盘止盈"
        elif winner_pct < 10: chip_status = "冰点/超跌"; chip_logic = "90%的人被套牢，上方全是压力"
        self._add_metric("筹码获利盘", f"{int(winner_pct)}%", chip_status, "超过90%说明容易发生踩踏式卖出。", chip_logic)

        # 3. 资金 (完全容错逻辑 - 修复报错的核心)
        flow_val = 0
        flow_status = "数据缺失"
        flow_logic = "该股暂无实时主力资金流数据，跳过此项判断"
        
        # 只有当flow不为空，并且包含了'主力净流入净额'这一列时，才去计算
        if not flow.empty and '主力净流入净额' in flow.columns:
            try:
                net_flow_3d = flow['主力净流入净额'].iloc[-3:].sum()
                flow_val = round(net_flow_3d / 100000000, 2)
                flow_status = "流入" if flow_val > 0 else "流出"
                if flow_val < -1: flow_status = "主力出逃"
                elif flow_val > 1: flow_status = "主力抢筹"
                flow_logic = f"近3日累计净{'流入' if flow_val>0 else '流出'} {abs(flow_val)} 亿"
            except:
                pass # 如果计算出错，保持默认值
        
        self._add_metric("主力资金 (近3日)", f"{flow_val} 亿元", flow_status, "股价涨但资金流出是诱多；股价跌但资金流入是洗盘。", flow_logic)

        # 4. 排名
        my_pct = spot['涨跌幅']
        all_stocks = self.data['all_spot']
        valid = all_stocks[~all_stocks['名称'].str.contains('ST|退')]
        rank = valid[valid['涨跌幅'] > my_pct].shape[0]
        percentile = 100 - (rank / len(valid) * 100)
        rps_status = "弱势"
        if percentile > 90: rps_status = "龙头/领涨"
        elif percentile > 70: rps_status = "强势"
        elif percentile < 30: rps_status = "滞涨/被抛弃"
        self._add_metric("全市场排名 (RPS)", f"击败了 {int(percentile)}% 的股票", rps_status, "机构只喜欢买前10%的优等生。", f"今日涨幅 {my_pct}%，处于市场{rps_status}地位")

        # 5. 乖离
        bias = (close - ma60) / ma60 * 100
        bias_status = "正常"
        if bias > 20: bias_status = "严重超买"
        elif bias < -20: bias_status = "严重超跌"
        self._add_metric("乖离率 (橡皮筋)", f"{int(bias)}%", bias_status, "正太多(>20%)说明涨过头了；负太多(<-20%)说明跌过头了。", f"当前偏离60日线 {int(bias)}%，{bias_status}")

        # 6. 止损
        hist['tr'] = np.maximum(hist['high'] - hist['low'], abs(hist['high'] - hist['close'].shift(1)))
        atr = hist['tr'].rolling(14).mean().iloc[-1]
        stop_price = close - 2 * atr
        self._add_metric("动态止损价", f"{round(stop_price, 2)}", "生命线", "如果收盘跌破这个价格，必须无脑卖出保命。", f"跌破 {round(stop_price, 2)} 建议离场")

        # 计算结论
        reasons = []
        if close < stop_price:
            self.diagnosis['verdict'] = "清仓卖出"; self.diagnosis['risk_level'] = "极高"; reasons.append("股价跌破ATR动态止损位，趋势反转。")
        elif trend_status == "破位":
            self.diagnosis['verdict'] = "清仓/离场"; self.diagnosis['risk_level'] = "高"; reasons.append("有效跌破60日生命线，机构多头格局破坏。")
        elif winner_pct > 95:
            self.diagnosis['verdict'] = "止盈/减仓"; self.diagnosis['risk_level'] = "中高"; reasons.append("获利盘极度拥挤(>95%)，防止主力高位兑现。")
        elif flow_val < -1 and trend_status == "多头":
            self.diagnosis['verdict'] = "逢高减仓"; self.diagnosis['risk_level'] = "中"; reasons.append("量价背离：股价在高位，但主力资金在大幅流出。")
        elif trend_status == "多头" and flow_val > 0:
            self.diagnosis['verdict'] = "持有/买入"; self.diagnosis['risk_level'] = "低"; reasons.append("趋势向上，且主力资金持续流入，状态健康。")
        else:
            self.diagnosis['verdict'] = "观望"; self.diagnosis['risk_level'] = "中"; reasons.append("多空平衡，无明显方向，建议等待。")
        self.diagnosis['core_logic'] = reasons

        self._calc_levels(close, stop_price)

    def _calc_levels(self, close, stop):
        self.levels_list.append(["🔴 动态止损 (Hard Stop)", round(stop, 2), "跌破此位无条件清仓"])
        df = self.data['hist']
        levels = {"MA20 (月线)": df['close'].rolling(20).mean().iloc[-1], "MA60 (机构成本)": df['close'].rolling(60).mean().iloc[-1], "近20日高点": df['high'].iloc[-20:].max()}
        for k, v in levels.items():
            if v > close: self.levels_list.append(["🔴 上方压力 (Resistance)", round(v, 2), k])
            else: self.levels_list.append(["🟢 下方支撑 (Support)", round(v, 2), k])

    def save_excel(self):
        if not self._fetch_data(): return
        self._analyze()
        filename = f"{self.symbol}_{self.data['spot']['名称']}_诊断.xlsx"
        print(f"💾 正在生成 Excel 文件: {filename} ...")
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            summary_data = [
                ["股票代码", self.symbol], ["股票名称", self.data['spot']['名称']],
                ["当前价格", self.data['spot']['最新价']], ["今日涨跌", f"{self.data['spot']['涨跌幅']}%"],
                ["", ""], ["🤖 最终建议", self.diagnosis['verdict']],
                ["🔥 风险等级", self.diagnosis['risk_level']], ["💡 核心理由", "\n".join(self.diagnosis['core_logic'])]
            ]
            pd.DataFrame(summary_data, columns=["项目", "内容"]).to_excel(writer, sheet_name='1.总览诊断', index=False)
            pd.DataFrame(self.metrics_list)[["指标名称", "当前数值", "状态判定", "判断理由 (AI分析)", "大白话解释 (含义)"]].to_excel(writer, sheet_name='2.指标深度解读', index=False)
            df_lv = pd.DataFrame(self.levels_list, columns=["类型", "价格", "说明"]); df_lv.sort_values(by="价格", ascending=False, inplace=True)
            df_lv.to_excel(writer, sheet_name='3.实战点位雷达', index=False)
        print(f"\n✅ 成功！报告已生成: {filename}")
        print("请在左侧文件列表中右键点击该文件，选择 'Download' 下载查看。")

if __name__ == "__main__":
    print("="*50)
    print("欢迎使用 Alpha Galaxy 机构诊断系统 (GitHub版)")
    print("="*50)
    code = input("请输入股票代码 (例如 600519): ").strip()
    if code:
        system = AlphaGalaxyExcelSystem(code)
        system.save_excel()
    else:
        print("未输入代码，程序退出。")
