# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd
import numpy as np
from snownlp import SnowNLP
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

class AlphaGalaxyFusionSystem:
    def __init__(self, symbol):
        self.symbol = str(symbol)
        self.data = {}
        self.report = {
            "verdict": "观望", "risk_level": "中", 
            "fusion_signals": [], # 存储共振信号
            "kelly_pos": 0, "logic": []
        }
        self.metrics = []
        self.levels = []
        
        # 指数映射
        if self.symbol.startswith('6'): self.index_id = 'sh000001'
        elif self.symbol.startswith('8') or self.symbol.startswith('4'): self.index_id = 'bj899050'
        else: self.index_id = 'sz399001'

    def _fetch_data(self):
        print(f"🚀 [多因子共振启动] 正在深度扫描 {self.symbol} ...")
        try:
            spot = ak.stock_zh_a_spot_em()
            target = spot[spot['代码'] == self.symbol]
            if target.empty: return False
            self.data['spot'] = target.iloc[0]
            
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d")
            
            # K线
            hist = ak.stock_zh_a_hist(symbol=self.symbol, period='daily', start_date=start, end_date=end, adjust='qfq')
            if hist is None or hist.empty: return False
            hist.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume', '换手率':'turnover'}, inplace=True)
            self.data['hist'] = hist

            # 资金流
            try:
                flow = ak.stock_individual_fund_flow(stock=self.symbol, market="sh" if self.symbol.startswith("6") else "sz")
                self.data['flow'] = flow.sort_values('日期').tail(10) if (flow is not None and not flow.empty) else pd.DataFrame()
            except: self.data['flow'] = pd.DataFrame()
            
            # 舆情
            try: self.data['news'] = ak.stock_news_em(symbol=self.symbol)
            except: self.data['news'] = pd.DataFrame()

            return True
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            return False

    # ================= 🧮 指标计算 =================
    def _calc_indicators(self, df):
        # 1. 基础均线
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 2. 量比 & 换手 (核心逻辑)
        # 量比 = (今日成交量 / 5日均量)
        df['vol_ma5'] = df['volume'].rolling(5).mean().shift(1)
        df['vol_ratio'] = df['volume'] / df['vol_ma5']
        # 换手率已在数据中: turnover
        
        # 3. MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        
        # 4. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 5. BOLL
        df['up'] = df['ma20'] + 2 * df['close'].rolling(20).std()
        df['dn'] = df['ma20'] - 2 * df['close'].rolling(20).std()
        
        # 6. ATR & 回撤
        df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
        df['atr'] = df['tr'].rolling(14).mean()
        roll_max = df['close'].rolling(250, min_periods=1).max()
        df['drawdown'] = (df['close'] / roll_max) - 1.0
        
        return df

    # ================= 🔗 共振分析 (Fusion Analysis) =================
    def _analyze_fusion(self, curr, flow_val, stop_price):
        signals = []
        logic = []
        
        # --- 组合A: 量比 + 换手 + 位置 ---
        # 逻辑：位置低，放量，换手活跃 -> 启动
        is_low_pos = curr['close'] < curr['ma60'] * 1.1 # 离60日线不远
        is_high_pos = curr['close'] > curr['ma60'] * 1.3 # 涨幅已大
        
        if is_low_pos and curr['vol_ratio'] > 1.8 and 3 < curr['turnover'] < 10:
            signals.append("🔥 底部放量启动")
            logic.append("✅ [共振A] 底部区域 + 量比放大(>1.8) + 换手健康 = 主力建仓/试盘。")
        
        elif is_high_pos and curr['turnover'] > 15 and curr['close'] < curr['open']:
            signals.append("⚠️ 高位滞涨出货")
            logic.append("❌ [共振A] 高位 + 巨量换手(>15%) + 收阴 = 主力可能趁乱出货。")
            
        elif curr['close'] > curr['ma20'] and curr['vol_ratio'] < 0.8 and curr['turnover'] < 3:
            signals.append("🔒 缩量锁筹上涨")
            logic.append("✅ [共振A] 股价上涨 + 量比缩小 + 换手低 = 筹码锁定良好，主力控盘。")

        # --- 组合B: 趋势 + 情绪 ---
        is_gold_cross = curr['dif'] > curr['dea']
        is_overbought = curr['rsi'] > 80
        is_oversold = curr['rsi'] < 20
        
        if is_gold_cross and not is_overbought:
            logic.append("✅ [共振B] MACD金叉且RSI未过热，趋势健康。")
        elif is_gold_cross and is_overbought:
            signals.append("⚠️ 趋势过热")
            logic.append("⚠️ [共振B] 虽然趋势向上，但RSI超买，短线可能回调，不宜追高。")
        elif not is_gold_cross and is_oversold:
            signals.append("💰 超跌反弹机会")
            logic.append("✅ [共振B] 虽然空头趋势，但RSI严重超跌，存在短线反抽概率。")

        # --- 组合C: 形态 + 资金 ---
        if curr['close'] < curr['dn'] and flow_val > 0:
            signals.append("💎 黄金坑 (资金底)")
            logic.append("✅ [共振C] 跌破布林下轨 + 主力资金逆势流入 = 假摔/抄底。")
        elif curr['close'] > curr['up'] and flow_val < -1:
            signals.append("☠️ 顶背离 (资金顶)")
            logic.append("❌ [共振C] 突破布林上轨 + 主力资金大幅流出 = 诱多/拉高出货。")

        return signals, logic

    def _analyze(self):
        df = self._calc_indicators(self.data['hist'].copy())
        curr = df.iloc[-1]
        close = curr['close']
        
        # 资金流
        flow_val = 0
        if not self.data['flow'].empty and '主力净流入净额' in self.data['flow'].columns:
            try: flow_val = round(self.data['flow']['主力净流入净额'].iloc[-3:].sum() / 1e8, 2)
            except: pass

        # 风控线
        stop_price = close - 2 * curr['atr']
        
        # --- ⚡ 运行共振分析 ---
        fusion_signals, fusion_logic = self._analyze_fusion(curr, flow_val, stop_price)
        
        # --- 舆情风控 ---
        veto = False
        if not self.data['news'].empty:
            txt = "".join(self.data['news'].head(10)['新闻标题'].tolist())
            if any(x in txt for x in ['立案', '调查', '退市']):
                veto = True
                fusion_logic.insert(0, "❌ [舆情] 触发黑名单关键词，一票否决。")

        # --- 最终决策 ---
        verdict = "观望"
        risk = "中"
        
        if veto: verdict = "避险卖出"; risk = "极高"
        elif close < stop_price:
            verdict = "清仓止损"; risk = "极高"
            fusion_logic.insert(0, f"❌ [风控] 跌破ATR硬止损位 {round(stop_price,2)}。")
        
        # 优先看共振信号
        elif "💎 黄金坑 (资金底)" in fusion_signals:
            verdict = "左侧低吸"; risk = "中"
        elif "☠️ 顶背离 (资金顶)" in fusion_signals or "⚠️ 高位滞涨出货" in fusion_signals:
            verdict = "清仓/离场"; risk = "高"
        elif "🔥 底部放量启动" in fusion_signals:
            verdict = "强力买入"; risk = "低"
        elif "🔒 缩量锁筹上涨" in fusion_signals:
            verdict = "坚定持有"; risk = "低"
        elif "⚠️ 趋势过热" in fusion_signals:
            verdict = "分批止盈"; risk = "中高"
        
        # 兜底逻辑
        elif curr['dif'] > curr['dea'] and close > curr['ma20']:
            verdict = "持有"; risk = "低"
        elif curr['dif'] < curr['dea']:
            verdict = "离场"; risk = "高"

        # 仓位建议
        base_pos = 0
        if verdict.startswith("强力") or verdict.startswith("坚定"): base_pos = 80
        elif verdict.startswith("持有") or verdict.startswith("低吸"): base_pos = 50
        elif verdict.startswith("分批"): base_pos = 30
        
        self.report['verdict'] = verdict
        self.report['risk_level'] = risk
        self.report['kelly_pos'] = base_pos
        self.report['fusion_signals'] = fusion_signals
        self.report['logic'] = fusion_logic

        # --- 记录指标 ---
        self._add_metric("量比 & 换手", f"{round(curr['vol_ratio'],2)} / {round(curr['turnover'],1)}%", "核心组合", "量比>1.5且换手3%-8%为最佳启动形态。", "-")
        self._add_metric("RSI & 趋势", f"{int(curr['rsi'])} / {'金叉' if curr['dif']>curr['dea'] else '死叉'}", "情绪组合", "趋势好但RSI>80需警惕。", "-")
        self._add_metric("资金 & 轨道", f"{flow_val}亿 / {'上轨' if close>curr['up'] else '通道内'}", "背离组合", "突破上轨但资金流出是诱多。", "-")
        
        self.levels.append(["🔴 动态止损", round(stop_price, 2), "硬风控"])
        self.levels.append(["🔴 布林上轨", round(curr['up'], 2), "压力"])
        self.levels.append(["🟢 布林下轨", round(curr['dn'], 2), "支撑"])

    def _add_metric(self, name, value, status, explanation, logic):
        self.metrics.append({"指标组合": name, "数值": value, "判定": status, "组合含义": explanation})

    def save_excel(self):
        if not self._fetch_data(): return
        self._analyze()
        
        filename = f"{self.symbol}_{self.data['spot']['名称']}_多因子共振版.xlsx"
        print(f"💾 生成共振报告: {filename} ...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            s_data = [
                ["代码", self.symbol], ["名称", self.data['spot']['名称']],
                ["最终建议", self.report['verdict']], ["风险等级", self.report['risk_level']],
                ["建议仓位", f"{self.report['kelly_pos']}%"], ["共振信号", " | ".join(self.report['fusion_signals']) if self.report['fusion_signals'] else "无明显共振"],
                ["", ""], ["决策逻辑", "\n".join(self.report['logic'])]
            ]
            pd.DataFrame(s_data, columns=["项目", "内容"]).to_excel(writer, sheet_name='决策看板', index=False)
            pd.DataFrame(self.metrics).to_excel(writer, sheet_name='多因子分析', index=False)
            pd.DataFrame(self.levels, columns=["类型", "价格", "说明"]).to_excel(writer, sheet_name='点位管理', index=False)
            
        print(f"✅ 完成！请下载。")

if __name__ == "__main__":
    print("Alpha Galaxy Fusion System (Multi-Factor)")
    code = input("Input Stock Code: ").strip()
    if code: AlphaGalaxyFusionSystem(code).save_excel()
