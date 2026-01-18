# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd
import numpy as np
from snownlp import SnowNLP
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

class AlphaGalaxyQuantSystem:
    def __init__(self, symbol):
        self.symbol = str(symbol)
        self.data = {}
        self.report = {
            "verdict": "观望",
            "risk_level": "中",
            "kelly_pos": 0,       # 建议仓位
            "win_rate": 0,        # 策略胜率
            "exp_return": 0,      # 期望收益
            "logic": []
        }
        self.metrics = []
        
        # 识别指数
        if self.symbol.startswith('6'):
            self.index_id = 'sh000001'; self.index_name = "上证指数"
        elif self.symbol.startswith('8') or self.symbol.startswith('4'):
            self.index_id = 'bj899050'; self.index_name = "北证50"
        else:
            self.index_id = 'sz399001'; self.index_name = "深证成指"

    def _fetch_data(self):
        print(f"🚀 [量化内核启动] 正在回测与分析 {self.symbol} ...")
        try:
            spot = ak.stock_zh_a_spot_em()
            target = spot[spot['代码'] == self.symbol]
            if target.empty: return False
            self.data['spot'] = target.iloc[0]
            self.data['all_spot'] = spot
            
            # 拉取更长的数据用于回测 (2年)
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
            
            # K线
            try:
                hist = ak.stock_zh_a_hist(symbol=self.symbol, period='daily', start_date=start, end_date=end, adjust='qfq')
                hist.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume', '换手率':'turnover'}, inplace=True)
                self.data['hist'] = hist
            except: return False

            # 指数 (用于滤网)
            try:
                idx = ak.stock_zh_index_daily(symbol=self.index_id)
                self.data['index'] = idx.tail(len(hist))
            except: self.data['index'] = pd.DataFrame()
            
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
            print(f"❌ 错误: {e}")
            return False

    # ================= ⚡ 核心：向量化回测引擎 =================
    def _run_backtest(self):
        """
        在历史数据上跑一遍策略，看看胜率如何。
        策略逻辑：均线多头(MA20>MA60) + 短期强势(收盘>MA20)
        """
        df = self.data['hist'].copy()
        
        # 1. 构造信号
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 信号：当 ma20 > ma60 且 close > ma20 时持有
        df['signal'] = np.where((df['ma20'] > df['ma60']) & (df['close'] > df['ma20']), 1, 0)
        
        # 2. 计算收益 (持有的下一天收益)
        df['pct_change'] = df['close'].pct_change().shift(-1) # 今天的信号决定明天的持仓
        df['strategy_ret'] = df['signal'] * df['pct_change']
        
        # 3. 统计指标 (最近1年)
        df_last_year = df.tail(250)
        
        # 胜率 (盈利天数 / 持仓天数)
        hold_days = df_last_year[df_last_year['signal'] == 1]
        if len(hold_days) > 0:
            win_days = hold_days[hold_days['strategy_ret'] > 0]
            win_rate = len(win_days) / len(hold_days)
            # 盈亏比
            avg_win = hold_days[hold_days['strategy_ret'] > 0]['strategy_ret'].mean()
            avg_loss = abs(hold_days[hold_days['strategy_ret'] < 0]['strategy_ret'].mean())
            wl_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        else:
            win_rate = 0
            wl_ratio = 0
            
        return win_rate, wl_ratio

    # ================= 🧮 凯利公式仓位管理 =================
    def _calc_kelly_position(self, win_rate, wl_ratio):
        """
        f* = (bp - q) / b
        b = 赔率(盈亏比), p = 胜率, q = 败率
        """
        if wl_ratio == 0 or win_rate == 0: return 0
        f = (win_rate * (wl_ratio + 1) - 1) / wl_ratio
        
        # 凯利公式太激进，通常用 "半凯利" (Half-Kelly)
        safe_f = f * 0.5 
        return max(0, min(safe_f, 1.0)) # 限制在 0% - 100%

    def _analyze(self):
        hist = self.data['hist']
        spot = self.data['spot']
        close = hist['close'].iloc[-1]
        
        # --- 1. 市场宏观滤网 (Market Regime) ---
        # 如果大盘指数跌破20日线，属于弱势，强制降仓
        market_ok = True
        idx_df = self.data['index']
        if not idx_df.empty:
            idx_close = idx_df['close'].iloc[-1]
            idx_ma20 = idx_df['close'].rolling(20).mean().iloc[-1]
            if idx_close < idx_ma20:
                market_ok = False
                self.report['logic'].append("🌍 宏观逆风：大盘指数处于空头趋势，建议降低预期。")
        
        # --- 2. 运行回测 (Backtest) ---
        win_rate, wl_ratio = self._run_backtest()
        # 凯利仓位
        kelly = self._calc_kelly_position(win_rate, wl_ratio)
        
        # 如果大盘不好，仓位打折
        final_pos = kelly if market_ok else kelly * 0.5
        
        self.report['win_rate'] = round(win_rate * 100, 1)
        self.report['kelly_pos'] = round(final_pos * 100, 1)
        
        # --- 3. 技术与资金分析 ---
        ma20 = hist['close'].rolling(20).mean().iloc[-1]
        ma60 = hist['close'].rolling(60).mean().iloc[-1]
        
        # 资金流
        flow_val = 0
        if not self.data['flow'].empty and '主力净流入净额' in self.data['flow'].columns:
            try: flow_val = round(self.data['flow']['主力净流入净额'].iloc[-3:].sum() / 1e8, 2)
            except: pass

        # 筹码
        df_chip = hist.tail(120).copy()
        df_chip['avg'] = (df_chip['open'] + df_chip['close'])/2
        winner_pct = (df_chip[df_chip['avg'] < close]['volume'].sum() / df_chip['volume'].sum() * 100)
        
        # 止损
        hist['tr'] = np.maximum(hist['high'] - hist['low'], abs(hist['high'] - hist['close'].shift(1)))
        atr = hist['tr'].rolling(14).mean().iloc[-1]
        stop = close - 2 * atr

        # --- 4. 最终裁决 (Verdict) ---
        # 逻辑：即使指标金叉，如果历史回测胜率<40%，也不买！
        
        reasons = []
        verdict = "观望"
        risk = "中"
        
        if close < stop:
            verdict = "清仓止损"; risk = "极高"
            reasons.append(f"❌ 触及ATR硬止损位 {round(stop, 2)}，风控离场。")
        elif win_rate < 0.45:
            verdict = "回避"; risk = "高"
            reasons.append(f"❌ 策略失效：该股历史趋势策略胜率仅 {self.report['win_rate']}%，股性不佳。")
        elif not market_ok and trend_status == "多头":
            verdict = "轻仓试错"; risk = "中高"
            reasons.append("⚠️ 逆势交易：个股虽强但大盘弱，仅建议轻仓。")
        elif close > ma20 and flow_val > 0 and winner_pct < 90:
            if win_rate > 0.55:
                verdict = "买入/加仓"; risk = "低"
                reasons.append(f"✅ 量化确认：策略历史胜率{self.report['win_rate']}%(高) + 资金流入。")
            else:
                verdict = "持有"; risk = "中"
                reasons.append("✅ 趋势良好，但历史胜率一般，建议持有不追高。")
        
        self.report['verdict'] = verdict
        self.report['risk_level'] = risk
        self.report['logic'].extend(reasons)
        
        # 记录指标用于Excel
        self._add_metric("历史回测胜率", f"{self.report['win_rate']}%", "优秀" if win_rate>0.6 else "一般", "过去1年用趋势策略做这只股的胜率。", "专业交易员只做高胜率的票")
        self._add_metric("凯利建议仓位", f"{self.report['kelly_pos']}%", "-", "基于胜率和赔率计算的科学仓位。", f"结合大盘环境，建议最大仓位 {self.report['kelly_pos']}%")
        self._add_metric("大盘环境", "多头" if market_ok else "空头", "安全" if market_ok else "危险", "大盘是否配合。", "覆巢之下无完卵")

        # 基础指标
        self._add_metric("主力资金", f"{flow_val}亿", "流入" if flow_val>0 else "流出", "主力动向", "近3日净额")
        
        # 生成点位
        self.levels_list.append(["🔴 动态止损", round(stop, 2), "硬风控"])
        if close < ma60: self.levels_list.append(["🔴 机构成本线", round(ma60, 2), "压力"])
        else: self.levels_list.append(["🟢 机构成本线", round(ma60, 2), "支撑"])

    def _add_metric(self, name, value, status, explanation, logic):
        self.metrics.append({"指标": name, "数值": value, "判定": status, "含义": explanation, "逻辑": logic})

    def save_excel(self):
        if not self._fetch_data(): return
        self._analyze()
        
        filename = f"{self.symbol}_{self.data['spot']['名称']}_量化验证版.xlsx"
        print(f"💾 生成专业报告: {filename} ...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: 决策看板
            s_data = [
                ["代码", self.symbol], ["名称", self.data['spot']['名称']],
                ["建议", self.report['verdict']], ["风险", self.report['risk_level']],
                ["回测胜率", f"{self.report['win_rate']}%"], ["建议仓位", f"{self.report['kelly_pos']}%"],
                ["", ""], ["核心逻辑", "\n".join(self.report['logic'])]
            ]
            pd.DataFrame(s_data, columns=["项目", "内容"]).to_excel(writer, sheet_name='决策看板', index=False)
            
            # Sheet 2: 量化数据
            pd.DataFrame(self.metrics).to_excel(writer, sheet_name='量化指标', index=False)
            
            # Sheet 3: 点位
            pd.DataFrame(self.levels_list, columns=["类型", "价格", "说明"]).to_excel(writer, sheet_name='点位管理', index=False)
            
        print(f"✅ 完成！请下载。")

if __name__ == "__main__":
    print("Alpha Galaxy Quant Verification (Pro)")
    code = input("Input Stock Code: ").strip()
    if code: AlphaGalaxyQuantSystem(code).save_excel()
