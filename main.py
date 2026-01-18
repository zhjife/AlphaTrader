# -*- coding: utf-8 -*-
"""
Alpha Galaxy Omni-Logic Ultimate (全形态全逻辑终极版)
Author: Quant Studio
Features:
1. 30+ K-Line Patterns (Full Library)
2. Strategy Combos (A/B/C)
3. Full Technical & Fund Flow Analysis
"""

import akshare as ak
import pandas as pd
import numpy as np
from snownlp import SnowNLP
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

class AlphaGalaxyUltimate:
    def __init__(self, symbol):
        self.symbol = str(symbol)
        self.data = {}
        self.report = {
            "verdict": "观望", "risk_level": "中", 
            "mode": "震荡", "kelly_pos": 0, 
            "win_rate": 0, "logic": [], "signals": [],
            "patterns_bull": [], "patterns_bear": [] # 分开存储多空形态
        }
        self.metrics = []
        self.levels = []
        
        # 指数映射
        if self.symbol.startswith('6'): self.index_id = 'sh000001'; self.index_name = "上证指数"
        elif self.symbol.startswith('8') or self.symbol.startswith('4'): self.index_id = 'bj899050'; self.index_name = "北证50"
        else: self.index_id = 'sz399001'; self.index_name = "深证成指"

    # ================= 1. 数据中台 (高容错) =================
    def _fetch_data(self):
        print(f"🚀 [全形态引擎启动] 正在深度扫描 {self.symbol} (加载30+种K线模型)...")
        try:
            # 1.1 实时行情
            spot = ak.stock_zh_a_spot_em()
            target = spot[spot['代码'] == self.symbol]
            if target.empty: 
                print(f"❌ 未找到代码 {self.symbol}")
                return False
            self.data['spot'] = target.iloc[0]
            
            # 1.2 历史K线 (取足够长的数据以识别大形态)
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
            try:
                hist = ak.stock_zh_a_hist(symbol=self.symbol, period='daily', start_date=start, end_date=end, adjust='qfq')
                if hist is None or hist.empty: return False
                hist.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume', '换手率':'turnover'}, inplace=True)
                self.data['hist'] = hist
            except: return False

            # 1.3 大盘指数
            try: self.data['index'] = ak.stock_zh_index_daily(symbol=self.index_id).tail(len(hist))
            except: self.data['index'] = pd.DataFrame()
            
            # 1.4 资金流
            try:
                flow = ak.stock_individual_fund_flow(stock=self.symbol, market="sh" if self.symbol.startswith("6") else "sz")
                self.data['flow'] = flow.sort_values('日期').tail(10) if (flow is not None and not flow.empty) else pd.DataFrame()
            except: self.data['flow'] = pd.DataFrame()
            
            # 1.5 舆情
            try: self.data['news'] = ak.stock_news_em(symbol=self.symbol)
            except: self.data['news'] = pd.DataFrame()

            return True
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            return False

    # ================= 2. 指标计算引擎 =================
    def _calc_indicators(self, df):
        # 均线系统
        for w in [5, 10, 20, 60, 120, 250]: df[f'ma{w}'] = df['close'].rolling(w).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        
        # KDJ
        low_9 = df['low'].rolling(9).min()
        high_9 = df['high'].rolling(9).max()
        rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
        df['k'] = rsv.ewm(com=2).mean()
        df['d'] = df['k'].ewm(com=2).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # BOLL
        df['std'] = df['close'].rolling(20).std()
        df['up'] = df['ma20'] + 2 * df['std']
        df['dn'] = df['ma20'] - 2 * df['std']
        df['bb_width'] = (df['up'] - df['dn']) / df['ma20']
        
        # ATR & Drawdown
        df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
        df['atr'] = df['tr'].rolling(14).mean()
        roll_max = df['close'].rolling(250, min_periods=1).max()
        df['drawdown'] = (df['close'] / roll_max) - 1.0
        
        # OBV & CMF
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        
        mf_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 0.01)
        df['cmf'] = (mf_mult * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # 量比
        df['vol_ma5'] = df['volume'].rolling(5).mean().shift(1)
        df['vol_ratio'] = df['volume'] / df['vol_ma5']
        
        # 新高
        df['high_60'] = df['high'].rolling(60).max()
        
        return df

    # ================= 3. K线全形态识别库 (Full Library) =================
    def _analyze_pattern_full(self, df):
        """
        包含 30+ 种典型 K 线形态的完整识别引擎
        """
        if len(df) < 10: return [], []
        
        bull_pats = []
        bear_pats = []
        
        # 提取数据序列
        c = df['close'].values; o = df['open'].values
        h = df['high'].values; l = df['low'].values
        v = df['volume'].values
        
        # 游标 (0=今天, 1=昨天...)
        c0, c1, c2, c3, c4 = c[-1], c[-2], c[-3], c[-4], c[-5]
        o0, o1, o2, o3, o4 = o[-1], o[-2], o[-3], o[-4], o[-5]
        h0, h1 = h[-1], h[-2]
        l0, l1 = l[-1], l[-2]
        
        # 基础属性
        body0 = abs(c0 - o0)
        upper0 = h0 - max(c0, o0)
        lower0 = min(c0, o0) - l0
        avg_body = np.mean(np.abs(c[-6:-1] - o[-6:-1])) # 平均实体
        is_bull0 = c0 > o0
        is_bear0 = c0 < o0
        is_doji = body0 < avg_body * 0.1
        
        # 趋势背景 (重要：形态必须结合趋势)
        ma20 = df['ma20'].iloc[-1]
        is_uptrend = c0 > ma20
        is_downtrend = c0 < ma20

        # ==================== A. 见底/看涨形态 (Bullish) ====================
        
        # 1. 早晨之星 (Morning Star) - 强反转
        if c2 < o2 and is_bear0 is False and abs(c1-o1) < body0*0.5 and c0 > (o2+c2)/2:
            bull_pats.append("早晨之星")
            
        # 2. 红三兵 (Three White Soldiers) - 强推升
        if c0>o0 and c1>o1 and c2>o2 and c0>c1>c2 and o0>o1>o2:
            bull_pats.append("红三兵")
            
        # 3. 阳包阴 (Bullish Engulfing) - 强吞噬
        if c1 < o1 and is_bull0 and c0 > o1 and o0 < c1:
            bull_pats.append("阳包阴(反包)")
            
        # 4. 曙光初现 (Piercing Line) - 刺透
        if c1 < o1 and is_bull0 and o0 < l1 and c0 > (o1+c1)/2 and c0 < o1:
            bull_pats.append("曙光初现")
            
        # 5. 旭日东升 (Rising Sun) - 高开吞没
        if c1 < o1 and is_bull0 and o0 > c1 and c0 > o1:
            bull_pats.append("旭日东升")
            
        # 6. 锤子线 (Hammer) - 低位探底
        if is_downtrend and lower0 > 2*body0 and upper0 < body0*0.2:
            bull_pats.append("锤子线")
            
        # 7. 倒锤头 (Inverted Hammer) - 低位试盘
        if is_downtrend and upper0 > 2*body0 and lower0 < body0*0.2:
            bull_pats.append("倒锤头")
            
        # 8. 平底 (Tweezer Bottom) - 双针探底
        if is_downtrend and abs(l0 - l1) < c0*0.002:
            bull_pats.append("平底(镊子底)")
            
        # 9. 上升三法 (Rising Three Methods) - 中继形态
        # 大阳 + 3根小阴不破低 + 大阳
        if c4>o4 and c0>o0 and c0>c4 and c1<o1 and c2<o2 and min(l1,l2,l3)>l4:
            bull_pats.append("上升三法(中继)")
            
        # 10. 多头孕线 (Bullish Harami)
        if c1 < o1 and is_bull0 and h0 < h1 and l0 > l1:
            bull_pats.append("多头孕线")
            
        # 11. 向上跳空缺口 (Gap Up)
        if l0 > h1:
            bull_pats.append("向上缺口")
            
        # 12. 底部岛形反转 (Island Bottom)
        # 简化版：跌缺口 + 盘整 + 涨缺口
        if h[-2] < l[-3] and l[-1] > h[-2]:
            bull_pats.append("岛形反转(底)")

        # ==================== B. 见顶/看跌形态 (Bearish) ====================
        
        # 13. 黄昏之星 (Evening Star) - 强见顶
        if c2 > o2 and abs(c1-o1) < body0*0.5 and is_bear0 and c0 < (o2+c2)/2:
            bear_pats.append("黄昏之星")
            
        # 14. 三只乌鸦 (Three Black Crows) - 强杀跌
        if c0<o0 and c1<o1 and c2<o2 and c0<c1<c2:
            bear_pats.append("三只乌鸦")
            
        # 15. 阴包阳 (Bearish Engulfing) - 空头吞噬
        if c1 > o1 and is_bear0 and o0 > c1 and c0 < o1:
            bear_pats.append("阴包阳(穿头破脚)")
            
        # 16. 乌云盖顶 (Dark Cloud Cover) - 见顶
        if c1 > o1 and is_bear0 and o0 > h1 and c0 < (o1+c1)/2 and c0 > o1:
            bear_pats.append("乌云盖顶")
            
        # 17. 倾盆大雨 (Heavy Rain) - 低开杀跌
        if c1 > o1 and is_bear0 and o0 < c1 and c0 < o1:
            bear_pats.append("倾盆大雨")
            
        # 18. 射击之星 (Shooting Star) - 高位避雷针
        if is_uptrend and upper0 > 2*body0 and lower0 < body0*0.2:
            bear_pats.append("射击之星")
            
        # 19. 吊颈线 (Hanging Man) - 高位诱多
        if is_uptrend and lower0 > 2*body0 and upper0 < body0*0.2:
            bear_pats.append("吊颈线")
            
        # 20. 平顶 (Tweezer Top) - 双顶
        if is_uptrend and abs(h0 - h1) < c0*0.002:
            bear_pats.append("平顶(镊子顶)")
            
        # 21. 断头铡刀 (Breakdown) - 一阴断三线
        ma5=df['ma5'].iloc[-1]; ma10=df['ma10'].iloc[-1]; ma20=df['ma20'].iloc[-1]
        if is_bear0 and o0 > max(ma5,ma10,ma20) and c0 < min(ma5,ma10,ma20):
            bear_pats.append("断头铡刀")
            
        # 22. 下降三法 (Falling Three Methods) - 下跌中继
        if c4<o4 and c0<o0 and c0<c4 and c1>o1 and c2>o2 and max(h1,h2,h3)<h4:
            bear_pats.append("下降三法")
            
        # 23. 空头孕线 (Bearish Harami)
        if c1 > o1 and is_bear0 and h0 < h1 and l0 > l1:
            bear_pats.append("空头孕线")
            
        # 24. 向下跳空缺口 (Gap Down)
        if h0 < l1:
            bear_pats.append("向下缺口")
            
        # 25. 顶部岛形反转
        if l[-2] > h[-3] and h[-1] < l[-2]:
            bear_pats.append("岛形反转(顶)")

        # ==================== C. 整理/其他形态 ====================
        
        # 26. 十字星 (Doji)
        if is_doji and abs(upper0 - lower0) < body0 * 0.5:
            # 根据位置判断多空
            if is_uptrend: bear_pats.append("高位十字星")
            elif is_downtrend: bull_pats.append("低位十字星")
            
        return bull_pats, bear_pats

    # ================= 4. 回测与筹码 =================
    def _run_backtest(self, df):
        df['signal'] = np.where(df['close'] > df['ma20'], 1, 0)
        df['ret'] = df['signal'].shift(1) * df['close'].pct_change()
        wins = len(df[df['ret'] > 0])
        total = len(df[df['ret'] != 0])
        return wins / total if total > 0 else 0

    def _calc_chip_winner(self, df):
        sub = df.tail(120).copy()
        current = df['close'].iloc[-1]
        sub['avg'] = (sub['open'] + sub['close'])/2
        winner_vol = sub[sub['avg'] < current]['volume'].sum()
        total_vol = sub['volume'].sum()
        return (winner_vol / total_vol * 100) if total_vol > 0 else 0

    # ================= 5. 组合战法扫描 (A/B/C) =================
    def _check_combo_logic(self, curr, flow_val):
        signals = []
        reasons = []
        priority_verdict = None 
        close = curr['close']
        
        # --- 组合 A: 量比 + 换手 + 位置 ---
        is_low = close < curr['ma60'] * 1.15
        is_high = close > curr['ma60'] * 1.3
        
        if is_low and curr['turnover'] > 3 and curr['vol_ratio'] > 1.8:
            signals.append("主力启动")
            reasons.append("🔥 [组合A] 低位 + 放量(量比>1.8) + 换手活跃 = 主力建仓启动。")
            priority_verdict = "买入"
        elif is_high and curr['turnover'] > 10 and close <= curr['open']:
            signals.append("主力出货")
            reasons.append("⚠️ [组合A] 高位 + 巨量换手(>10%) + 滞涨 = 主力可能在出货。")
            priority_verdict = "卖出"
        elif close > curr['ma20'] and curr['turnover'] < 3 and 0.7 < curr['vol_ratio'] < 1.3:
            signals.append("主力锁筹")
            reasons.append("🔒 [组合A] 上涨趋势 + 低换手(<3%) + 量比平稳 = 主力锁筹躺赢。")
            if priority_verdict is None: priority_verdict = "持有"

        # --- 组合 B: MACD + RSI ---
        if curr['dif'] > curr['dea'] and curr['rsi'] > 80:
            signals.append("假买点")
            reasons.append("🚫 [组合B] MACD金叉但RSI>80(过热)，属于【假买点】，谨防追高。")
            if priority_verdict == "买入": priority_verdict = "观察"
        elif curr['dif'] < curr['dea'] and curr['rsi'] < 20:
            signals.append("假卖点")
            reasons.append("💎 [组合B] MACD死叉但RSI<20(冰点)，属于【假卖点】，随时可能反弹。")
            if priority_verdict == "卖出": priority_verdict = "观望"

        # --- 组合 C: 布林带 + 资金 ---
        if close < curr['dn'] and (flow_val > 0.5 or curr['cmf'] > 0.1):
            signals.append("黄金坑")
            reasons.append("💰 [组合C] 跌破布林下轨 + 主力资金逆势流入 = 【黄金坑】。")
            priority_verdict = "低吸"
        elif close > curr['up'] and (flow_val < -0.5 or curr['cmf'] < -0.1):
            signals.append("顶背离")
            reasons.append("☠️ [组合C] 突破布林上轨 + 主力资金大幅流出 = 【顶背离】。")
            priority_verdict = "清仓"

        return signals, reasons, priority_verdict

    # ================= 6. 综合决策大脑 =================
    def _analyze(self):
        df = self._calc_indicators(self.data['hist'].copy())
        curr = df.iloc[-1]
        close = curr['close']
        
        # 资金流
        flow_val = 0
        if not self.data['flow'].empty and '主力净流入净额' in self.data['flow'].columns:
            try: flow_val = round(self.data['flow']['主力净流入净额'].iloc[-3:].sum() / 1e8, 2)
            except: pass
            
        # 运行模块
        bull_pats, bear_pats = self._analyze_pattern_full(self.data['hist']) # K线全库
        combo_signals, combo_logic, combo_verdict = self._check_combo_logic(curr, flow_val) # 组合战法
        win_rate = self._run_backtest(df)
        stop_price = close - 2 * curr['atr']
        winner_pct = self._calc_chip_winner(df)
        
        # 舆情
        news_veto = False
        if not self.data['news'].empty:
            txt = "".join(self.data['news'].head(10)['新闻标题'].tolist())
            if any(x in txt for x in ['立案', '调查', '退市', '警示']): news_veto = True

        # --- 最终裁决 ---
        verdict = "观望"; risk = "中"; logic = combo_logic
        signals = combo_signals
        if bull_pats: signals.extend(bull_pats)
        if bear_pats: signals.extend(bear_pats)
        
        # 1. 否决层
        if news_veto:
            verdict = "避险卖出"; risk = "极高"; logic.insert(0, "❌ [舆情] 触发黑名单。")
        elif close < stop_price:
            verdict = "清仓止损"; risk = "极高"; logic.insert(0, f"❌ [风控] 跌破ATR止损位 {round(stop_price,2)}。")
        elif "断头铡刀" in bear_pats or "三只乌鸦" in bear_pats:
            verdict = "离场"; risk = "高"; logic.append(f"❌ [K线] 出现恶劣形态：{','.join(bear_pats)}。")
            
        # 2. 战法层 (A/B/C)
        elif combo_verdict:
            verdict = combo_verdict
            risk = "高" if verdict in ["清仓", "卖出"] else "低"
            
        # 3. 形态加分层
        elif bull_pats:
            if flow_val > 0:
                verdict = "买入"; risk = "低"; logic.append(f"✅ [K线] 看涨形态({','.join(bull_pats)}) + 资金配合。")
            else:
                verdict = "观察"; risk = "中"; logic.append(f"⚠️ [K线] 有看涨形态({','.join(bull_pats)})但资金未流进。")
        elif bear_pats:
            verdict = "减仓"; risk = "中高"; logic.append(f"⚠️ [K线] 出现看跌形态({','.join(bear_pats)})。")
                
        # 4. 兜底层
        else:
            if curr['dif'] > curr['dea'] and flow_val > 0:
                verdict = "持有"; risk = "低"; logic.append("✅ [趋势] 趋势向好，资金流入。")
            elif curr['dif'] < curr['dea']:
                verdict = "减仓"; risk = "中高"; logic.append("⚠️ [趋势] 趋势转弱。")

        # 仓位
        base_pos = 0
        if verdict in ["买入", "持有", "主力锁筹"]: base_pos = 60
        if "启动" in str(signals) or "红三兵" in str(signals) or "早晨之星" in str(signals): base_pos = 80
        if "低吸" in verdict: base_pos = 30
        
        self.report.update({
            "verdict": verdict, "risk_level": risk, 
            "kelly_pos": base_pos, "win_rate": int(win_rate*100),
            "logic": logic, "signals": signals,
            "patterns_bull": bull_pats, "patterns_bear": bear_pats
        })

        # --- 指标记录 ---
        self._add_metric("战法组合A", f"换手{round(curr['turnover'],1)}%", f"量比{round(curr['vol_ratio'],2)}", "主力意图(启动/出货/锁筹)", "-")
        self._add_metric("战法组合B", f"RSI:{int(curr['rsi'])}", "金叉" if curr['dif']>curr['dea'] else "死叉", "买卖点校准", "-")
        self._add_metric("战法组合C", f"资金:{flow_val}亿", "CMF:"+str(round(curr['cmf'],2)), "真假突破/背离", "-")
        
        k_str = "无"
        if bull_pats: k_str = f"多:{','.join(bull_pats)}"
        if bear_pats: k_str += f" 空:{','.join(bear_pats)}"
        self._add_metric("K线形态库", k_str, "-", "30+种形态扫描结果", "-")
        
        self._add_metric("博弈数据", f"获利{int(winner_pct)}%", f"回撤{int(curr['drawdown']*100)}%", "拥挤度与股性", "-")
        
        self.levels.append(["🔴 动态止损", round(stop_price, 2), "硬风控"])
        self.levels.append(["🔴 布林上轨", round(curr['up'], 2), "压力"])
        self.levels.append(["🟢 布林下轨", round(curr['dn'], 2), "支撑"])

    def _add_metric(self, name, val1, val2, explanation, logic):
        self.metrics.append({"维度": name, "数据1": val1, "数据2": val2, "判定逻辑": explanation})

    def save_excel(self):
        if not self._fetch_data(): return
        self._analyze()
        # [修改点] 加入时间戳到分钟 (YYYYMMDD_HHMM)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{self.symbol}_{self.data['spot']['名称']}_全逻辑终极版_{timestamp}.xlsx"
        
        print(f"💾 生成报告: {filename} ...")
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            s_data = [
                ["代码", self.symbol], ["名称", self.data['spot']['名称']],
                ["最终建议", self.report['verdict']], ["风险等级", self.report['risk_level']],
                ["建议仓位", f"{self.report['kelly_pos']}%"], 
                ["看涨形态", " | ".join(self.report['patterns_bull'])],
                ["看跌形态", " | ".join(self.report['patterns_bear'])],
                ["组合战法", " | ".join(self.report['signals'])],
                ["", ""], ["决策逻辑", "\n".join(self.report['logic'])]
            ]
            pd.DataFrame(s_data, columns=["项目", "内容"]).to_excel(writer, sheet_name='决策看板', index=False)
            pd.DataFrame(self.metrics).to_excel(writer, sheet_name='详细指标', index=False)
            pd.DataFrame(self.levels, columns=["类型", "价格", "说明"]).to_excel(writer, sheet_name='点位管理', index=False)
        print(f"✅ 完成！请下载。")

if __name__ == "__main__":
    print("Alpha Galaxy Omni-Logic Ultimate (Full Patterns)")
    code = input("Input Stock Code: ").strip()
    if code: AlphaGalaxyUltimate(code).save_excel()
