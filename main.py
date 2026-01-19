# -*- coding: utf-8 -*-
"""
Alpha Galaxy Omni-Logic Ultimate (点位增强版)
优化内容:
1. [Levels] 点位管理大幅增强：新增MA20/MA60、箱体高低点、筹码均价
2. [Logic] 保持原有所有判断逻辑、形态、指标不变
3. [Full] 包含完整字典、舆情、无限循环
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
            "score": 0, "mode": "震荡", "kelly_pos": 0, 
            "logic": [], "signals": [],
            "patterns_bull": [], "patterns_bear": []
        }
        self.metrics = []
        self.levels = []
        self.history_metrics = {}
        
        # 指数映射
        if self.symbol.startswith('6'): self.index_name = "上证指数"
        elif self.symbol.startswith('8') or self.symbol.startswith('4'): self.index_name = "北证50"
        else: self.index_name = "深证成指"

    # ================= 1. 数据中台 =================
    def _fetch_data(self):
        print(f"🚀 [全维扫描] 正在读取 {self.symbol} ...")
        
        # 1.1 实时行情
        try:
            spot = ak.stock_zh_a_spot_em()
            target = spot[spot['代码'] == self.symbol]
            
            if not target.empty:
                target = target.copy()
                for col in ['市盈率-动态', '市净率', '总市值', '换手率', '最新价']:
                    if col in target.columns:
                        target[col] = pd.to_numeric(target[col], errors='coerce')
                self.data['spot'] = target.iloc[0]
            else:
                print(f"⚠️ 实时列表未匹配，启用强行读取模式...")
                self.data['spot'] = {
                    '名称': self.symbol, '最新价': 0, '市盈率-动态': -1, '市净率': -1, '换手率': 0
                }
        except Exception as e:
            print(f"⚠️ 实时接口异常，跳过基本面...")
            self.data['spot'] = {'名称': self.symbol, '市盈率-动态': -1, '市净率': -1}

        # 1.2 历史K线
        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
            hist = ak.stock_zh_a_hist(symbol=self.symbol, period='daily', start_date=start, end_date=end, adjust='qfq')
            
            if hist is None or hist.empty:
                print(f"❌ 无法读取K线数据，请确认代码 {self.symbol} 是否正确。")
                return False
                
            hist.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume', '换手率':'turnover'}, inplace=True)
            self.data['hist'] = hist
            
            # 补全缺失数据
            if self.data['spot'].get('最新价', 0) == 0: self.data['spot']['最新价'] = hist.iloc[-1]['close']
            if self.data['spot'].get('换手率', 0) == 0 and 'turnover' in hist.columns: self.data['spot']['换手率'] = hist.iloc[-1]['turnover']
                 
        except Exception as e:
            print(f"❌ K线数据获取失败: {e}")
            return False

        # 1.3 资金流 & 舆情
        try:
            flow = ak.stock_individual_fund_flow(stock=self.symbol, market="sh" if self.symbol.startswith("6") else "sz")
            self.data['flow'] = flow.sort_values('日期').tail(10) if (flow is not None and not flow.empty) else pd.DataFrame()
        except: self.data['flow'] = pd.DataFrame()
        
        try: self.data['news'] = ak.stock_news_em(symbol=self.symbol)
        except: self.data['news'] = pd.DataFrame()

        return True

    # ================= 2. 舆情分析引擎 =================
    def _analyze_sentiment(self):
        try:
            if self.data['news'].empty: return 0, "无近期舆情"
            news_df = self.data['news'].head(10)
            titles = news_df['新闻标题'].tolist()
            full_text = "。".join(titles)
            
            pos_kw = ['增长', '预增', '突破', '利好', '回购', '获批', '中标', '大涨', '新高']
            neg_kw = ['立案', '调查', '亏损', '减持', '警示', '违规', '大跌', '退市', '被查']
            
            hard_score = 0
            keywords = []
            for t in titles:
                for kw in pos_kw:
                    if kw in t: hard_score += 2; keywords.append(kw)
                for kw in neg_kw:
                    if kw in t: hard_score -= 10; keywords.append(kw)
            
            s = SnowNLP(full_text)
            soft_score = (s.sentiments - 0.5) * 10
            total = max(min(hard_score + soft_score, 20), -20)
            return round(total, 1), f"关键词:{list(set(keywords))}" if keywords else "舆情平稳"
        except: return 0, "舆情分析略过"

    # ================= 3. 指标计算 =================
    def _calc_indicators(self, df):
        # MA
        for w in [5, 10, 20, 60, 120, 250]: df[f'ma{w}'] = df['close'].rolling(w).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        
        # KDJ
        low_9 = df['low'].rolling(9).min(); high_9 = df['high'].rolling(9).max()
        rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
        df['k'] = rsv.ewm(com=2, adjust=False).mean()
        df['d'] = df['k'].ewm(com=2, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # RSI (Wilder)
        delta = df['close'].diff()
        up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
        for period in [6, 12, 24]:
            ema_up = up.ewm(alpha=1/period, adjust=False).mean()
            ema_down = down.ewm(alpha=1/period, adjust=False).mean()
            rs = ema_up / ema_down
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi_6'] 
        
        # BOLL
        df['std'] = df['close'].rolling(20).std()
        df['up'] = df['ma20'] + 2 * df['std']
        df['dn'] = df['ma20'] - 2 * df['std']
        df['bb_width'] = (df['up'] - df['dn']) / df['ma20'] 
        
        # ATR & Drawdown
        df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
        df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        roll_max = df['close'].rolling(250, min_periods=1).max()
        df['drawdown'] = (df['close'] / roll_max) - 1.0

        # ADX & CCI & BIAS
        up_move = df['high'] - df['high'].shift(1); down_move = df['low'].shift(1) - df['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_smooth = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()

        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True))
        df['bias'] = (df['close'] - df['ma20']) / df['ma20'] * 100
        
        # CMF & Vol Ratio & PCT
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        mf_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 0.01)
        df['cmf'] = (mf_mult * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vol_ma5'] = df['volume'].rolling(5).mean().shift(1)
        df['vol_ratio'] = df['volume'] / df['vol_ma5']
        df['pct_change'] = df['close'].pct_change() * 100
        
        return df

    # ================= 4. 筹码获利盘 =================
    def _calc_chip_winner(self, df):
        if len(df) < 120: return 50
        sub = df.tail(60).copy()
        current_price = df['close'].iloc[-1]
        sub['avg_price'] = (sub['open'] + sub['close'] + sub['high'] + sub['low']) / 4
        winner_vol = sub[sub['avg_price'] < current_price]['volume'].sum()
        total_vol = sub['volume'].sum()
        if total_vol == 0: return 0
        return (winner_vol / total_vol) * 100

    # ================= 5. K线形态识别 =================
    def _analyze_pattern_full(self, df):
        if len(df) < 20: return [], [], 0
        bull_pats, bear_pats = [], []
        score = 0 
        
        c = df['close'].values; o = df['open'].values
        h = df['high'].values; l = df['low'].values
        v = df['volume'].values
        ma5 = df['ma5'].values; ma10 = df['ma10'].values; ma20 = df['ma20'].values
        
        c0, c1, c2, c3, c4 = c[-1], c[-2], c[-3], c[-4], c[-5]
        o0, o1, o2, o3, o4 = o[-1], o[-2], o[-3], o[-4], o[-5]
        h0, h1 = h[-1], h[-2]; l0, l1 = l[-1], l[-2]
        v0, v1 = v[-1], v[-2]
        
        body0 = abs(c0 - o0)
        upper0 = h0 - max(c0, o0); lower0 = min(c0, o0) - l0
        is_bull0 = c0 > o0; is_bear0 = c0 < o0
        is_downtrend = c0 < ma20[-1]; is_uptrend = c0 > ma20[-1]

        # [买入]
        if c2 < o2 and is_bear0 is False and abs(c1-o1) < body0*0.5 and c0 > (o2+c2)/2: bull_pats.append("早晨之星"); score += 20
        if is_downtrend and lower0 > 2*body0 and upper0 < body0*0.2: bull_pats.append("锤子线"); score += 15
        if is_downtrend and upper0 > 2*body0 and lower0 < body0*0.2: bull_pats.append("倒锤头"); score += 10
        if c1 < o1 and is_bull0 and c0 > o1 and o0 < c1: bull_pats.append("阳包阴"); score += 20
        if c1 < o1 and is_bull0 and o0 < l1 and c0 > (o1+c1)/2 and c0 < o1: bull_pats.append("曙光初现"); score += 15
        if is_downtrend and abs(l0 - l1) < c0*0.002: bull_pats.append("平底"); score += 10
        if c1 < o1 and is_bull0 and h0 < h1 and l0 > l1: bull_pats.append("多头孕线"); score += 10
        if c0>o0 and c1>o1 and c2>o2 and c0>c1>c2: bull_pats.append("红三兵"); score += 15
        if c4>o4 and c0>o0 and c0>c4 and c1<o1 and c2<o2: bull_pats.append("上升三法"); score += 20
        if c2>o2 and c1<o1 and c0>o0 and c0>c2 and o1<c2: bull_pats.append("多方炮"); score += 20
        if l0 > h1: bull_pats.append("向上缺口"); score += 15
        if is_bull0 and c0 > max(ma5[-1], ma10[-1], ma20[-1]) and o0 < min(ma5[-1], ma10[-1], ma20[-1]): bull_pats.append("一阳穿三线"); score += 25
        if v0 > v1*1.9 and c0 > np.max(c[-20:-1]): bull_pats.append("倍量过左峰"); score += 20
        diff = max(ma5[-1],ma10[-1],ma20[-1]) - min(ma5[-1],ma10[-1],ma20[-1])
        if (diff/c0 < 0.015) and is_bull0: bull_pats.append("金蜘蛛"); score += 15
        if (h1-max(c1,o1)) > abs(c1-o1) and c0>h1 and is_bull0: bull_pats.append("仙人指路"); score += 15
        if c1 < o1 and is_bull0 and o0 > c1 and c0 > o1: bull_pats.append("旭日东升"); score += 20
        if h[-2] < l[-3] and l[-1] > h[-2]: bull_pats.append("岛形反转(底)"); score += 30
        if c1 < o1 and is_bull0 and o0 > h1: bull_pats.append("踢脚线"); score += 30
        if l0 <= ma20[-1] and c0 > ma20[-1] and c1 > ma20[-1] and is_bull0: bull_pats.append("蜻蜓点水"); score += 10

        # [卖出]
        if c2 > o2 and abs(c1-o1) < body0*0.5 and is_bear0 and c0 < (o2+c2)/2: bear_pats.append("黄昏之星"); score -= 20
        if c1 > o1 and is_bear0 and o0 > h1 and c0 < (o1+c1)/2: bear_pats.append("乌云盖顶"); score -= 20
        if is_bear0 and o0 > max(ma5[-1],ma10[-1],ma20[-1]) and c0 < min(ma5[-1],ma10[-1],ma20[-1]): bear_pats.append("断头铡刀"); score -= 30
        if c0<o0 and c1<o1 and c2<o2 and c0<c1<c2: bear_pats.append("三只乌鸦"); score -= 25
        if h0 < l1: bear_pats.append("向下缺口"); score -= 15
        if c1 > o1 and is_bear0 and o0 > c1 and c0 < o1: bear_pats.append("阴包阳"); score -= 20
        if not is_downtrend and upper0 > 2*body0 and lower0 < body0*0.2: bear_pats.append("射击之星"); score -= 15
        if is_uptrend and lower0 > 2*body0 and upper0 < body0*0.2: bear_pats.append("吊颈线"); score -= 15
        if is_uptrend and abs(h0 - h1) < c0*0.002: bear_pats.append("平顶"); score -= 10
        if c1 > o1 and is_bear0 and o0 < c1 and c0 < o1: bear_pats.append("倾盆大雨"); score -= 20
        if c1 > o1 and is_bear0 and h0 < h1 and l0 > l1: bear_pats.append("空头孕线"); score -= 10
        if l[-2] > h[-3] and h[-1] < l[-2]: bear_pats.append("岛形反转(顶)"); score -= 30
        if upper0 > 2*body0 and abs(o0-c0) < 0.01*c0 and lower0 < 0.1*body0: bear_pats.append("墓碑线"); score -= 20
        
        return bull_pats, bear_pats, score

    # ================= 6. 核心逻辑 =================
    def _check_combo_logic(self, curr, flow_val, sentiment_score, k_score, winner_pct):
        signals = []
        reasons = []
        score = 0
        priority_verdict = None 
        close = curr['close']
        
        # 1. 基础技术
        if close > curr['ma20']: score += 20
        if curr['adx'] > 25: score += 10
        if curr['cci'] > 100: score += 10
        score += k_score + sentiment_score

        # 2. 基本面
        pe = self.data['spot'].get('市盈率-动态', -1)
        pb = self.data['spot'].get('市净率', -1)
        if 0 < pe <= 20: score += 15; reasons.append(f"💎 [基本面] 低估值(PE={pe})")
        elif pe < 0: score -= 10; reasons.append(f"⚠️ [基本面] 亏损股")
        if pb > 10: score -= 5; reasons.append(f"⚠️ [基本面] 高市净率")

        # 3. 风控
        if curr.get('turnover', 0) > 15 and abs(curr['pct_change']) < 3:
            score -= 20; reasons.append("💀 [风控] 高换手滞涨")
        
        if winner_pct > 95: score -= 10; reasons.append(f"⚠️ [筹码] 获利盘高({int(winner_pct)}%)")
        elif winner_pct < 5: score += 10; reasons.append(f"💰 [筹码] 获利盘低({int(winner_pct)}%)")

        # 4. 战法
        is_low = close < curr['ma60'] * 1.15
        if is_low and curr.get('turnover', 0) > 3 and curr['vol_ratio'] > 1.8:
            signals.append("主力启动")
            reasons.append("🔥 [组合A] 低位放量启动")
            score += 15
            priority_verdict = "买入"
        elif close > curr['ma20'] and curr.get('turnover', 0) < 3 and 0.7 < curr['vol_ratio'] < 1.3:
            signals.append("主力锁筹")
            reasons.append("🔒 [组合A] 缩量锁筹")
            score += 10
            if priority_verdict is None: priority_verdict = "持有"

        if curr['dif'] > curr['dea'] and curr['rsi'] > 80:
            signals.append("假买点")
            reasons.append("🚫 [组合B] MACD金叉但RSI过热")
            score -= 5
            if priority_verdict == "买入": priority_verdict = "观察"
        
        if curr['j'] < 0: reasons.append(f"📈 [指标] J值超卖"); score += 10
        elif curr['j'] > 100: reasons.append(f"📉 [指标] J值钝化"); score -= 5
        
        if close < curr['dn'] and (flow_val > 0.5 or curr['cmf'] > 0.1):
            signals.append("黄金坑")
            reasons.append("💰 [组合C] 跌破下轨+资金流入")
            score += 20
            priority_verdict = "低吸"
            
        if curr['bb_width'] < 0.10: reasons.append(f"⚡ [变盘] 布林带宽收窄")
        if curr['cmf'] > 0.1: score += 5; reasons.append(f"🌊 [资金] CMF积极")

        return signals, reasons, priority_verdict, score

    # ================= 7. 综合分析主控 =================
    def _analyze(self):
        df = self._calc_indicators(self.data['hist'].copy())
        winner_pct = self._calc_chip_winner(df)
        
        curr = df.iloc[-1]
        close = curr['close']
        
        flow_val = 0
        if not self.data['flow'].empty and '主力净流入净额' in self.data['flow'].columns:
            try: flow_val = round(self.data['flow']['主力净流入净额'].iloc[-3:].sum() / 1e8, 2)
            except: pass
            
        s_score, s_msg = self._analyze_sentiment()
        bull_pats, bear_pats, k_score = self._analyze_pattern_full(df)
        combo_signals, combo_logic, combo_verdict, final_score = self._check_combo_logic(curr, flow_val, s_score, k_score, winner_pct)
        stop_price = close - 2 * curr['atr']
        
        verdict = "观望"; risk = "中"
        
        if s_score < -10:
            verdict = "避险卖出"; risk = "极高"
        elif close < stop_price:
            verdict = "清仓止损"; risk = "极高"; combo_logic.insert(0, f"❌ [风控] 跌破ATR止损")
        elif "断头铡刀" in bear_pats or "三只乌鸦" in bear_pats:
            verdict = "离场"; risk = "高"; combo_logic.append(f"❌ [K线] 恶劣形态")
        elif combo_verdict:
            verdict = combo_verdict
        elif final_score >= 60:
            verdict = "买入" if flow_val > 0 else "观察"
            risk = "低" if flow_val > 0 else "中"
        elif final_score < 0:
            verdict = "减仓"; risk = "高"

        risk = "高" if verdict in ["清仓", "卖出", "离场"] else risk

        base_pos = 0
        if verdict in ["买入", "持有", "主力锁筹"]: base_pos = 60
        if final_score > 80: base_pos = 80
        if "低吸" in verdict: base_pos = 30
        if s_score < 0: base_pos = max(0, base_pos - 20)
        
        curr_1 = df.iloc[-2]; curr_2 = df.iloc[-3]
        
        self.report.update({
            "verdict": verdict, "risk_level": risk, 
            "score": int(final_score), "kelly_pos": base_pos, 
            "logic": combo_logic, "signals": combo_signals,
            "patterns_bull": bull_pats, "patterns_bear": bear_pats
        })

        self._add_metric("核心指标", f"RSI:{int(curr['rsi'])}", f"ATR:{round(curr['atr'],2)}", "RSI>80过热", "-")
        self._add_metric("趋势数据", f"ADX:{int(curr['adx'])}", f"CCI:{int(curr['cci'])}", "ADX>25强趋势", "-")
        self._add_metric("资金筹码", f"主力:{flow_val}亿", f"获利盘:{int(winner_pct)}%", "获利>90%有风险", "-")
        spot_name = self.data['spot'].get('名称', self.symbol)
        pe_val = self.data['spot'].get('市盈率-动态','-')
        self._add_metric("基本面/舆情", f"PE:{pe_val}", f"舆情:{s_score}", "PE<20低估", "-")
        
        self.history_metrics = {
            "pct_0": curr['pct_change'], "pct_1": curr_1['pct_change'], "pct_2": curr_2['pct_change'],
            "cmf_0": curr['cmf'], "cmf_1": curr_1['cmf'], "cmf_2": curr_2['cmf']
        }
        
        # [优化点] 增强点位输出 (区分压力与支撑)
        # 1. 动态止损
        self.levels.append(["🔴 止损(ATR)", round(stop_price, 2), "硬止损位"])
        # 2. 均线系统 (根据当前价格判断是压是撑)
        ma20 = curr['ma20']; ma60 = curr['ma60']
        ma20_type = "🟢 MA20支撑" if close > ma20 else "🔴 MA20压力"
        ma60_type = "🟢 生命线(MA60)" if close > ma60 else "🔴 生命线(MA60)"
        self.levels.append([ma20_type, round(ma20, 2), "趋势线"])
        self.levels.append([ma60_type, round(ma60, 2), "牛熊分界"])
        # 3. 箱体 (20日)
        high_20 = df['high'].tail(20).max()
        low_20 = df['low'].tail(20).min()
        self.levels.append(["🔴 近期箱顶", round(high_20, 2), "20日新高压力"])
        self.levels.append(["🟢 近期箱底", round(low_20, 2), "20日新低支撑"])
        # 4. 筹码成本 (粗略估算60日均价)
        avg_cost = df['close'].tail(60).mean()
        self.levels.append(["🌊 筹码均价", round(avg_cost, 2), "60日成本区"])
        # 5. 布林轨
        self.levels.append(["🔴 布林上轨", round(curr['up'], 2), "冲高回落压力"])
        self.levels.append(["🟢 布林下轨", round(curr['dn'], 2), "超跌反弹支撑"])

    def _add_metric(self, name, val1, val2, explanation, logic):
        self.metrics.append({"维度": name, "数据1": val1, "数据2": val2, "判定逻辑": explanation})

    def save_excel(self):
        if not self._fetch_data(): return
        self._analyze()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        spot_name = self.data['spot'].get('名称', self.symbol)
        filename = f"{self.symbol}_{spot_name}_增强版_{timestamp}.xlsx"
        
        print(f"💾 生成报告: {filename} ...")
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            s_data = [
                ["代码", self.symbol], ["名称", spot_name],
                ["建议", self.report['verdict']], ["总分", self.report['score']],
                ["仓位", f"{self.report['kelly_pos']}%"], ["风险", self.report['risk_level']],
                ["组合战法", " | ".join(self.report['signals'])],
                ["", ""], ["决策逻辑", "\n".join(self.report['logic'])]
            ]
            pd.DataFrame(s_data, columns=["项目", "内容"]).to_excel(writer, sheet_name='决策看板', index=False)
            
            metrics_df = pd.DataFrame(self.metrics)
            extra_rows = [
                {"维度":"涨跌幅回顾", "数据1":f"今:{round(self.history_metrics['pct_0'],2)}%", "数据2":f"昨:{round(self.history_metrics['pct_1'],2)}%", "判定逻辑":"近期走势"},
                {"维度":"资金回顾", "数据1":f"今:{round(self.history_metrics['cmf_0'],2)}", "数据2":f"昨:{round(self.history_metrics['cmf_1'],2)}", "判定逻辑":"CMF趋势"}
            ]
            metrics_df = pd.concat([metrics_df, pd.DataFrame(extra_rows)], ignore_index=True)
            metrics_df.to_excel(writer, sheet_name='详细指标', index=False)
            
            pd.DataFrame(self.levels, columns=["类型", "价格", "说明"]).to_excel(writer, sheet_name='点位管理', index=False)
            
            # 字典页
            patterns_desc = [
                ['形态名称', '类型', '大白话说明'],
                ['早晨之星', '买入', '底部三日组合：阴线+星线+阳线，强力见底'],
                ['锤子线', '买入', '底部长下影线，主力试盘后拉回，支撑强'],
                ['倒锤头', '买入', '底部长上影线，主力低位试盘，抛压减轻'],
                ['阳包阴', '买入', '今日阳线完全包住昨日阴线，多头反击'],
                ['曙光初现', '买入', '大阴线后低开高走，阳线刺入阴线一半'],
                ['平底', '买入', '两日最低价相同，筑底成功'],
                ['多头孕线', '买入', '长阴包含小K线，底部孕育，变盘在即'],
                ['红三兵', '买入', '连续三天阳线稳步推升'],
                ['上升三法', '买入', '大阳后接三小阴不破低，再接大阳'],
                ['多方炮', '买入', '阳阴阳组合，洗盘结束，再次上攻'],
                ['向上缺口', '买入', '向上跳空不回补，主力强势特征'],
                ['一阳穿三线', '买入', '大阳线同时突破5/10/20均线'],
                ['倍量过左峰', '买入', '成交量翻倍且价格突破前期高点'],
                ['金蜘蛛', '买入', '均线粘合后放量向上发散'],
                ['仙人指路', '买入', '今日大阳线突破昨日的长上影线'],
                ['旭日东升', '买入', '大阴线后高开高走，包含前一日阴线'],
                ['岛形反转(底)', '买入', '下跌缺口+盘整+上涨缺口，超强反转'],
                ['踢脚线', '买入', '大阴线后直接高开高走，主力暴力反转'],
                ['蜻蜓点水', '买入', '股价回踩均线后立即弹起'],
                ['黄昏之星', '卖出', '顶部三日组合：阳线+星线+阴线'],
                ['乌云盖顶', '卖出', '大阳后接大阴，吃掉一半涨幅'],
                ['阴包阳', '卖出', '空头吞噬，阴线包住阳线'],
                ['三只乌鸦', '卖出', '连续三根阴线杀跌'],
                ['射击之星', '卖出', '高位长上影线，冲高回落'],
                ['吊颈线', '卖出', '高位长下影线，主力诱多'],
                ['断头铡刀', '卖出', '一阴断多线，趋势崩塌'],
                ['向下缺口', '卖出', '向下跳空不回补，极弱势'],
                ['倾盆大雨', '卖出', '低开低走大阴线，吞没前日涨幅'],
                ['空头孕线', '卖出', '高位长阳包含小K线，滞涨信号'],
                ['岛形反转(顶)', '卖出', '上涨缺口+盘整+下跌缺口，见顶信号'],
                ['墓碑线', '卖出', '高位T字线，多头力竭']
            ]
            pd.DataFrame(patterns_desc[1:], columns=patterns_desc[0]).to_excel(writer, sheet_name='形态图解', index=False)

            indicators_desc = [
                ['指标名称', '实战含义', '判断标准'],
                ['量比', '量能变化', '>1.5为放量；0.5-1.0为缩量(锁筹)'],
                ['CMF', '资金流', '连续为正且递增，说明主力持续拿货'],
                ['ADX', '趋势强度', '>25表示趋势强劲；<20表示震荡(观望)'],
                ['RSI', '强弱指标', '50-80为强势区，>80过热，<20超卖'],
                ['CCI', '顺势指标', '>100表示进入加速区，<-100表示超跌'],
                ['J值(KDJ)', '超买超卖', '<0为超跌反弹机会；>100为钝化风险'],
                ['ATR', '真实波幅', '用于计算动态止损位，波动越大止损越宽'],
                ['BIAS', '乖离率', '正值过大要回调，负值过大有反弹'],
                ['布林带宽', '变盘前兆', '数值越小(<0.10)说明筹码越集中，即将变盘'],
                ['PE(市盈率)', '估值', '0<PE<20为低估，PE<0为亏损'],
                ['PB(市净率)', '资产价格', 'PB>10风险较高'],
                ['获利盘%', '筹码分布', '>90%意味上方无套牢但有抛压；<10%为超跌']
            ]
            pd.DataFrame(indicators_desc[1:], columns=indicators_desc[0]).to_excel(writer, sheet_name='指标说明书', index=False)

        print(f"✅ 完成！请下载。")

# ================= 7. 程序入口 =================
if __name__ == "__main__":
    print("="*40)
    print("🚀 Alpha Galaxy 最终增强版 (More Levels)")
    print("👉 输入股票代码并回车 (输入 q 退出)")
    print("="*40)
    
    while True:
        try:
            print("\n" + "-"*30) 
            code = input(">> 请输入代码: ").strip()
            if code.lower() in ['q', 'exit', 'quit']:
                print("程序已退出。")
                break
            if code: 
                AlphaGalaxyUltimate(code).save_excel()
        except KeyboardInterrupt:
            print("\n程序已停止。")
            break
        except Exception as e:
            print(f"发生未知错误: {e}")
