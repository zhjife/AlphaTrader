# -*- coding: utf-8 -*-
"""
Alpha Galaxy Omni-Logic Ultimate (完全体·增强版) - EastMoney Optimized
================================================
【数据源】: 东方财富 (Akshare)
【特  性】: 实时数据、资金流、舆情分析、全指标
【优  化】: 增加重试机制，增加接口容错
"""

import akshare as ak
import pandas as pd
import numpy as np
from snownlp import SnowNLP
from datetime import datetime, timedelta
import warnings
import os
import time  # 新增 time 用于重试延迟

warnings.filterwarnings('ignore')

class AlphaGalaxyUltimate:
    def __init__(self, symbol):
        self.raw_symbol = str(symbol).strip()
        self.is_index = False
        
        # 智能识别指数
        # 逻辑：以sh/sz开头且后续为指数常用代码，或399开头
        lower_sym = self.raw_symbol.lower()
        if lower_sym.startswith(('sh000', 'sh000001', 'sz399', '399')):
            self.is_index = True
        # 特殊：如果输入纯数字 000001，默认当做平安银行；输入 sh000001 才是上证指数
        
        self.symbol = self.raw_symbol
        self.data = {}
        self.report = {
            "verdict": "观望", "risk_level": "中", 
            "score": 0, "mode": "震荡", "kelly_pos": 0, 
            "logic": [], "signals": [],
            "patterns_bull": [], "patterns_bear": [],
            "box_info": {} 
        }
        self.metrics = []
        self.levels = []
        self.history_metrics = {}
        
        self.index_name = "市场指数" if self.is_index else "个股"

    # ================= 1. 数据中台 (东方财富·高强版) =================
    def _fetch_data(self):
        print(f"🚀 [全维扫描] 正在连接东方财富接口: {self.symbol} ...")
        
        # 定义重试次数
        max_retries = 3
        
        # --- 1.1 K线数据获取 (核心) ---
        hist = None
        for attempt in range(max_retries):
            try:
                end = datetime.now().strftime("%Y%m%d")
                start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
                
                if self.is_index:
                    # 指数处理逻辑
                    code = self.symbol
                    # 补全前缀以适配接口
                    if code.isdigit():
                        if code.startswith('000'): code = 'sh' + code
                        elif code.startswith('399'): code = 'sz' + code
                    
                    try:
                        # 尝试接口 A: 东方财富指数历史
                        hist = ak.stock_zh_index_daily_em(symbol=code, start_date=start, end_date=end)
                    except:
                        # 尝试接口 B: 新浪指数历史 (备用)
                        clean_code = code.replace('sh', '').replace('sz', '')
                        hist = ak.stock_zh_index_daily(symbol=clean_code)
                
                else:
                    # 个股处理逻辑
                    # 清洗代码：akshare 个股接口通常只需要数字，或者特定的 sh/sz 前缀
                    # stock_zh_a_hist 需要纯数字代码
                    code_num = "".join(filter(str.isdigit, self.symbol))
                    
                    # 尝试接口 A: 东方财富个股历史 (复权数据)
                    hist = ak.stock_zh_a_hist(symbol=code_num, period='daily', start_date=start, end_date=end, adjust='qfq')
                
                if hist is not None and not hist.empty:
                    break # 获取成功，跳出重试
                    
            except Exception as e:
                print(f"⚠️ 第 {attempt + 1} 次尝试连接失败: {e}")
                time.sleep(1) # 休息1秒重试

        # 如果重试后依然失败
        if hist is None or hist.empty:
            print(f"❌ 无法连接数据源。请检查：\n1. 代码 {self.symbol} 是否正确\n2. 是否在海外IP(GitHub Codespaces)被墙\n3. 请尝试本地运行。")
            return False

        # 标准化列名
        rename_map = {
            '日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', 
            '成交量':'volume', '成交额':'amount', '换手率':'turnover',
            '振幅':'amplitude', '涨跌幅':'pct_chg', '涨跌额':'chg'
        }
        hist.rename(columns=rename_map, inplace=True)
        self.data['hist'] = hist

        # --- 1.2 实时快照 (Spot) ---
        try:
            if self.is_index:
                # 指数实时
                code_num = "".join(filter(str.isdigit, self.symbol))
                try:
                    spot_df = ak.stock_zh_index_spot_em(symbol=code_num) # 某些指数代码需尝试不同参数
                    if spot_df.empty: spot_df = ak.stock_zh_index_spot()
                except:
                    spot_df = pd.DataFrame() # 兜底
                
                if not spot_df.empty and '最新价' in spot_df.columns:
                     # 尝试找到对应的行，若找不到则取第一行
                    self.data['spot'] = spot_df.iloc[0].to_dict()
                else:
                    # 构造假数据以免报错
                    self.data['spot'] = {'名称': self.symbol, '最新价': hist.iloc[-1]['close']}
            else:
                # 个股实时 (akshare 实时接口拉取所有A股，速度较慢，但数据全)
                # 优化：直接尝试单股接口如果存在(akshare变动频繁)，这里用通用逻辑
                spot_all = ak.stock_zh_a_spot_em()
                code_num = "".join(filter(str.isdigit, self.symbol))
                target = spot_all[spot_all['代码'] == code_num]
                
                if not target.empty:
                    target = target.copy()
                    # 强制转数值
                    for col in ['市盈率-动态', '市净率', '总市值', '换手率', '最新价']:
                        if col in target.columns:
                            target[col] = pd.to_numeric(target[col], errors='coerce')
                    self.data['spot'] = target.iloc[0].to_dict()
                else:
                    # 如果实时接口找不到（如停牌），用K线最后一日填充
                    self.data['spot'] = {
                        '名称': self.symbol, 
                        '最新价': hist.iloc[-1]['close'], 
                        '市盈率-动态': -1, 
                        '市净率': -1, 
                        '换手率': hist.iloc[-1].get('turnover', 0)
                    }
        except Exception as e:
            print(f"⚠️ 实时数据获取受限: {e} (将使用收盘数据代替)")
            self.data['spot'] = {'名称': self.symbol, '最新价': hist.iloc[-1]['close'], '市盈率-动态': -1}

        # --- 1.3 资金流 & 舆情 (容错处理) ---
        # 如果是指数，或者网络不通，这部分直接跳过，不影响主程序运行
        self.data['flow'] = pd.DataFrame()
        self.data['news'] = pd.DataFrame()
        
        if not self.is_index:
            try:
                # 资金流
                code_num = "".join(filter(str.isdigit, self.symbol))
                market_type = "sh" if code_num.startswith("6") else "sz" # 简易判断
                flow = ak.stock_individual_fund_flow(stock=code_num, market=market_type)
                if flow is not None and not flow.empty:
                    self.data['flow'] = flow.sort_values('日期').tail(10)
            except:
                pass # 资金流接口常变，失败则忽略
                
            try:
                # 个股新闻
                code_num = "".join(filter(str.isdigit, self.symbol))
                news = ak.stock_news_em(symbol=code_num)
                if news is not None and not news.empty:
                    self.data['news'] = news
            except:
                pass # 舆情接口同理

        return True

    # ================= 2. 舆情分析引擎 (保持不变) =================
    def _analyze_sentiment(self):
        # ... (保持原代码不变) ...
        if self.is_index: return 0, "指数不分析个股舆情"
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
    
    # ... (其余方法 _calc_indicators, _analyze_pattern_full 等均保持原代码不变) ...
    # 为节省篇幅，请确保将原文件剩余部分（从 _calc_indicators 开始到最后）完整复制到这里。
    # 也就是将上面 _fetch_data 替换掉原文件对应部分即可。

    # ================= 2. 舆情分析引擎 =================
    def _analyze_sentiment(self):
        if self.is_index: return 0, "指数不分析个股舆情"
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

    # ================= 3. 指标计算 (完整保留) =================
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
        
        # RSI (Wilder平滑，保持原版逻辑)
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

        # ADX & CCI & BIAS (完整保留)
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
        
        # CMF & Vol Ratio & PCT (完整保留)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        mf_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 0.01)
        df['cmf'] = (mf_mult * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vol_ma5'] = df['volume'].rolling(5).mean().shift(1)
        df['vol_ratio'] = df['volume'] / df['vol_ma5']
        df['pct_change'] = df['close'].pct_change() * 100
        
        return df

    # ================= 4. 筹码 & 箱体 (新增+原有) =================
    def _calc_chip_winner(self, df):
        if self.is_index: return 50 # 指数无筹码概念
        if len(df) < 120: return 50
        sub = df.tail(60).copy()
        current_price = df['close'].iloc[-1]
        sub['avg_price'] = (sub['open'] + sub['close'] + sub['high'] + sub['low']) / 4
        winner_vol = sub[sub['avg_price'] < current_price]['volume'].sum()
        total_vol = sub['volume'].sum()
        if total_vol == 0: return 0
        return (winner_vol / total_vol) * 100

    def _calc_box_theory(self, df):
        """新增：60日箱体理论深度分析"""
        if len(df) < 60: return {}
        subset = df.tail(60)
        box_high = subset['high'].max()
        box_low = subset['low'].min()
        curr_price = df.iloc[-1]['close']
        
        box_height = box_high - box_low
        box_mid = (box_high + box_low) / 2
        
        if box_height == 0: pos_pct = 50
        else: pos_pct = (curr_price - box_low) / box_height * 100
        
        status = "箱体内震荡"
        suggestion = "高抛低吸"
        
        # 判定逻辑
        if curr_price > box_high:
            status = "突破箱体上沿(牛市特征)"
            suggestion = "主升浪持有/追涨"
        elif curr_price < box_low:
            status = "跌破箱体下沿(破位)"
            suggestion = "止损/观望"
        elif 80 <= pos_pct <= 100:
            status = "箱体顶部区域"
            suggestion = "注意压力，减仓或等待突破"
        elif 0 <= pos_pct <= 20:
            status = "箱体底部区域"
            suggestion = "支撑较强，可尝试低吸"
        
        return {
            "box_high": box_high,
            "box_low": box_low,
            "box_mid": box_mid,
            "pos_pct": pos_pct,
            "status": status,
            "suggestion": suggestion
        }

    # ================= 5. K线形态识别 (完整恢复) =================
    def _analyze_pattern_full(self, df):
        # 这里完全恢复了第一版的判断逻辑，没有任何删减
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

        # [买入形态 - 完整版]
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

        # [卖出形态 - 完整版]
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

    # ================= 6. 核心逻辑 (完整保留 + 箱体叠加) =================
    def _check_combo_logic(self, curr, flow_val, sentiment_score, k_score, winner_pct, box_info):
        signals = []
        reasons = []
        score = 0
        priority_verdict = None 
        close = curr['close']
        
        # 1. 基础技术 (保留)
        if close > curr['ma20']: score += 20
        if curr['adx'] > 25: score += 10
        if curr['cci'] > 100: score += 10
        score += k_score + sentiment_score

        # 2. 基本面 (保留，指数不判断PE)
        pe = self.data['spot'].get('市盈率-动态', -1)
        pb = self.data['spot'].get('市净率', -1)
        if not self.is_index:
            if 0 < pe <= 20: score += 15; reasons.append(f"💎 [基本面] 低估值(PE={pe})")
            elif pe < 0: score -= 10; reasons.append(f"⚠️ [基本面] 亏损股")
            if pb > 10: score -= 5; reasons.append(f"⚠️ [基本面] 高市净率")

        # 3. 风控 (保留，指数不判断换手/获利盘)
        if not self.is_index:
            if curr.get('turnover', 0) > 15 and abs(curr['pct_change']) < 3:
                score -= 20; reasons.append("💀 [风控] 高换手滞涨")
            
            if winner_pct > 95: score -= 10; reasons.append(f"⚠️ [筹码] 获利盘高({int(winner_pct)}%)")
            elif winner_pct < 5: score += 10; reasons.append(f"💰 [筹码] 获利盘低({int(winner_pct)}%)")

        # 4. 箱体逻辑 (新增)
        if box_info:
            b_status = box_info['status']
            if "突破箱体上沿" in b_status:
                score += 25
                signals.append("箱体突破")
                reasons.append("🚀 [箱体] 有效突破60日箱顶，空间打开")
                priority_verdict = "买入"
            elif "跌破箱体下沿" in b_status:
                score -= 30
                reasons.append("❌ [箱体] 跌破60日箱底，破位风险")
                priority_verdict = "清仓"
            elif "箱体底部" in b_status and curr['j'] < 10:
                score += 20
                signals.append("箱体低吸")
                reasons.append("💰 [箱体] 底部区域+J值超卖")
                priority_verdict = "低吸"
            elif "箱体顶部" in b_status and curr['j'] > 90:
                score -= 15
                reasons.append("⚠️ [箱体] 顶部区域+J值钝化，注意回调")

        # 5. 战法 (完整保留)
        is_low = close < curr['ma60'] * 1.15
        if is_low and curr.get('turnover', 0) > 3 and curr['vol_ratio'] > 1.8 and not self.is_index:
            signals.append("主力启动")
            reasons.append("🔥 [组合A] 低位放量启动")
            score += 15
            priority_verdict = "买入"
        elif close > curr['ma20'] and curr.get('turnover', 0) < 3 and 0.7 < curr['vol_ratio'] < 1.3 and not self.is_index:
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
        box_info = self._calc_box_theory(df) # 新增箱体计算
        
        curr = df.iloc[-1]
        close = curr['close']
        
        flow_val = 0
        if not self.data['flow'].empty and '主力净流入净额' in self.data['flow'].columns:
            try: flow_val = round(self.data['flow']['主力净流入净额'].iloc[-3:].sum() / 1e8, 2)
            except: pass
            
        s_score, s_msg = self._analyze_sentiment()
        bull_pats, bear_pats, k_score = self._analyze_pattern_full(df)
        combo_signals, combo_logic, combo_verdict, final_score = self._check_combo_logic(curr, flow_val, s_score, k_score, winner_pct, box_info)
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
            verdict = "买入" if flow_val >= 0 else "观察"
            risk = "低" if flow_val > 0 else "中"
        elif final_score < 0:
            verdict = "减仓"; risk = "高"

        risk = "高" if verdict in ["清仓", "卖出", "离场"] else risk

        base_pos = 0
        if verdict in ["买入", "持有", "主力锁筹"]: base_pos = 60
        if final_score > 80: base_pos = 80
        if "低吸" in verdict: base_pos = 30
        if s_score < 0: base_pos = max(0, base_pos - 20)
        
        self.report.update({
            "verdict": verdict, "risk_level": risk, 
            "score": int(final_score), "kelly_pos": base_pos, 
            "logic": combo_logic, "signals": combo_signals,
            "patterns_bull": bull_pats, "patterns_bear": bear_pats,
            "box_info": box_info
        })

        self._add_metric("核心指标", f"RSI:{int(curr['rsi'])}", f"ATR:{round(curr['atr'],2)}", "RSI>80过热", "-")
        self._add_metric("趋势数据", f"ADX:{int(curr['adx'])}", f"CCI:{int(curr['cci'])}", "ADX>25强趋势", "-")
        self._add_metric("资金筹码", f"主力:{flow_val}亿", f"获利盘:{int(winner_pct)}%", "获利>90%有风险", "-")
        spot_name = self.data['spot'].get('名称', self.symbol)
        pe_val = self.data['spot'].get('市盈率-动态','-')
        self._add_metric("基本面/舆情", f"PE:{pe_val}", f"舆情:{s_score}", "PE<20低估", "-")
        
        self.history_metrics = {
            "pct_0": curr['pct_change'], "pct_1": df.iloc[-2]['pct_change'], 
            "cmf_0": curr['cmf'], "cmf_1": df.iloc[-2]['cmf']
        }
        
        # [点位增强] 输出具体压力支撑 (带逻辑判断)
        # 1. 动态止损
        self.levels.append(["🔴 止损(ATR)", round(stop_price, 2), "硬止损位"])
        # 2. 箱体点位 (新增)
        if box_info:
            self.levels.append(["⬛ 箱体顶部", round(box_info['box_high'], 2), "60日震荡上沿(强压)"])
            self.levels.append(["⬛ 箱体中轴", round(box_info['box_mid'], 2), "强弱分界线"])
            self.levels.append(["⬛ 箱体底部", round(box_info['box_low'], 2), "60日震荡下沿(强撑)"])
        # 3. 均线系统 (根据当前价格判断是压是撑)
        ma20 = curr['ma20']; ma60 = curr['ma60']
        ma20_type = "🟢 MA20支撑" if close > ma20 else "🔴 MA20压力"
        ma60_type = "🟢 生命线(MA60)" if close > ma60 else "🔴 生命线(MA60)"
        self.levels.append([ma20_type, round(ma20, 2), "趋势线"])
        self.levels.append([ma60_type, round(ma60, 2), "牛熊分界"])
        # 4. 筹码成本 (粗略估算)
        if not self.is_index:
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
        filename = f"{'指数' if self.is_index else '个股'}_{self.symbol}_{spot_name}_{timestamp}.xlsx"
        
        print(f"💾 生成报告: {filename} ...")
        
        box_data = self.report.get('box_info', {})
        box_str = f"{box_data.get('status','-')} (位置:{int(box_data.get('pos_pct',0))}%)" if box_data else "数据不足"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 1. 决策看板 (包含新增的箱体分析)
            s_data = [
                ["代码", self.symbol], ["名称", spot_name],
                ["建议", self.report['verdict']], ["总分", self.report['score']],
                ["仓位", f"{self.report['kelly_pos']}%"], ["风险", self.report['risk_level']],
                ["", ""],
                ["【箱体分析】", box_str],
                ["操作建议", box_data.get('suggestion', '-')],
                ["上方压力", round(box_data.get('box_high', 0), 2)],
                ["下方支撑", round(box_data.get('box_low', 0), 2)],
                ["", ""],
                ["组合战法", " | ".join(self.report['signals'])],
                ["", ""], ["决策逻辑", "\n".join(self.report['logic'])]
            ]
            pd.DataFrame(s_data, columns=["项目", "内容"]).to_excel(writer, sheet_name='决策看板', index=False)
            
            # 2. 详细指标 (保留历史回顾)
            metrics_df = pd.DataFrame(self.metrics)
            extra_rows = [
                {"维度":"涨跌幅回顾", "数据1":f"今:{round(self.history_metrics['pct_0'],2)}%", "数据2":f"昨:{round(self.history_metrics['pct_1'],2)}%", "判定逻辑":"近期走势"},
                {"维度":"资金回顾", "数据1":f"今:{round(self.history_metrics['cmf_0'],2)}", "数据2":f"昨:{round(self.history_metrics['cmf_1'],2)}", "判定逻辑":"CMF趋势"}
            ]
            metrics_df = pd.concat([metrics_df, pd.DataFrame(extra_rows)], ignore_index=True)
            metrics_df.to_excel(writer, sheet_name='详细指标', index=False)
            
            # 3. 点位管理 (动态)
            pd.DataFrame(self.levels, columns=["类型", "价格", "说明"]).to_excel(writer, sheet_name='点位管理', index=False)
            
            # 4. 形态图解 (完整恢复)
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

            # 5. 指标说明 (完整恢复)
            indicators_desc = [
                ['指标名称', '实战含义', '判断标准'],
                ['箱体位置%', '0%为箱底，100%为箱顶，>100%为突破', '新增'],
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

        print(f"✅ 完成！请下载文件: {filename}")

# ================= 8. 程序入口 =================
if __name__ == "__main__":
    print("="*50)
    print("🚀 Alpha Galaxy Ultimate (Complete & Enhanced)")
    print("👉 个股模式: 输入 600519")
    print("👉 指数模式: 输入 sh000001 (上证), sz399001 (深成)")
    print("👉 箱体增强: 自动判断突破/破位/震荡状态")
    print("="*50)
    
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
