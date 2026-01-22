import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from snownlp import SnowNLP
from datetime import datetime, timedelta
import warnings
import io
import plotly.graph_objects as go # 保留导入但不在UI层渲染，防报错

# ================= 1. 页面配置 (回归 V1 简洁配置) =================
st.set_page_config(page_title="Alpha Galaxy Ultimate (稳定版)", layout="wide")
warnings.filterwarnings('ignore')

# ================= 2. 静态数据 (来自 V2，保留完整形态库) =================
PATTERN_DESCRIPTIONS = [
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

INDICATORS_DESCRIPTIONS = [
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

# ================= 3. 全市场数据缓存层 (来自 V2，保留核心优化) =================

@st.cache_data(ttl=600)  # 全局快照缓存10分钟
def get_global_market_spot():
    try:
        df = ak.stock_zh_a_spot_em()
        df['代码'] = df['代码'].astype(str)
        spot_dict = df.set_index('代码').to_dict('index')
        return spot_dict, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=600)  # 个股历史数据缓存
def fetch_stock_history(symbol, is_index=False):
    raw_symbol = str(symbol)
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
    
    try:
        if is_index:
            code = raw_symbol
            if code.isdigit():
                if code.startswith('000'): code = 'sh' + code
                elif code.startswith('399'): code = 'sz' + code
            try: hist = ak.stock_zh_index_daily_em(symbol=code, start_date=start, end_date=end)
            except: hist = ak.stock_zh_index_daily(symbol=code)
            flow_data = pd.DataFrame()
            news_data = pd.DataFrame()
        else:
            hist = ak.stock_zh_a_hist(symbol=raw_symbol, period='daily', start_date=start, end_date=end, adjust='qfq')
            try:
                flow = ak.stock_individual_fund_flow(stock=raw_symbol, market="sh" if raw_symbol.startswith("6") else "sz")
                flow_data = flow.sort_values('日期').tail(10) if (flow is not None and not flow.empty) else pd.DataFrame()
            except: flow_data = pd.DataFrame()
            try: news_data = ak.stock_news_em(symbol=raw_symbol)
            except: news_data = pd.DataFrame()

        if hist is None or hist.empty: return None, "K线数据为空"
        hist.rename(columns={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume', '换手率':'turnover'}, inplace=True)
        return {'hist': hist, 'flow': flow_data, 'news': news_data}, None
    except Exception as e:
        return None, str(e)

def get_data_bundle(code):
    is_index = False
    spot_info = {}
    if code.lower().startswith(('sh', 'sz')) or code.startswith('399') or code in ['000001', '000300', '000016', '000905']:
        if code.lower().startswith(('sh', 'sz')) or code.startswith('399'): is_index = True

    if not is_index:
        global_spot, err = get_global_market_spot()
        if global_spot and code in global_spot:
            raw_spot = global_spot[code]
            for col in ['市盈率-动态', '市净率', '总市值', '换手率', '最新价']:
                 if col in raw_spot:
                     try: raw_spot[col] = float(raw_spot[col])
                     except: pass
            spot_info = raw_spot
        else:
            try:
                spot = ak.stock_zh_a_spot_em()
                target = spot[spot['代码'] == code]
                if not target.empty: spot_info = target.iloc[0].to_dict()
                else: spot_info = {'名称': code, '最新价': 0, '市盈率-动态': -1}
            except: pass
    else:
        try:
            simple_code = code.replace('sh','').replace('sz','')
            spot_df = ak.stock_zh_index_spot_em(symbol=simple_code)
            if not spot_df.empty: spot_info = spot_df.iloc[0].to_dict()
            else: spot_info = {'名称': code, '最新价': 0}
        except: spot_info = {'名称': code, '最新价': 0}

    hist_bundle, err = fetch_stock_history(code, is_index)
    if err: return None, False, err
    if '最新价' not in spot_info or spot_info['最新价'] == 0:
        spot_info['最新价'] = hist_bundle['hist'].iloc[-1]['close']
    
    return {'hist': hist_bundle['hist'], 'spot': spot_info, 'flow': hist_bundle['flow'], 'news': hist_bundle['news']}, is_index, None

# ================= 4. 核心逻辑类 (来自 V2，完全保留) =================

class AlphaGalaxyLogic:
    def __init__(self, symbol, df_hist, spot_data, flow_data, news_data, is_index):
        self.symbol = str(symbol)
        self.is_index = is_index
        self.data = {'hist': df_hist, 'spot': spot_data, 'flow': flow_data, 'news': news_data}
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

    def _analyze_sentiment(self):
        if self.is_index: return 0, "指数不分析个股舆情"
        try:
            if self.data['news'].empty: return 0, "无近期舆情"
            news_df = self.data['news'].head(10)
            titles = news_df['新闻标题'].tolist()
            full_text = "。".join(titles)
            pos_kw = ['增长', '预增', '突破', '利好', '回购', '获批', '中标', '大涨', '新高']
            neg_kw = ['立案', '调查', '亏损', '减持', '警示', '违规', '大跌', '退市', '被查']
            hard_score = 0; keywords = []
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

    def _calc_indicators(self, df):
        for w in [5, 10, 20, 60, 120, 250]: df[f'ma{w}'] = df['close'].rolling(w).mean()
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        low_9 = df['low'].rolling(9).min(); high_9 = df['high'].rolling(9).max()
        rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
        df['k'] = rsv.ewm(com=2, adjust=False).mean()
        df['d'] = df['k'].ewm(com=2, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        delta = df['close'].diff()
        up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
        for period in [6, 12, 24]:
            ema_up = up.ewm(alpha=1/period, adjust=False).mean()
            ema_down = down.ewm(alpha=1/period, adjust=False).mean()
            rs = ema_up / ema_down
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi_6'] 
        df['std'] = df['close'].rolling(20).std()
        df['up'] = df['ma20'] + 2 * df['std']
        df['dn'] = df['ma20'] - 2 * df['std']
        df['bb_width'] = (df['up'] - df['dn']) / df['ma20'] 
        df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
        df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        roll_max = df['close'].rolling(250, min_periods=1).max()
        df['drawdown'] = (df['close'] / roll_max) - 1.0
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
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        mf_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 0.01)
        df['cmf'] = (mf_mult * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vol_ma5'] = df['volume'].rolling(5).mean().shift(1)
        df['vol_ratio'] = df['volume'] / df['vol_ma5']
        df['pct_change'] = df['close'].pct_change() * 100
        return df

    def _calc_chip_winner(self, df):
        if self.is_index: return 50 
        if len(df) < 120: return 50
        sub = df.tail(60).copy()
        current_price = df['close'].iloc[-1]
        sub['avg_price'] = (sub['open'] + sub['close'] + sub['high'] + sub['low']) / 4
        winner_vol = sub[sub['avg_price'] < current_price]['volume'].sum()
        total_vol = sub['volume'].sum()
        if total_vol == 0: return 0
        return (winner_vol / total_vol) * 100

    def _calc_box_theory(self, df):
        if len(df) < 60: return {}
        subset = df.tail(60)
        box_high = subset['high'].max()
        box_low = subset['low'].min()
        curr_price = df.iloc[-1]['close']
        box_height = box_high - box_low
        box_mid = (box_high + box_low) / 2
        if box_height == 0: pos_pct = 50
        else: pos_pct = (curr_price - box_low) / box_height * 100
        status = "箱体内震荡"; suggestion = "高抛低吸"
        if curr_price > box_high: status = "突破箱体上沿(牛市特征)"; suggestion = "主升浪持有/追涨"
        elif curr_price < box_low: status = "跌破箱体下沿(破位)"; suggestion = "止损/观望"
        elif 80 <= pos_pct <= 100: status = "箱体顶部区域"; suggestion = "注意压力，减仓或等待突破"
        elif 0 <= pos_pct <= 20: status = "箱体底部区域"; suggestion = "支撑较强，可尝试低吸"
        return {"box_high": box_high, "box_low": box_low, "box_mid": box_mid, "pos_pct": pos_pct, "status": status, "suggestion": suggestion}

    def _analyze_pattern_full(self, df):
        if len(df) < 20: return [], [], 0
        bull_pats, bear_pats = [], []; score = 0 
        c = df['close'].values; o = df['open'].values; h = df['high'].values; l = df['low'].values; v = df['volume'].values
        ma5 = df['ma5'].values; ma10 = df['ma10'].values; ma20 = df['ma20'].values
        c0, c1, c2, c3, c4 = c[-1], c[-2], c[-3], c[-4], c[-5]
        o0, o1, o2, o3, o4 = o[-1], o[-2], o[-3], o[-4], o[-5]
        h0, h1 = h[-1], h[-2]; l0, l1 = l[-1], l[-2]
        v0, v1 = v[-1], v[-2]
        body0 = abs(c0 - o0); upper0 = h0 - max(c0, o0); lower0 = min(c0, o0) - l0
        is_bull0 = c0 > o0; is_bear0 = c0 < o0; is_downtrend = c0 < ma20[-1]; is_uptrend = c0 > ma20[-1]

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

    def _check_combo_logic(self, curr, flow_val, sentiment_score, k_score, winner_pct, box_info):
        signals = []; reasons = []; score = 0; priority_verdict = None 
        close = curr['close']
        
        if close > curr['ma20']: score += 20
        if curr['adx'] > 25: score += 10
        if curr['cci'] > 100: score += 10
        score += k_score + sentiment_score

        pe = self.data['spot'].get('市盈率-动态', -1); pb = self.data['spot'].get('市净率', -1)
        if not self.is_index:
            if 0 < pe <= 20: score += 15; reasons.append(f"💎 [基本面] 低估值(PE={pe})")
            elif pe < 0: score -= 10; reasons.append(f"⚠️ [基本面] 亏损股")
            if pb > 10: score -= 5; reasons.append(f"⚠️ [基本面] 高市净率")

        if not self.is_index:
            if curr.get('turnover', 0) > 15 and abs(curr['pct_change']) < 3: score -= 20; reasons.append("💀 [风控] 高换手滞涨")
            if winner_pct > 95: score -= 10; reasons.append(f"⚠️ [筹码] 获利盘高({int(winner_pct)}%)")
            elif winner_pct < 5: score += 10; reasons.append(f"💰 [筹码] 获利盘低({int(winner_pct)}%)")

        if box_info:
            b_status = box_info['status']
            if "突破箱体上沿" in b_status: score += 25; signals.append("箱体突破"); reasons.append("🚀 [箱体] 有效突破60日箱顶，空间打开"); priority_verdict = "买入"
            elif "跌破箱体下沿" in b_status: score -= 30; reasons.append("❌ [箱体] 跌破60日箱底，破位风险"); priority_verdict = "清仓"
            elif "箱体底部" in b_status and curr['j'] < 10: score += 20; signals.append("箱体低吸"); reasons.append("💰 [箱体] 底部区域+J值超卖"); priority_verdict = "低吸"
            elif "箱体顶部" in b_status and curr['j'] > 90: score -= 15; reasons.append("⚠️ [箱体] 顶部区域+J值钝化，注意回调")

        is_low = close < curr['ma60'] * 1.15
        if is_low and curr.get('turnover', 0) > 3 and curr['vol_ratio'] > 1.8 and not self.is_index:
            signals.append("主力启动"); reasons.append("🔥 [组合A] 低位放量启动"); score += 15; priority_verdict = "买入"
        elif close > curr['ma20'] and curr.get('turnover', 0) < 3 and 0.7 < curr['vol_ratio'] < 1.3 and not self.is_index:
            signals.append("主力锁筹"); reasons.append("🔒 [组合A] 缩量锁筹"); score += 10
            if priority_verdict is None: priority_verdict = "持有"

        if curr['dif'] > curr['dea'] and curr['rsi'] > 80:
            signals.append("假买点"); reasons.append("🚫 [组合B] MACD金叉但RSI过热"); score -= 5
            if priority_verdict == "买入": priority_verdict = "观察"
        
        if curr['j'] < 0: reasons.append(f"📈 [指标] J值超卖"); score += 10
        elif curr['j'] > 100: reasons.append(f"📉 [指标] J值钝化"); score -= 5
        
        if close < curr['dn'] and (flow_val > 0.5 or curr['cmf'] > 0.1):
            signals.append("黄金坑"); reasons.append("💰 [组合C] 跌破下轨+资金流入"); score += 20; priority_verdict = "低吸"
            
        if curr['bb_width'] < 0.10: reasons.append(f"⚡ [变盘] 布林带宽收窄")
        if curr['cmf'] > 0.1: score += 5; reasons.append(f"🌊 [资金] CMF积极")

        return signals, reasons, priority_verdict, score

    def analyze(self):
        df = self._calc_indicators(self.data['hist'].copy())
        winner_pct = self._calc_chip_winner(df)
        box_info = self._calc_box_theory(df)
        curr = df.iloc[-1]; close = curr['close']
        
        flow_val = 0
        if not self.data['flow'].empty and '主力净流入净额' in self.data['flow'].columns:
            try: flow_val = round(self.data['flow']['主力净流入净额'].iloc[-3:].sum() / 1e8, 2)
            except: pass
            
        s_score, s_msg = self._analyze_sentiment()
        bull_pats, bear_pats, k_score = self._analyze_pattern_full(df)
        combo_signals, combo_logic, combo_verdict, final_score = self._check_combo_logic(curr, flow_val, s_score, k_score, winner_pct, box_info)
        stop_price = close - 2 * curr['atr']
        
        verdict = "观望"; risk = "中"
        if s_score < -10: verdict = "避险卖出"; risk = "极高"
        elif close < stop_price: verdict = "清仓止损"; risk = "极高"; combo_logic.insert(0, f"❌ [风控] 跌破ATR止损")
        elif "断头铡刀" in bear_pats or "三只乌鸦" in bear_pats: verdict = "离场"; risk = "高"; combo_logic.append(f"❌ [K线] 恶劣形态")
        elif combo_verdict: verdict = combo_verdict
        elif final_score >= 60: verdict = "买入" if flow_val >= 0 else "观察"; risk = "低" if flow_val > 0 else "中"
        elif final_score < 0: verdict = "减仓"; risk = "高"

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
        pe_val = self.data['spot'].get('市盈率-动态','-')
        self._add_metric("基本面/舆情", f"PE:{pe_val}", f"舆情:{s_score}", "PE<20低估", "-")
        
        self.history_metrics = {"pct_0": curr['pct_change'], "pct_1": df.iloc[-2]['pct_change'], "cmf_0": curr['cmf'], "cmf_1": df.iloc[-2]['cmf']}
        
        self.levels.append(["🔴 止损(ATR)", round(stop_price, 2), "硬止损位"])
        if box_info:
            self.levels.append(["⬛ 箱体顶部", round(box_info['box_high'], 2), "60日震荡上沿(强压)"])
            self.levels.append(["⬛ 箱体中轴", round(box_info['box_mid'], 2), "强弱分界线"])
            self.levels.append(["⬛ 箱体底部", round(box_info['box_low'], 2), "60日震荡下沿(强撑)"])
        ma20 = curr['ma20']; ma60 = curr['ma60']
        ma20_type = "🟢 MA20支撑" if close > ma20 else "🔴 MA20压力"
        ma60_type = "🟢 生命线(MA60)" if close > ma60 else "🔴 生命线(MA60)"
        self.levels.append([ma20_type, round(ma20, 2), "趋势线"])
        self.levels.append([ma60_type, round(ma60, 2), "牛熊分界"])
        if not self.is_index:
            avg_cost = df['close'].tail(60).mean()
            self.levels.append(["🌊 筹码均价", round(avg_cost, 2), "60日成本区"])
        self.levels.append(["🔴 布林上轨", round(curr['up'], 2), "冲高回落压力"])
        self.levels.append(["🟢 布林下轨", round(curr['dn'], 2), "超跌反弹支撑"])
        
        return df 

    def _add_metric(self, name, val1, val2, explanation, logic):
        self.metrics.append({"维度": name, "数据1": val1, "数据2": val2, "判定逻辑": explanation})

    def generate_excel_bytes(self):
        spot_name = self.data['spot'].get('名称', self.symbol)
        box_data = self.report.get('box_info', {})
        box_str = f"{box_data.get('status','-')} (位置:{int(box_data.get('pos_pct',0))}%)" if box_data else "数据不足"
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
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
            metrics_df = pd.DataFrame(self.metrics)
            extra_rows = [
                {"维度":"涨跌幅回顾", "数据1":f"今:{round(self.history_metrics['pct_0'],2)}%", "数据2":f"昨:{round(self.history_metrics['pct_1'],2)}%", "判定逻辑":"近期走势"},
                {"维度":"资金回顾", "数据1":f"今:{round(self.history_metrics['cmf_0'],2)}", "数据2":f"昨:{round(self.history_metrics['cmf_1'],2)}", "判定逻辑":"CMF趋势"}
            ]
            metrics_df = pd.concat([metrics_df, pd.DataFrame(extra_rows)], ignore_index=True)
            metrics_df.to_excel(writer, sheet_name='详细指标', index=False)
            pd.DataFrame(self.levels, columns=["类型", "价格", "说明"]).to_excel(writer, sheet_name='点位管理', index=False)
            pd.DataFrame(PATTERN_DESCRIPTIONS[1:], columns=PATTERN_DESCRIPTIONS[0]).to_excel(writer, sheet_name='形态图解', index=False)
            pd.DataFrame(INDICATORS_DESCRIPTIONS[1:], columns=INDICATORS_DESCRIPTIONS[0]).to_excel(writer, sheet_name='指标说明书', index=False)
        return output.getvalue()

# ================= 5. 前端交互 (回归 V1 稳定显示模式) =================

st.title("🚀 Alpha Galaxy Ultimate (完整版)")
st.markdown("### 全维扫描 | 智能缓存 | 箱体战法")

# V1 样式的输入区
col_input, col_btn = st.columns([3, 1])
with col_input:
    stock_code = st.text_input("请输入股票代码 (如 600519)", value="", placeholder="支持A股/北交所")

with col_btn:
    st.write("") 
    st.write("") 
    run_btn = st.button("▶️ 开始扫描", type="primary", use_container_width=True)

if run_btn:
    if not stock_code:
        st.error("⚠️ 请先输入股票代码")
    else:
        with st.spinner(f"正在深度分析 {stock_code}..."):
            # 使用 V2 的数据获取函数
            data_pack, is_index, err = get_data_bundle(stock_code)
            
            if err:
                st.error(f"❌ 分析失败: {err}")
            else:
                # 使用 V2 的逻辑类
                app = AlphaGalaxyLogic(stock_code, data_pack['hist'], data_pack['spot'], data_pack['flow'], data_pack['news'], is_index)
                app.analyze()
                
                # === 界面展示 (V1 风格) ===
                excel_data = app.generate_excel_bytes()
                spot_name = app.data['spot'].get('名称', stock_code)
                st.success(f"✅ [{spot_name}] 分析完成！")
                
                # 1. 核心大屏
                r = app.report
                c1, c2, c3, c4 = st.columns(4)
                
                verdict_color = "normal" if r['verdict']=="观望" else "inverse"
                c1.metric("最终建议", r['verdict'], delta_color=verdict_color)
                c2.metric("综合评分", r['score'])
                c3.metric("建议仓位", f"{r['kelly_pos']}%")
                c4.metric("风险等级", r['risk_level'])
                
                st.divider()

                # 2. 箱体信息 (V2 特有，嵌入在 V1 布局中)
                box = r.get('box_info', {})
                if box:
                    st.info(f"📦 **箱体状态**: {box.get('status')} | 当前位置: {int(box.get('pos_pct', 0))}%")

                # 3. 核心分栏 (点位 & 逻辑)
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    st.subheader("🎯 关键点位")
                    levels_df = pd.DataFrame(app.levels, columns=["类型", "价格", "说明"])
                    st.dataframe(levels_df, use_container_width=True, hide_index=True)
                
                with c_right:
                    st.subheader("💡 决策逻辑")
                    for logic in r['logic']:
                        st.write(f"- {logic}")
                    
                    if r['signals']: st.info(f"🔥 战法信号: {', '.join(r['signals'])}")
                    if r['patterns_bull']: st.success(f"🐂 多头形态: {', '.join(r['patterns_bull'])}")
                    if r['patterns_bear']: st.error(f"🐻 空头形态: {', '.join(r['patterns_bear'])}")

                # 4. 下载按钮
                st.divider()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                file_name = f"{stock_code}_{spot_name}_稳定版_{timestamp}.xlsx"
                st.download_button(
                    label=f"📥 下载完整报告 (含所有形态图解)",
                    data=excel_data,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )