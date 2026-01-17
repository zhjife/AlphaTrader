# main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import akshare as ak
import pandas as pd
import pandas_ta as ta
import numpy as np

app = FastAPI()

# 允许跨域，方便前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProfessionalTrader:
    def __init__(self, symbol):
        self.symbol = symbol
        self.score = 50  # 初始分
        self.report = {
            "technical": [], "capital": [], "fundamental": [], 
            "risk": [], "verdict": ""
        }

    def fetch_data(self):
        try:
            # 1. 个股历史K线 (前复权, 200天)
            self.df = ak.stock_zh_a_hist(symbol=self.symbol, period="daily", adjust="qfq").tail(200)
            if len(self.df) < 60: return False
            
            # 2. 实时行情
            spot = ak.stock_zh_a_spot_em()
            self.spot_data = spot[spot['代码'] == self.symbol].iloc[0]
            
            # 3. 资金流向
            market_type = "sh" if self.symbol.startswith('6') else "sz"
            self.flow = ak.stock_individual_fund_flow(stock=self.symbol, market=market_type).tail(20)
            
            # 4. 新闻
            self.news = ak.stock_news_em(symbol=self.symbol).head(5)
            
            # 5. 大盘指数 (上证指数) 用于RPS计算
            self.index_df = ak.stock_zh_index_daily(symbol="sh000001").tail(200)
            
            return True
        except Exception as e:
            print(f"Data Fetch Error: {e}")
            return False

    # --- 模块1: 基础K线形态 (12种) ---
    def detect_candlestick_patterns(self):
        df = self.df
        k3, k2, k1 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
        
        # 辅助函数
        def body(row): return abs(row['收盘']-row['开盘'])
        def is_red(row): return row['收盘'] > row['开盘']
        def is_green(row): return row['收盘'] < row['开盘']
        def is_doji(row): return body(row) < (row['最高']-row['最低'])*0.1
        
        msgs = []
        # 1. 缺口理论
        if k3['最低'] > k2['最高']:
            self.score += 10
            msgs.append("【缺口】🚀 向上跳空缺口：多头强势逼空，若三日不补缺则为强势上涨中继。")
        
        # 2. 红三兵
        if is_red(k1) and is_red(k2) and is_red(k3) and k3['收盘']>k2['收盘']>k1['收盘']:
            self.score += 10
            msgs.append("【形态】💂 红三兵：连续三日阳线推进，多头趋势确立。")
            
        # 3. 启明星
        if is_green(k1) and is_doji(k2) and is_red(k3) and k3['收盘'] > (k1['开盘']+k1['收盘'])/2:
            self.score += 15
            msgs.append("【形态】🌅 启明星：见底回升强烈信号。")
            
        # 4. 穿头破脚 (阳包阴)
        if is_green(k2) and is_red(k3) and k3['收盘']>k2['开盘'] and k3['开盘']<k2['收盘']:
            self.score += 10
            msgs.append("【形态】🐯 阳包阴(吞没)：一阳吞两线，多头反攻。")

        self.report['technical'] += msgs

    # --- 模块2: A股特色战法 (黄金坑/蚂蚁上树/老鸭头) ---
    def analyze_special_morphology(self):
        df = self.df
        close = df['收盘']
        ma5 = ta.sma(close, length=5)
        ma10 = ta.sma(close, length=10)
        ma60 = ta.sma(close, length=60)
        
        msgs = []
        
        # 1. 老鸭头 (均线战法)
        if ma5.iloc[-1] > ma10.iloc[-1] > ma60.iloc[-1]:
            if ma5.iloc[-2] <= ma10.iloc[-2]: # 刚金叉
                self.score += 15
                msgs.append("【战法】🦆 老鸭头：均线多头回档后再次张口，主升浪特征。")
                
        # 2. 黄金坑
        curr = close.iloc[-1]
        last_ma60 = ma60.iloc[-1]
        min_10 = close.tail(10).min()
        if curr > last_ma60 and min_10 < last_ma60 * 0.95:
            self.score += 20
            msgs.append("【战法】💰 黄金坑：主力挖坑洗盘结束，强势收复生命线。")
            
        # 3. 蚂蚁上树 (5连小阳)
        recent = df.tail(5)
        red_count = sum(1 for _, r in recent.iterrows() if r['收盘']>r['开盘'])
        max_gain = max((r['收盘']-r['前收盘'])/r['前收盘'] for _, r in recent.iterrows())
        if red_count >= 4 and max_gain < 0.03:
            self.score += 15
            msgs.append("【战法】🐜 蚂蚁上树：连续小阳线温和推升，控盘极佳。")
            
        self.report['technical'] = msgs + self.report['technical']

    # --- 模块3: 筹码分布 (CYQ) ---
    def analyze_chip_distribution(self):
        df = self.df
        curr = df['收盘'].iloc[-1]
        
        # 简易估算：过去60天成交量加权均价
        total_vol = 0
        total_amt = 0
        winner_vol = 0
        
        for i in range(60):
            idx = -1 - i
            if abs(idx) > len(df): break
            row = df.iloc[idx]
            vol = row['成交量']
            price = row['收盘']
            decay = 0.98 ** i # 时间衰减
            
            eff_vol = vol * decay
            total_vol += eff_vol
            total_amt += price * eff_vol
            
            if price < curr: winner_vol += eff_vol
            
        avg_cost = total_amt / total_vol if total_vol else 0
        winner_ratio = (winner_vol / total_vol) * 100 if total_vol else 0
        
        msgs = []
        if winner_ratio > 90:
            self.score += 10
            msgs.append(f"【筹码】🏆 获利盘 {int(winner_ratio)}%，上方无套牢盘，锁仓拉升。")
        elif winner_ratio < 10:
            msgs.append(f"【筹码】🧊 获利盘仅 {int(winner_ratio)}%，底部磨底阶段。")
            
        self.report['capital'].append(f"市场平均成本约 {round(avg_cost, 2)} 元。")
        self.report['technical'] += msgs

    # --- 模块4: 相对强度 (RPS) ---
    def analyze_rps(self):
        # 个股20日涨幅 vs 大盘20日涨幅
        stock_ret = (self.df['收盘'].iloc[-1] / self.df['收盘'].iloc[-20]) - 1
        index_ret = (self.index_df['close'].iloc[-1] / self.index_df['close'].iloc[-20]) - 1
        
        alpha = stock_ret - index_ret
        if alpha > 0.1:
            self.score += 10
            self.report['technical'].append(f"【RPS】🔥 强势：近20日跑赢大盘 {round(alpha*100,1)}%。")
        elif alpha < -0.05:
            self.score -= 10
            self.report['technical'].append(f"【RPS】🥀 弱势：近20日跑输大盘 {abs(round(alpha*100,1))}%。")

    # --- 模块5: 基础分析与风控 ---
    def analyze_basics(self):
        # 资金
        net_flow = self.flow['主力净流入-净额'].iloc[-1]
        if net_flow > 0:
            self.score += 5
            self.report['capital'].append(f"【资金】今日主力净流入 {round(net_flow/10000)} 万元。")
        else:
            self.score -= 5
            self.report['capital'].append(f"【资金】今日主力净流出 {abs(round(net_flow/10000))} 万元。")
            
        # 估值
        pe = self.spot_data['市盈率-动态']
        if 0 < pe < 20: 
            self.score += 5
            self.report['fundamental'].append(f"【估值】动态PE {pe}倍，处于低估区间。")
            
        # 风控 (ATR止损)
        atr = ta.atr(self.df['最高'], self.df['最低'], self.df['收盘'], length=14).iloc[-1]
        stop_loss = self.df['收盘'].iloc[-1] - 2 * atr
        self.report['risk'].append(f"【止损】建议止损价：{round(stop_loss, 2)} (2倍ATR)。")

    def generate_report(self):
        if not self.fetch_data(): return {"error": "获取数据失败，请检查代码"}
        
        # 执行所有分析模块
        self.detect_candlestick_patterns()
        self.analyze_special_morphology()
        self.analyze_chip_distribution()
        self.analyze_rps()
        self.analyze_basics()
        
        # 限制分数
        self.score = max(0, min(100, self.score))
        
        # 结论
        verdict = "观望 (Hold)"
        if self.score >= 80: verdict = "强力买入 (Strong Buy) 🔥"
        elif self.score >= 60: verdict = "谨慎增持 (Buy)"
        elif self.score <= 40: verdict = "卖出/规避 (Sell)"
        
        return {
            "name": self.spot_data['名称'],
            "price": self.spot_data['最新价'],
            "pct": self.spot_data['涨跌幅'],
            "score": int(self.score),
            "verdict": verdict,
            "report": self.report,
            "news": [{"title": n['新闻标题'], "date": n['发布时间'][5:16]} for _, n in self.news.iterrows()]
        }

@app.get("/analyze/{code}")
def analyze(code: str):
    trader = ProfessionalTrader(code)
    return trader.generate_report()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
