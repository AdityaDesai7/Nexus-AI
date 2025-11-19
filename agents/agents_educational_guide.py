"""
agents_educational_guide.py
Complete educational guide for all 13 institutional agents
Shows data → logic → output for EACH agent
"""

class AgentEducationalGuide:
    """Educational guide for investors to understand each agent"""
    
    @staticmethod
    def get_all_agents_explanation():
        """Return detailed explanation for all 13 agents"""
        
        agents = {
            1: {
                "name": "🌍 FII Momentum Detector",
                "category": "Research-Backed Agent",
                "what_it_does": """
                Tracks Foreign Institutional Investor (FII) money flows.
                
                Why? Because FIIs are big money (₹ Lakh Crores) and when they buy/sell,
                the stock price MUST follow!
                """,
                "real_world_example": """
                Imagine 10 billionaires from USA/Europe suddenly start buying RELIANCE.
                Even if they buy quietly, their money is SO BIG that price will go up!
                """,
                "inputs": {
                    "FII Today": "How much money FIIs invested TODAY (in ₹ Crore)",
                    "FII 30-Day Average": "Average daily FII money in last 30 days",
                    "Price Change": "Did stock price go up or down?"
                },
                "logic": """
                IF FII Ratio (Today/Average) > 2.0 AND Price UP:
                    → Signal = "STRONG BUY" 
                    → Reason: BIG foreign money coming in + Price confirming!
                
                ELIF FII Ratio < 0.5:
                    → Signal = "SELL"
                    → Reason: Foreign money LEAVING = bad sign
                
                ELSE:
                    → Signal = "NEUTRAL"
                """,
                "output_example": """
                Input: FII Today = ₹500 Cr, FII 30-Avg = ₹200 Cr
                Ratio = 500/200 = 2.5x
                Output: ✅ BULLISH (2.5x is > 2.0)
                Confidence: 85%
                """
            },
            
            2: {
                "name": "🔨 Execution Breakdown Detector",
                "category": "Research-Backed Agent",
                "what_it_does": """
                Detects when BIG PLAYERS split huge orders into many small pieces.
                
                Why? Because if you want to buy ₹100 Crore of stock secretly,
                you don't do it in ONE order - everyone will see!
                You break it into 50 orders of ₹2 Crore each.
                """,
                "real_world_example": """
                A mutual fund wants to accumulate RELIANCE quietly.
                Instead of 1 order of ₹500 Cr (which raises red flags),
                they place 100 orders of ₹5 Cr each spread across the day.
                """,
                "inputs": {
                    "Recent Orders": "Last 60 minutes of orders from NSE",
                    "Order Sizes": "Are orders similar sizes?",
                    "Order Timing": "Are they at similar prices/times?"
                },
                "logic": """
                IF 20+ small orders appear at SAME PRICE within 60 minutes:
                    → Signal = "ACCUMULATION DETECTED"
                    → Reason: Looks like institutional buying in pieces!
                
                ELIF Pattern of consistent SELLING orders:
                    → Signal = "DISTRIBUTION DETECTED"
                    → Reason: Institution selling quietly!
                """,
                "output_example": """
                Observed: 25 orders of ₹3-4Cr each at ₹2500 between 10:00-11:00 AM
                Output: ✅ ACCUMULATION PHASE DETECTED
                Action: BULLISH - Big player accumulating
                """
            },
            
            3: {
                "name": "📊 Volume Pattern Recognition",
                "category": "Research-Backed Agent",
                "what_it_does": """
                Analyzes VOLUME trends. This is KEY because:
                
                Research shows: When volume INCREASES, price change is PERMANENT
                When volume DECREASES, price change is just NOISE
                """,
                "real_world_example": """
                Scenario 1: Price up 5% on NORMAL volume = Just noise, will go back
                Scenario 2: Price up 5% on 5x NORMAL volume = REAL buying, will stay up!
                """,
                "inputs": {
                    "Last 30 Days Volume": "Average volume in previous 30 days",
                    "Last 5 Days Volume": "Recent volume trend",
                    "Price Movement": "Is price going up or down?"
                },
                "logic": """
                IF Volume INCREASING + Price INCREASING:
                    → Signal = "ACCUMULATION"
                    → Reason: Big volume + price up = institutions buying!
                
                ELIF Volume DECREASING + Price DECREASING:
                    → Signal = "DISTRIBUTION"
                    → Reason: Big volume + price down = institutions selling!
                """,
                "output_example": """
                30-Day Avg Volume: 1M shares
                Recent 5-Day Avg: 5M shares (5x increase!)
                Price Change: +8%
                Output: ✅ STRONG ACCUMULATION
                Confidence: 92%
                """
            },
            
            4: {
                "name": "📈 LOFS Strategy (Long Only Filter)",
                "category": "Research-Backed Agent",
                "what_it_does": """
                Simple but POWERFUL strategy from academic research:
                
                "Buy when price > 20-day average"
                
                Why? Because this identifies uptrends. When price breaks above
                20-day average = TREND CONFIRMED!
                """,
                "real_world_example": """
                If ₹2000 was the average for 20 days, and today price = ₹2100:
                → Price is above average = UPTREND = BUY SIGNAL
                
                If price = ₹1900:
                → Price is below average = DOWNTREND = SELL SIGNAL
                """,
                "inputs": {
                    "Current Price": "Today's closing price",
                    "20-Day Moving Average": "Average of last 20 days prices"
                },
                "logic": """
                IF Current Price > 20-Day Moving Average:
                    → Signal = "BUY"
                    → Reason: Confirmed uptrend!
                
                ELIF Current Price < 20-Day Moving Average:
                    → Signal = "SELL"
                    → Reason: Confirmed downtrend or sideways!
                """,
                "output_example": """
                Current Price: ₹2500
                20-Day Average: ₹2300
                Output: ✅ BUY SIGNAL
                Reason: Price is ₹200 ABOVE average = strong uptrend
                """
            },
            
            5: {
                "name": "🎛️ Master Institutional Aggregator",
                "category": "Research-Backed Agent",
                "what_it_does": """
                Combines ALL 12 OTHER AGENTS into ONE final score.
                
                Like a teacher averaging marks from 12 different subjects
                into ONE final grade!
                """,
                "real_world_example": """
                Agent 1 says: BUY (score 70)
                Agent 2 says: SELL (score 30)
                Agent 3 says: BUY (score 80)
                ...and 9 more
                
                Master = Average of all = FINAL SCORE
                """,
                "inputs": {
                    "All 12 Agent Scores": "Individual signals from agents 1-12"
                },
                "logic": """
                Final Score = (Agent1 + Agent2 + ... + Agent12) / 12
                
                IF Score > 75:
                    → Signal = "STRONG BUY"
                
                ELIF Score > 50:
                    → Signal = "BUY"
                
                ELIF Score < 25:
                    → Signal = "STRONG SELL"
                
                ELSE:
                    → Signal = "HOLD"
                """,
                "output_example": """
                Agent scores: 75, 60, 80, 55, 70, 65, 72, 58, 68, 71, 77, 62
                Average = 68.75
                Output: ✅ MODERATE BUY SIGNAL
                Confidence: 68.75%
                """
            },
            
            6: {
                "name": "💧 Liquidity Detector",
                "category": "Detection Agent",
                "what_it_does": """
                Checks: Can you BUY/SELL this stock easily without moving price?
                
                Like checking if a shop has stock available or if it's empty!
                """,
                "real_world_example": """
                High Liquidity: RELIANCE - 50M shares traded daily (EASY to buy/sell)
                Low Liquidity: Some small stock - 100K shares daily (HARD to buy/sell)
                """,
                "inputs": {
                    "Average Daily Volume": "How many shares traded per day on average?"
                },
                "logic": """
                IF Average Daily Volume > 1 Million:
                    → Signal = "EXCELLENT LIQUIDITY"
                
                ELIF Average Daily Volume > 500K:
                    → Signal = "GOOD LIQUIDITY"
                
                ELSE:
                    → Signal = "LOW LIQUIDITY - RISKY"
                """,
                "output_example": """
                Average Daily Volume: 3M shares
                Output: ✅ EXCELLENT LIQUIDITY
                Meaning: You can easily buy or sell thousands of shares!
                """
            },
            
            7: {
                "name": "👥 Smart Money Tracker",
                "category": "Detection Agent",
                "what_it_does": """
                Tracks holdings of SMART MONEY (Mutual Funds, Insurance Companies)
                
                If SMART MONEY is buying → Good sign
                If SMART MONEY is selling → Bad sign
                """,
                "real_world_example": """
                Mutual funds hold 5% of RELIANCE shares
                Next quarter = They increase to 7% holding
                → They're BUYING = Bullish signal!
                """,
                "inputs": {
                    "Previous Holdings %": "What % did they own last quarter?",
                    "Current Holdings %": "What % do they own now?"
                },
                "logic": """
                IF Current Holdings > Previous Holdings:
                    → Signal = "BUYING" (BULLISH)
                
                ELIF Current Holdings < Previous Holdings:
                    → Signal = "SELLING" (BEARISH)
                """,
                "output_example": """
                Previous: 5% holdings
                Current: 7% holdings
                Output: ✅ BULLISH - Smart money increased by 2%!
                Confidence: 80%
                """
            },
            
            8: {
                "name": "🧊 Iceberg Detector",
                "category": "Detection Agent",
                "what_it_does": """
                Finds HIDDEN large orders in NSE order book.
                
                Like detecting an iceberg - you only see tip, 90% is hidden!
                Institutions hide orders to avoid moving price.
                """,
                "real_world_example": """
                Visible: 1M share buy order at ₹2500
                Behind scenes: Actually wants to buy 10M shares!
                Shows small amounts, then immediately re-orders.
                """,
                "inputs": {
                    "Order Book Data": "Real-time NSE order book (L2 data)",
                    "Order Patterns": "Are similar orders repeating?"
                },
                "logic": """
                IF Small orders repeatedly placed at same price:
                    → Signal = "ICEBERG ORDER DETECTED"
                    → Reason: Hiding large institutional order!
                """,
                "output_example": """
                Observed: 100K share orders repeating at ₹2500 every minute
                Total visible = 100K, but pattern suggests 5M hidden!
                Output: ⚠️ ICEBERG ORDER DETECTED (Bullish)
                """
            },
            
            9: {
                "name": "📋 Block Order Tracker",
                "category": "Detection Agent",
                "what_it_does": """
                Tracks LARGE BLOCK TRADES (5+ Lakh shares at once)
                
                When someone buys/sells 5L+ shares = Usually institutional = Important!
                """,
                "real_world_example": """
                Block deal: 10L RELIANCE shares traded at ₹2500
                = Some institution just bought ₹25 Crore worth!
                This is SIGNIFICANT signal!
                """,
                "inputs": {
                    "NSE Block Deals": "Daily block trade data from NSE"
                },
                "logic": """
                IF Block BUY deals > Block SELL deals today:
                    → Signal = "ACCUMULATION BY INSTITUTIONS"
                
                ELIF Block SELL deals > Block BUY deals:
                    → Signal = "DISTRIBUTION BY INSTITUTIONS"
                """,
                "output_example": """
                Today's blocks: 3 buy blocks (25L shares), 1 sell block (8L shares)
                Output: ✅ NET BUYING - Institutions accumulating!
                """
            },
            
            10: {
                "name": "📈 Accumulation Detector",
                "category": "Detection Agent",
                "what_it_does": """
                Detects when stock is near SUPPORT level.
                
                Support = Floor price where buyers step in
                When price near support + institutional activity = Big opportunity!
                """,
                "real_world_example": """
                Support at ₹2300
                Current price: ₹2320
                + High FII buying
                = Institutions accumulating near support = BULLISH!
                """,
                "inputs": {
                    "Support Level": "Lowest price in last 20 days",
                    "Current Price": "Today's price",
                    "Volume": "Is there institutional volume?"
                },
                "logic": """
                IF Price close to Support (< 2%) + High Volume:
                    → Signal = "ACCUMULATION PHASE"
                    → Reason: Institutions buying near floor!
                """,
                "output_example": """
                Support: ₹2300
                Current: ₹2310 (only ₹10 away = 0.43%)
                Volume: 5x normal
                Output: ✅ STRONG ACCUMULATION - Institutional buying!
                """
            },
            
            11: {
                "name": "📉 Distribution Detector",
                "category": "Detection Agent",
                "what_it_does": """
                OPPOSITE of accumulation.
                Detects when stock near RESISTANCE level
                + institutions selling = Top forming = Bearish!
                """,
                "real_world_example": """
                Resistance at ₹2500
                Current price: ₹2490
                + High institutional selling
                = Institutions distributing at top = BEARISH!
                """,
                "inputs": {
                    "Resistance Level": "Highest price in last 20 days",
                    "Current Price": "Today's price",
                    "Volume": "Is there institutional selling volume?"
                },
                "logic": """
                IF Price close to Resistance (< 2%) + High Selling Volume:
                    → Signal = "DISTRIBUTION PHASE"
                    → Reason: Institutions selling near ceiling!
                """,
                "output_example": """
                Resistance: ₹2500
                Current: ₹2495 (only ₹5 away = 0.2%)
                Volume: 5x normal (SELLING)
                Output: ⚠️ DISTRIBUTION - Institutions exiting!
                """
            },
            
            12: {
                "name": "🚀 Breakout Detector",
                "category": "Detection Agent",
                "what_it_does": """
                Detects EXPLOSIVE price moves with HIGH VOLUME.
                
                When price breaks resistance/support with 5x volume = BREAKOUT!
                = Institutions pushing price through a level!
                """,
                "real_world_example": """
                Resistance at ₹2500
                Today: Price crosses ₹2500 + Volume = 5M (vs normal 1M)
                = BREAKOUT! Price will likely continue up!
                """,
                "inputs": {
                    "Current Volume": "Today's volume",
                    "Average Volume": "Normal daily volume",
                    "Price vs Resistance": "Did price break out?"
                },
                "logic": """
                IF Current Volume > 5x Average Volume:
                    AND Price breaks Resistance/Support:
                    → Signal = "BREAKOUT DETECTED"
                    → Reason: High volume confirms breakout is real!
                """,
                "output_example": """
                Normal Volume: 1M shares
                Today: 5.2M shares
                Price: Broke ₹2500 resistance
                Output: 🚀 STRONG BREAKOUT!
                Prediction: Price will continue upward!
                """
            },
            
            13: {
                "name": "🔢 IFI Calculator (Institutional Footprint)",
                "category": "Detection Agent",
                "what_it_does": """
                Calculates INSTITUTIONAL PARTICIPATION INDEX
                
                Measures: What % of today's trading is by institutions vs retail?
                High IFI = Institutions very active = Strong signal!
                """,
                "real_world_example": """
                IFI = 8.5
                Meaning: Today's trading has EXTREME institutional activity
                vs IFI = 0.5 = Just retail trading, no institution
                """,
                "inputs": {
                    "Volume Ratio": "Today's volume / Average volume",
                    "Price Change": "How much did price move?",
                    "Volatility": "Is volatility high?"
                },
                "logic": """
                IFI Score = (Volume Ratio × |Price Change|) / 2
                
                IF IFI > 3:
                    → "EXTREME INSTITUTIONAL ACTIVITY"
                
                ELIF IFI > 1:
                    → "MODERATE INSTITUTIONAL ACTIVITY"
                
                ELSE:
                    → "LOW INSTITUTIONAL ACTIVITY"
                """,
                "output_example": """
                Volume Ratio: 4.5x
                Price Change: 3.2%
                IFI = (4.5 × 3.2) / 2 = 7.2
                Output: 🔥 EXTREME INSTITUTIONAL ACTIVITY!
                Meaning: Huge institutional players very active today!
                """
            }
        }
        
        return agents
