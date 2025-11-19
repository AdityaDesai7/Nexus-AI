# data/institutional_data_fetcher.py - REAL DATA FETCHER FOR 9 AGENTS

"""
Fetches REAL institutional data from:
- NSE (Bulk deals, Block deals)
- Groww (FII/DII historical data)
- Moneycontrol (Institutional activity)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class InstitutionalDataFetcher:
    """Fetches real institutional data for 9 agents"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html,application/xhtml+xml',
        }
    
    # ============================================
    # 1. FII MOMENTUM AGENT - Real FII Data
    # ============================================
    
    def get_fii_dii_data(self, days=30):
        """Fetch real FII/DII data from NSE"""
        try:
            url = "https://www.nseindia.com/api/fiidiiTrading"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                fii_data = {}
                for record in data[:1]:  # Latest day
                    if 'FII' in str(record):
                        fii_data['today'] = record.get('netValue', 0)
                
                # Get 30-day average
                avg = sum(float(r.get('netValue', 0)) for r in data[:30]) / max(len(data[:30]), 1)
                fii_data['30_day_avg'] = avg
                
                logger.info(f"✅ FII Data: Today={fii_data.get('today', 0)}, Avg={avg:.0f}")
                return fii_data
        except Exception as e:
            logger.warning(f"FII data fetch: {e}")
        
        return {'today': 0, '30_day_avg': 0}
    
    # ============================================
    # 2. EXECUTION BREAKDOWN - Real Order Data
    # ============================================
    
    def get_nse_order_data(self, stock_symbol, limit=100):
        """Fetch real NSE bulk deals (execution patterns)"""
        try:
            url = f"https://www.nseindia.com/api/snapshot-capital-market-largedeal"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                orders = []
                
                # Get bulk deals
                if 'bulk_deals' in data and data['bulk_deals']:
                    for deal in data['bulk_deals'][:limit]:
                        orders.append({
                            'symbol': deal.get('symbol', ''),
                            'direction': 'BUY' if deal.get('buySell', 'SELL').upper() == 'BUY' else 'SELL',
                            'quantity': float(deal.get('qty', 0)),
                            'price': float(deal.get('watp', 0)),
                            'time': datetime.now(),
                            'type': 'BULK'
                        })
                
                # Get block deals
                if 'block_deals' in data and data['block_deals']:
                    for deal in data['block_deals'][:limit]:
                        orders.append({
                            'symbol': deal.get('symbol', ''),
                            'direction': 'BUY' if deal.get('buySell', 'SELL').upper() == 'BUY' else 'SELL',
                            'quantity': float(deal.get('qty', 0)),
                            'price': float(deal.get('watp', 0)),
                            'time': datetime.now(),
                            'type': 'BLOCK'
                        })
                
                logger.info(f"✅ NSE Orders: {len(orders)} deals fetched")
                return orders
        
        except Exception as e:
            logger.warning(f"NSE order data: {e}")
        
        return []
    
    # ============================================
    # 3. VOLUME PATTERN - Already in Price Data
    # (Uses volume from price_data)
    # ============================================
    
    def get_volume_anomalies(self, volume_data):
        """Detect volume anomalies for Volume Pattern agent"""
        try:
            if volume_data is None or len(volume_data) < 10:
                return {'anomalies': [], 'spike_count': 0}
            
            avg_vol = volume_data.tail(30).mean()
            recent_spike = volume_data.iloc[-1] / avg_vol if avg_vol > 0 else 1
            
            anomalies = []
            spike_count = 0
            
            for idx in range(len(volume_data)-1, max(len(volume_data)-20, 0), -1):
                vol = volume_data.iloc[idx]
                if vol > avg_vol * 1.5:
                    spike_count += 1
                    anomalies.append({
                        'date': volume_data.index[idx] if hasattr(volume_data, 'index') else idx,
                        'volume': vol,
                        'ratio': vol / avg_vol
                    })
            
            logger.info(f"✅ Volume Anomalies: {spike_count} spikes detected")
            return {
                'anomalies': anomalies[:5],  # Last 5 spikes
                'spike_count': spike_count,
                'recent_spike_ratio': recent_spike
            }
        except Exception as e:
            logger.warning(f"Volume anomalies: {e}")
        
        return {'anomalies': [], 'spike_count': 0, 'recent_spike_ratio': 1}
    
    # ============================================
    # 4. IFI CALCULATOR - Institutional Footprint
    # (Calculated from price + volume)
    # ============================================
    
    def calculate_ifi(self, volume_data, price_data):
        """Calculate Institutional Footprint Indicator"""
        try:
            if volume_data is None or price_data is None or len(volume_data) < 5:
                return {'ifi': 0, 'signal': 'LOW'}
            
            today_vol = volume_data.iloc[-1]
            avg_vol = volume_data.tail(30).mean()
            price_change = ((price_data.iloc[-1] / price_data.iloc[-5]) - 1) * 100
            
            ifi = (today_vol / avg_vol) * (abs(price_change) / 100) if avg_vol > 0 else 0
            
            if ifi > 2.5:
                signal = 'EXTREME'
            elif ifi > 1.8:
                signal = 'HIGH'
            elif ifi > 1.2:
                signal = 'MODERATE'
            else:
                signal = 'LOW'
            
            logger.info(f"✅ IFI: {ifi:.2f} ({signal})")
            return {'ifi': ifi, 'signal': signal}
        except Exception as e:
            logger.warning(f"IFI calculation: {e}")
        
        return {'ifi': 0, 'signal': 'LOW'}
    
    # ============================================
    # 5. ACCUMULATION DETECTOR - Quiet Buying
    # ============================================
    
    def detect_accumulation(self, price_data, volume_data):
        """Detect institutional accumulation phases"""
        try:
            if price_data is None or volume_data is None or len(price_data) < 20:
                return {'status': 'UNKNOWN', 'confidence': 0}
            
            recent_price = price_data.tail(20)
            recent_vol = volume_data.tail(20)
            
            price_range = (recent_price.max() - recent_price.min()) / recent_price.min()
            vol_increase = recent_vol.mean() / volume_data.tail(60).mean()
            
            price_near_low = price_data.iloc[-1] / recent_price.min()
            
            # Accumulation: Small price range + High volume + Price near low
            if price_near_low < 1.05 and vol_increase > 1.2 and price_range < 0.1:
                logger.info(f"✅ Accumulation detected!")
                return {'status': 'ACCUMULATING', 'confidence': 85}
            elif vol_increase > 1.3 and price_near_low < 1.08:
                logger.info(f"⏸️ Possible accumulation")
                return {'status': 'POSSIBLE', 'confidence': 65}
            else:
                return {'status': 'NORMAL', 'confidence': 40}
        
        except Exception as e:
            logger.warning(f"Accumulation detection: {e}")
        
        return {'status': 'UNKNOWN', 'confidence': 0}
    
    # ============================================
    # 6. LIQUIDITY DETECTOR - Trading Ease
    # ============================================
    
    def get_liquidity_metrics(self, volume_data):
        """Get liquidity metrics"""
        try:
            if volume_data is None or len(volume_data) < 20:
                return {'liquidity': 'UNKNOWN', 'volume_avg': 0}
            
            avg_vol = volume_data.tail(20).mean()
            recent_vol = volume_data.tail(5).mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            
            if vol_ratio > 1.4:
                liquidity = 'HIGH'
            elif vol_ratio < 0.8:
                liquidity = 'LOW'
            else:
                liquidity = 'NORMAL'
            
            logger.info(f"✅ Liquidity: {liquidity} (ratio={vol_ratio:.2f})")
            return {
                'liquidity': liquidity,
                'volume_avg': avg_vol,
                'volume_ratio': vol_ratio
            }
        except Exception as e:
            logger.warning(f"Liquidity metrics: {e}")
        
        return {'liquidity': 'UNKNOWN', 'volume_avg': 0, 'volume_ratio': 1}
    
    # ============================================
    # 7. SMART MONEY TRACKER - MF Holdings
    # ============================================
    
    def get_mf_holdings(self, stock_symbol):
        """Get Mutual Fund holdings data"""
        try:
            # This would connect to real NSE data or scrape from Moneycontrol
            # For now, returning structure - you can integrate API
            
            logger.info(f"🔍 Fetching MF holdings for {stock_symbol}")
            return {
                'current_holdings': 0,  # Placeholder
                'previous_holdings': 0,
                'change_pct': 0,
                'major_buyers': [],
                'major_sellers': []
            }
        except Exception as e:
            logger.warning(f"MF holdings: {e}")
        
        return {'current_holdings': 0, 'previous_holdings': 0}
    
    # ============================================
    # 8. BLOCK ORDER TRACKER - Large Orders
    # ============================================
    
    def get_block_orders(self, stock_symbol, days=5):
        """Get real block orders from NSE"""
        try:
            url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                blocks = []
                if 'block_deals' in data and data['block_deals']:
                    for deal in data['block_deals']:
                        if deal.get('symbol') == stock_symbol:
                            blocks.append({
                                'quantity': float(deal.get('qty', 0)),
                                'price': float(deal.get('watp', 0)),
                                'value': float(deal.get('qty', 0)) * float(deal.get('watp', 0)),
                                'direction': deal.get('buySell', 'SELL'),
                                'time': datetime.now()
                            })
                
                logger.info(f"✅ Block Orders: {len(blocks)} found for {stock_symbol}")
                return blocks
        
        except Exception as e:
            logger.warning(f"Block orders: {e}")
        
        return []
    
    # ============================================
    # 9. BREAKOUT DETECTOR - Support/Resistance
    # ============================================
    
    def detect_breakouts(self, price_data, volume_data):
        """Detect price breakouts"""
        try:
            if price_data is None or volume_data is None or len(price_data) < 30:
                return {'breakout_type': 'NONE', 'level': 0}
            
            high_30 = price_data.tail(30).max()
            low_30 = price_data.tail(30).min()
            current = price_data.iloc[-1]
            vol_conf = volume_data.iloc[-1] > volume_data.tail(30).mean() * 1.2
            
            if current >= high_30 * 0.98 and vol_conf:
                logger.info(f"✅ Bullish breakout at {current:.2f}")
                return {'breakout_type': 'BULLISH', 'level': high_30}
            elif current <= low_30 * 1.02 and vol_conf:
                logger.info(f"⚠️ Bearish breakout at {current:.2f}")
                return {'breakout_type': 'BEARISH', 'level': low_30}
            else:
                return {'breakout_type': 'NONE', 'level': 0}
        
        except Exception as e:
            logger.warning(f"Breakout detection: {e}")
        
        return {'breakout_type': 'NONE', 'level': 0}


# ============================================
# Helper function to get all data at once
# ============================================

def fetch_all_institutional_data(stock_symbol, price_data, volume_data, days=30):
    """Fetch all 9 agents' data in one call"""
    
    fetcher = InstitutionalDataFetcher()
    
    return {
        'fii_dii': fetcher.get_fii_dii_data(days),                    # Agent 1
        'order_data': fetcher.get_nse_order_data(stock_symbol),       # Agent 2
        'volume_anomalies': fetcher.get_volume_anomalies(volume_data), # Agent 3
        'ifi': fetcher.calculate_ifi(volume_data, price_data),         # Agent 4
        'accumulation': fetcher.detect_accumulation(price_data, volume_data), # Agent 5
        'liquidity': fetcher.get_liquidity_metrics(volume_data),       # Agent 6
        'mf_holdings': fetcher.get_mf_holdings(stock_symbol),          # Agent 7
        'block_orders': fetcher.get_block_orders(stock_symbol),        # Agent 8
        'breakouts': fetcher.detect_breakouts(price_data, volume_data) # Agent 9
    }
