import numpy as np
import warnings
import sys
import random
import json
import urllib.request
import time

sys.setrecursionlimit(2000)
warnings.filterwarnings("ignore")

# ==========================================
# 1. DYNAMIC ASSET DISCOVERY (BINANCE)
# ==========================================

def fetch_top_volume_pairs(limit=25):
    """
    Dynamically fetches the top USDT spot pairs sorted by 24h trading volume.
    Excludes stablecoin pairs and wrapper tokens.
    """
    url = "https://api.binance.com/api/v3/ticker/24hr"
    excluded_symbols = {
        "USDCUSDT", "FDUSDUSDT", "USDTUSDT", "BUSDUSDT", 
        "TUSDUSDT", "EURUSDT", "WBTCUSDT", "DAIUSDT", "PYUSDUSDT"
    }
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            # Filter for USDT spot pairs
            usdt_pairs = [
                item for item in data 
                if item['symbol'].endswith('USDT') and item['symbol'] not in excluded_symbols
            ]
            # Sort by 24h USD Quote Volume descending
            usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            return [item['symbol'] for item in usdt_pairs[:limit]]
    except Exception as e:
        print(f"[Warning] Failed to fetch top pairs dynamically ({e}). Using default list.")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "PEPEUSDT"]


# ==========================================
# 2. PHYSICS INCLUSION KERNEL
# ==========================================

class InclusionDomain:
    def __init__(self, name, initial_val=0.5):
        self.name = name
        self.val = initial_val
        self.velocity = 0.0

    def update(self, force, dt):
        self.velocity = (self.velocity * 0.85) + (force * 0.15)
        self.val += self.velocity * dt
        self.val = float(np.clip(self.val, 0.0, 1.0))


# ==========================================
# 3. SUBSET SUM MOVEMENT PREDICTOR
# ==========================================

class SubsetMovementPredictor:
    def __init__(self):
        self.domains = {}

    def fetch_kline_data(self, symbol, days=150):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={days}"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                prices = [float(kline[4]) for kline in data] # Closing price
                return np.array(prices)
        except Exception as e:
            return None

    def prepare_subset_pool(self, prices, target_window=7):
        returns = np.diff(prices) / prices[:-1] * 100.0
        recent_movements = returns[-target_window:]
        target_sum = np.sum(recent_movements)
        history_pool = returns[:-target_window]
        return history_pool, target_sum

    def solve_subset_annealing(self, history_pool, target_sum, steps=1000):
        n = len(history_pool)
        self.domains = {i: InclusionDomain(f"day_{i}", initial_val=random.uniform(0.4, 0.6)) for i in range(n)}
        
        dt = 0.05
        perturbation = 0.01

        for step in range(steps):
            current_sum = sum(self.domains[i].val * history_pool[i] for i in range(n))
            error = abs(current_sum - target_sum)
            
            if error < 1e-3:
                break
                
            for i in range(n):
                domain = self.domains[i]
                orig = domain.val
                domain.val = min(1.0, orig + perturbation)
                sum_up = sum(self.domains[j].val * history_pool[j] for j in range(n))
                
                sensitivity = (sum_up - current_sum) / perturbation if perturbation > 0 else history_pool[i]
                domain.val = orig
                
                force = (target_sum - current_sum) * sensitivity * 0.01
                domain.update(force, dt)

        subset_indices = [i for i in range(n) if self.domains[i].val > 0.65]
        return subset_indices

    def predict_movement(self, history_pool, subset_indices, current_price):
        if not subset_indices:
            return current_price, 0.0
            
        next_day_returns = [history_pool[idx + 1] for idx in subset_indices if idx + 1 < len(history_pool)]
        if not next_day_returns:
            return current_price, 0.0
            
        predicted_return_pct = np.mean(next_day_returns)
        predicted_price = current_price * (1 + (predicted_return_pct / 100.0))
        return predicted_price, predicted_return_pct


# ==========================================
# 4. MARKET-WIDE RUNNER
# ==========================================

if __name__ == "__main__":
    # Change limit to 30, 50, or 100 to scale across the rest of the market
    TOP_N_COINS = 25
    
    print(f"\n[System] Fetching top {TOP_N_COINS} market pairs by trading volume...")
    symbols = fetch_top_volume_pairs(limit=TOP_N_COINS)
    
    predictor = SubsetMovementPredictor()
    results = []
    
    print("\n" + "="*80)
    print(f" RUNNING SUBSET SUM ANNEALING ACROSS TOP {len(symbols)} MARKET PAIRS")
    print("="*80)

    for idx, symbol in enumerate(symbols, start=1):
        clean_name = symbol.replace("USDT", "")
        print(f" [{idx:02d}/{len(symbols):02d}] Processing {clean_name:<10}...", end="\r")
        
        prices = predictor.fetch_kline_data(symbol=symbol, days=150)
        if prices is None or len(prices) < 10:
            continue
            
        current_price = prices[-1]
        history_pool, target_sum = predictor.prepare_subset_pool(prices, target_window=7)
        subset_indices = predictor.solve_subset_annealing(history_pool, target_sum, steps=1000)
        
        predicted_price, expected_move = predictor.predict_movement(history_pool, subset_indices, current_price)
        
        results.append({
            "symbol": clean_name,
            "price": current_price,
            "target_mom": target_sum,
            "subset_count": len(subset_indices),
            "projected_price": predicted_price,
            "expected_move": expected_move
        })
        
        time.sleep(0.05)

    print("\n" + " "*80) # Clear progress line
    
    # Sort results by highest projected expected move (%)
    results.sort(key=lambda x: x['expected_move'], reverse=True)

    # Output Market Leaderboard
    print("="*85)
    print(f"{'RANK & COIN':<15} | {'CURRENT PRICE':<14} | {'7D MOMENTUM':<12} | {'MATCHED DAYS':<12} | {'PROJECTED':<12} | {'FORECAST %':<8}")
    print("-" * 85)
    
    for rank, r in enumerate(results, start=1):
        dec = 6 if r['price'] < 0.01 else (4 if r['price'] < 1 else 2)
        p_str = f"${r['price']:,.{dec}f}"
        proj_str = f"${r['projected_price']:,.{dec}f}"
        coin_rank = f"#{rank:<2} {r['symbol']}"
        print(f"{coin_rank:<15} | {p_str:<14} | {r['target_mom']:>+11.2f}% | {r['subset_count']:^12} | {proj_str:<12} | {r['expected_move']:>+7.2f}%")
        
    print("="*85)
