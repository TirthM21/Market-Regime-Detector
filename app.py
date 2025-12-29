import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import requests
import json
import yfinance as yf
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")


class DataSource:
    """Unified data source with yfinance and custom API support"""
    
    def __init__(self, use_yfinance=True, api_url=None):
        self.use_yfinance = use_yfinance
        self.api_url = api_url.rstrip('/') if api_url else None
        self.last_price = None
        self.last_fetch = None
        self.ticker = None
        
    def connect(self):
        """Test connection to data source"""
        if self.use_yfinance:
            try:
                test = yf.Ticker("RELIANCE.NS")
                info = test.info
                return True
            except:
                return False
        else:
            if not self.api_url:
                return False
            try:
                response = requests.get(f"{self.api_url}/", timeout=5)
                return response.status_code == 200
            except:
                return False
    
    def set_symbol(self, symbol):
        """Set the symbol to track"""
        if self.use_yfinance:
            self.ticker = yf.Ticker(symbol)
        
    def get_current_price(self, symbol):
        """Get current price from data source"""
        try:
            if self.use_yfinance:
                if not self.ticker or self.ticker.ticker != symbol:
                    self.ticker = yf.Ticker(symbol)
                
                # Get latest data
                data = self.ticker.history(period='1d', interval='1m')
                if not data.empty:
                    price = float(data['Close'].iloc[-1])
                    volume = int(data['Volume'].iloc[-1]) if 'Volume' in data else 0
                    self.last_price = price
                    self.last_fetch = datetime.now()
                    return {
                        'price': price,
                        'volume': volume,
                        'timestamp': datetime.now(),
                        'high': float(data['High'].iloc[-1]),
                        'low': float(data['Low'].iloc[-1]),
                        'open': float(data['Open'].iloc[-1])
                    }
            else:
                # Use custom API
                if not symbol.endswith(('.NS', '.BO')):
                    symbol = f"{symbol}.NS"
                
                response = requests.get(
                    f"{self.api_url}/stock",
                    params={"symbol": symbol, "res": "num"},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        stock_data = data.get("data", {})
                        price = stock_data.get("last_price")
                        
                        if isinstance(price, dict):
                            price = price.get("value", 0)
                        
                        if price and price > 0:
                            self.last_price = price
                            self.last_fetch = datetime.now()
                            return {
                                'price': float(price),
                                'volume': stock_data.get('volume', 0),
                                'timestamp': datetime.now(),
                                'high': float(stock_data.get('high', price)),
                                'low': float(stock_data.get('low', price)),
                                'open': float(stock_data.get('open', price))
                            }
        except Exception as e:
            print(f"Data fetch error: {e}")
        
        return None


class OHLCBar:
    """OHLC candlestick bar"""
    
    def __init__(self, timestamp, open_price):
        self.timestamp = timestamp
        self.open = open_price
        self.high = open_price
        self.low = open_price
        self.close = open_price
        self.volume = 0
        self.tick_count = 1
        self.regime = 1

    def update(self, price, volume=0):
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.tick_count += 1

    @property
    def range_pct(self):
        """High-Low range as percentage of close"""
        if self.close > 0:
            return (self.high - self.low) / self.close
        return 0

    @property
    def body_pct(self):
        """Body size as percentage"""
        if self.open > 0:
            return abs(self.close - self.open) / self.open
        return 0
    
    @property
    def return_pct(self):
        """Return percentage"""
        if self.open > 0:
            return (self.close - self.open) / self.open
        return 0


class RobustRegimeDetector:
    """
    Robust regime detection using multiple signals:
    1. Volatility (range-based)
    2. Price momentum (absolute returns)
    3. Trend strength
    4. Z-score normalization for adaptive thresholds
    """
    
    REGIME_NAMES = ['CALM', 'ACTIVE', 'VOLATILE']
    REGIME_COLORS = ['#3fb950', '#58a6ff', '#f85149']
    REGIME_BG = ['#1a3d1a', '#1a2d3d', '#3d1a1a']
    
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.current_regime = 1
        
        # Buffers for features
        self.volatilities = deque(maxlen=lookback)
        self.returns = deque(maxlen=lookback)
        self.body_sizes = deque(maxlen=lookback)
        self.regime_history = deque(maxlen=100)
        
        # Statistics
        self.regime_switches = 0
        self.last_switch_time = None
        
        # Adaptive thresholds (z-scores)
        self.vol_z_thresholds = [-0.5, 0.5]  # Z-score thresholds
        self.calibrated = False
        self.calibration_count = 0
        
        # Running statistics for normalization
        self.vol_mean = 0.001
        self.vol_std = 0.0005
        
    def _update_running_stats(self, vols):
        """Update running mean and std for normalization"""
        if len(vols) < 5:
            return
        
        vols_arr = np.array(vols)
        # Use robust estimators
        self.vol_mean = np.median(vols_arr)
        self.vol_std = np.std(vols_arr)
        
        # Ensure minimum std to avoid division by zero
        if self.vol_std < 1e-6:
            self.vol_std = 1e-6
    
    def calibrate(self, bars):
        """Calibrate detector using historical bars"""
        if len(bars) < 12:
            return
        
        # Extract features
        vols = []
        rets = []
        bodies = []
        
        for bar in bars:
            if bar.close > 0:
                vols.append(bar.range_pct)
                rets.append(abs(bar.return_pct))
                bodies.append(bar.body_pct)
        
        if len(vols) < 12:
            return
        
        vols = np.array(vols)
        rets = np.array(rets)
        bodies = np.array(bodies)
        
        # Update running statistics
        self._update_running_stats(vols)
        
        # Calculate composite volatility score
        vol_normalized = (vols - np.median(vols)) / (np.std(vols) + 1e-6)
        ret_normalized = (rets - np.median(rets)) / (np.std(rets) + 1e-6)
        
        # Composite score (volatility + momentum)
        composite_scores = 0.7 * vol_normalized + 0.3 * ret_normalized
        
        # Use percentiles on composite score for thresholds
        p30 = np.percentile(composite_scores, 30)
        p70 = np.percentile(composite_scores, 70)
        
        # Set z-score thresholds with safety checks
        self.vol_z_thresholds[0] = max(p30, -1.5)  # Lower bound
        self.vol_z_thresholds[1] = min(p70, 1.5)   # Upper bound
        
        # Ensure separation
        if self.vol_z_thresholds[1] - self.vol_z_thresholds[0] < 0.3:
            mid = (self.vol_z_thresholds[0] + self.vol_z_thresholds[1]) / 2
            self.vol_z_thresholds[0] = mid - 0.3
            self.vol_z_thresholds[1] = mid + 0.3
        
        self.calibrated = True
        self.calibration_count += 1
        
        print(f"Calibration #{self.calibration_count}:")
        print(f"  Vol mean={self.vol_mean:.5f}, std={self.vol_std:.5f}")
        print(f"  Z-thresholds: low={self.vol_z_thresholds[0]:.2f}, high={self.vol_z_thresholds[1]:.2f}")
    
    def detect_regime(self, bar):
        """Detect market regime for current bar"""
        
        # Update buffers
        if bar.close > 0:
            self.volatilities.append(bar.range_pct)
            self.returns.append(abs(bar.return_pct))
            self.body_sizes.append(bar.body_pct)
        
        # Need minimum data
        if len(self.volatilities) < 5:
            bar.regime = 1
            return 1
        
        # Update running statistics
        self._update_running_stats(list(self.volatilities))
        
        # Calculate features for current bar
        current_vol = bar.range_pct
        current_ret = abs(bar.return_pct)
        
        # Normalize using z-scores
        vol_z = (current_vol - self.vol_mean) / self.vol_std
        
        # Get recent momentum signal
        recent_rets = list(self.returns)[-5:]
        ret_mean = np.mean(recent_rets)
        ret_z = (current_ret - ret_mean) / (np.std(recent_rets) + 1e-6)
        
        # Composite score
        composite_z = 0.7 * vol_z + 0.3 * ret_z
        
        # Regime classification based on composite z-score
        if composite_z < self.vol_z_thresholds[0]:
            raw_regime = 0  # CALM
        elif composite_z < self.vol_z_thresholds[1]:
            raw_regime = 1  # ACTIVE
        else:
            raw_regime = 2  # VOLATILE
        
        # Add hysteresis to prevent rapid switching
        if len(self.regime_history) >= 2:
            recent_regimes = list(self.regime_history)[-2:]
            
            # If recently switched, be more conservative
            if len(set(recent_regimes)) > 1:
                # Require stronger evidence to switch again
                margin = 0.2
                if raw_regime > self.current_regime:
                    if composite_z < self.vol_z_thresholds[raw_regime - 1] + margin:
                        raw_regime = self.current_regime
                elif raw_regime < self.current_regime:
                    if composite_z > self.vol_z_thresholds[raw_regime] - margin:
                        raw_regime = self.current_regime
        
        # Update state
        if raw_regime != self.current_regime:
            self.regime_switches += 1
            self.last_switch_time = datetime.now()
            self.current_regime = raw_regime
        
        self.regime_history.append(self.current_regime)
        bar.regime = self.current_regime
        
        return self.current_regime
    
    def get_statistics(self):
        """Get regime statistics"""
        if len(self.regime_history) < 5:
            return {}
        
        history = list(self.regime_history)
        total = len(history)
        
        # Calculate percentages
        stats = {
            'calm_pct': (history.count(0) / total) * 100,
            'active_pct': (history.count(1) / total) * 100,
            'volatile_pct': (history.count(2) / total) * 100,
            'switches': self.regime_switches,
            'current_duration': 1
        }
        
        # Calculate current regime duration
        if history:
            current = history[-1]
            duration = 0
            for r in reversed(history):
                if r == current:
                    duration += 1
                else:
                    break
            stats['current_duration'] = duration
        
        # Average volatility per regime
        if self.volatilities:
            vols = list(self.volatilities)
            regimes = list(self.regime_history)[-len(vols):]
            
            for regime_id in [0, 1, 2]:
                regime_vols = [v for v, r in zip(vols, regimes) if r == regime_id]
                if regime_vols:
                    stats[f'avg_vol_{regime_id}'] = np.mean(regime_vols)
        
        # Add normalization stats
        stats['vol_mean'] = self.vol_mean
        stats['vol_std'] = self.vol_std
        
        return stats


class MarketDashboard:
    """Main dashboard application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Stock Regime Detection Dashboard v2.0")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0d1117')
        
        # Setup theme
        self.setup_theme()
        
        # Data queue for thread-safe communication
        self.data_queue = queue.Queue()
        
        # State
        self.data_source = None
        self.connected = False
        self.streaming = False
        self.running = False
        
        # Trading data
        self.current_symbol = None
        self.bar_duration = 5  # seconds
        self.max_bars = 20
        self.ohlc_bars = deque(maxlen=self.max_bars)
        self.current_bar = None
        self.bar_start_time = None
        
        # Regime detector
        self.regime_detector = RobustRegimeDetector(lookback=30)
        
        # UI Setup
        self.setup_ui()
        self.setup_chart()
        
        # Start queue processor
        self.process_queue()
        
    def setup_theme(self):
        """Setup dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        bg = '#0d1117'
        fg = '#c9d1d9'
        accent = '#238636'
        
        style.configure('TFrame', background=bg)
        style.configure('TLabelframe', background=bg, foreground=fg, borderwidth=2)
        style.configure('TLabelframe.Label', background=bg, foreground=fg, font=('Segoe UI', 10, 'bold'))
        style.configure('TLabel', background=bg, foreground=fg, font=('Segoe UI', 10))
        style.configure('TButton', background=accent, foreground='white', font=('Segoe UI', 9, 'bold'), padding=6)
        style.map('TButton', background=[('active', '#2ea043')])
        style.configure('Stop.TButton', background='#da3633', foreground='white')
        style.map('Stop.TButton', background=[('active', '#f85149')])
        style.configure('TCheckbutton', background=bg, foreground=fg, font=('Segoe UI', 9))
        style.configure('TRadiobutton', background=bg, foreground=fg, font=('Segoe UI', 9))
        
    def setup_ui(self):
        """Setup user interface"""
        # Main container
        main = ttk.Frame(self.root, padding="15")
        main.grid(row=0, column=0, sticky='nsew')
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(2, weight=1)
        
        # Header
        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 15))
        
        tk.Label(header, text="📊 ADVANCED REGIME DETECTION v2.0", 
                font=('JetBrains Mono', 18, 'bold'), bg='#0d1117', fg='#58a6ff').pack(side='left')
        
        self.status_label = tk.Label(header, text="● OFFLINE", font=('Segoe UI', 11, 'bold'),
                                     bg='#0d1117', fg='#f85149')
        self.status_label.pack(side='right', padx=10)
        
        # Control panel
        control = ttk.LabelFrame(main, text="Control Panel", padding="15")
        control.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 15))
        
        # Data source selection
        source_frame = ttk.Frame(control)
        source_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(source_frame, text="Data Source:").pack(side='left', padx=(0, 10))
        
        self.source_var = tk.StringVar(value="yfinance")
        ttk.Radiobutton(source_frame, text="yfinance", variable=self.source_var, 
                       value="yfinance", command=self.on_source_change).pack(side='left', padx=5)
        ttk.Radiobutton(source_frame, text="Custom API", variable=self.source_var, 
                       value="api", command=self.on_source_change).pack(side='left', padx=5)
        
        self.api_entry = ttk.Entry(source_frame, width=40)
        self.api_entry.insert(0, "https://military-jobye-haiqstudios-14f59639.koyeb.app")
        self.api_entry.pack(side='left', padx=(10, 10))
        self.api_entry.config(state='disabled')
        
        self.connect_btn = ttk.Button(source_frame, text="Connect", command=self.connect)
        self.connect_btn.pack(side='left', padx=5)
        
        ttk.Separator(control, orient='horizontal').pack(fill='x', pady=12)
        
        # Symbol and controls
        trade_frame = ttk.Frame(control)
        trade_frame.pack(fill='x')
        
        ttk.Label(trade_frame, text="Symbol:").pack(side='left', padx=(0, 5))
        self.symbol_entry = ttk.Entry(trade_frame, width=12, font=('JetBrains Mono', 11))
        self.symbol_entry.insert(0, "RELIANCE.NS")
        self.symbol_entry.pack(side='left', padx=(0, 15))
        
        ttk.Label(trade_frame, text="Bar Duration (s):").pack(side='left', padx=(0, 5))
        self.duration_var = tk.StringVar(value="5")
        duration_spin = ttk.Spinbox(trade_frame, from_=3, to=60, textvariable=self.duration_var, width=8)
        duration_spin.pack(side='left', padx=(0, 15))
        
        self.start_btn = ttk.Button(trade_frame, text="▶ Start", command=self.start_stream, state='disabled')
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(trade_frame, text="■ Stop", command=self.stop_stream, 
                                   state='disabled', style='Stop.TButton')
        self.stop_btn.pack(side='left', padx=5)
        
        self.recal_btn = ttk.Button(trade_frame, text="⟳ Recalibrate", command=self.recalibrate, state='disabled')
        self.recal_btn.pack(side='left', padx=5)
        
        self.export_btn = ttk.Button(trade_frame, text="💾 Export", command=self.export_data, state='disabled')
        self.export_btn.pack(side='left', padx=5)
        
        # Price display
        price_frame = ttk.Frame(trade_frame)
        price_frame.pack(side='right')
        ttk.Label(price_frame, text="Last Price:").pack(side='left', padx=(0, 5))
        self.price_label = tk.Label(price_frame, text="---.--", font=('JetBrains Mono', 16, 'bold'),
                                    bg='#0d1117', fg='#7ee787')
        self.price_label.pack(side='left')
        
        # Chart and stats
        content = ttk.Frame(main)
        content.grid(row=2, column=0, sticky='nsew', padx=(0, 10))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        
        chart_frame = ttk.LabelFrame(content, text="OHLC Chart with Regime Detection", padding="10")
        chart_frame.grid(row=0, column=0, sticky='nsew')
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        self.chart_container = ttk.Frame(chart_frame)
        self.chart_container.grid(row=0, column=0, sticky='nsew')
        
        # Side panel
        side = ttk.Frame(main)
        side.grid(row=2, column=1, sticky='nsew')
        
        # Statistics
        stats_frame = ttk.LabelFrame(side, text="Regime Statistics", padding="10")
        stats_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=15, width=35, bg='#161b22', fg='#c9d1d9',
                                 font=('JetBrains Mono', 9), relief='flat', wrap='word')
        self.stats_text.pack(fill='both', expand=True)
        
        # Alerts
        alert_frame = ttk.LabelFrame(side, text="Alerts & Events", padding="10")
        alert_frame.pack(fill='both', expand=True)
        
        self.alert_text = tk.Text(alert_frame, height=12, width=35, bg='#161b22', fg='#c9d1d9',
                                 font=('Consolas', 9), relief='flat', wrap='word')
        self.alert_text.pack(fill='both', expand=True)
        
        # Bottom stats bar
        stats_bar = ttk.Frame(main)
        stats_bar.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(15, 0))
        
        self.stat_labels = {}
        for label in ['Bars', 'Regime', 'Vol%', 'Z-Score', 'Switches', 'Duration']:
            frame = ttk.Frame(stats_bar)
            frame.pack(side='left', padx=15)
            ttk.Label(frame, text=f"{label}:", font=('Segoe UI', 9)).pack(side='left')
            val = tk.Label(frame, text="--", font=('JetBrains Mono', 10, 'bold'),
                          bg='#0d1117', fg='#8b949e')
            val.pack(side='left', padx=(5, 0))
            self.stat_labels[label] = val
    
    def setup_chart(self):
        """Setup matplotlib chart"""
        plt.style.use('dark_background')
        
        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
        self.ax.set_facecolor('#161b22')
        
        self.ax.tick_params(colors='#8b949e', labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color('#30363d')
        self.ax.grid(True, alpha=0.2, color='#30363d', linestyle='--', linewidth=0.8)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_container)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.ax.set_title('Waiting for data...', color='#c9d1d9', fontsize=12, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def on_source_change(self):
        """Handle data source change"""
        if self.source_var.get() == "api":
            self.api_entry.config(state='normal')
        else:
            self.api_entry.config(state='disabled')
    
    def connect(self):
        """Connect to data source"""
        try:
            use_yf = self.source_var.get() == "yfinance"
            api_url = self.api_entry.get() if not use_yf else None
            
            self.data_source = DataSource(use_yfinance=use_yf, api_url=api_url)
            
            if self.data_source.connect():
                self.connected = True
                self.connect_btn.config(state='disabled')
                self.start_btn.config(state='normal')
                self.status_label.config(text="● CONNECTED", fg='#7ee787')
                self.add_alert("Connected to data source", "success")
            else:
                messagebox.showerror("Error", "Failed to connect to data source")
        except Exception as e:
            messagebox.showerror("Error", f"Connection error: {str(e)}")
    
    def start_stream(self):
        """Start streaming data"""
        symbol = self.symbol_entry.get().strip()
        if not symbol:
            messagebox.showerror("Error", "Enter a symbol")
            return
        
        try:
            self.bar_duration = int(self.duration_var.get())
        except:
            self.bar_duration = 5
        
        # Reset state
        self.current_symbol = symbol
        self.ohlc_bars.clear()
        self.current_bar = None
        self.bar_start_time = None
        self.regime_detector = RobustRegimeDetector(lookback=30)
        
        # Set symbol
        self.data_source.set_symbol(symbol)
        
        # Update UI
        self.streaming = True
        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.recal_btn.config(state='normal')
        self.export_btn.config(state='normal')
        self.status_label.config(text=f"● STREAMING: {symbol}", fg='#58a6ff')
        
        self.add_alert(f"Started streaming {symbol}", "success")
        
        # Start threads
        threading.Thread(target=self.data_fetcher, daemon=True).start()
        threading.Thread(target=self.bar_manager, daemon=True).start()
        
        # Start chart updates
        self.update_chart()
    
    def stop_stream(self):
        """Stop streaming"""
        self.running = False
        self.streaming = False
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.recal_btn.config(state='disabled')
        self.status_label.config(text="● CONNECTED", fg='#7ee787')
        
        self.add_alert("Streaming stopped", "info")
    
    def data_fetcher(self):
        """Fetch data in background thread"""
        while self.running:
            try:
                data = self.data_source.get_current_price(self.current_symbol)
                if data:
                    self.data_queue.put(('price', data))
                time.sleep(1.0)  # Poll every second
            except Exception as e:
                print(f"Fetch error: {e}")
                time.sleep(2.0)
    
    def bar_manager(self):
        """Manage OHLC bars"""
        while self.running:
            time.sleep(0.1)
            
            if self.current_bar and self.bar_start_time:
                elapsed = (datetime.now() - self.bar_start_time).total_seconds()
                
                if elapsed >= self.bar_duration:
                    # Complete the bar
                    self.ohlc_bars.append(self.current_bar)
                    
                    # Calibrate if enough bars and not yet calibrated
                    if len(self.ohlc_bars) >= 12 and self.regime_detector.calibration_count == 0:
                        self.regime_detector.calibrate(list(self.ohlc_bars))
                        self.data_queue.put(('alert', {
                            'message': 'Initial calibration complete',
                            'type': 'success'
                        }))
                    
                    # Recalibrate periodically
                    if len(self.ohlc_bars) % 20 == 0 and len(self.ohlc_bars) >= 20:
                        self.regime_detector.calibrate(list(self.ohlc_bars))
                        self.data_queue.put(('alert', {
                            'message': f'Auto-recalibration #{self.regime_detector.calibration_count}',
                            'type': 'info'
                        }))
                    
                    # Detect regime
                    prev_regime = self.regime_detector.current_regime
                    new_regime = self.regime_detector.detect_regime(self.current_bar)
                    
                    if new_regime != prev_regime:
                        self.data_queue.put(('regime_change', {
                            'from': prev_regime,
                            'to': new_regime
                        }))
                    
                    # Start new bar
                    last_price = self.current_bar.close
                    self.current_bar = OHLCBar(datetime.now(), last_price)
                    self.bar_start_time = datetime.now()
    
    def recalibrate(self):
        """Manually recalibrate detector"""
        if len(self.ohlc_bars) >= 12:
            self.regime_detector.calibrate(list(self.ohlc_bars))
            self.add_alert(f"Manual recalibration #{self.regime_detector.calibration_count} complete", "success")
        else:
            self.add_alert("Need at least 12 bars to recalibrate", "warning")
    
    def export_data(self):
        """Export session data to JSON"""
        if not self.ohlc_bars:
            messagebox.showinfo("Info", "No data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"regime_data_{self.current_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filename:
            try:
                stats = self.regime_detector.get_statistics()
                
                data = {
                    'symbol': self.current_symbol,
                    'timestamp': datetime.now().isoformat(),
                    'bars': [
                        {
                            'timestamp': bar.timestamp.isoformat(),
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'regime': bar.regime,
                            'range_pct': bar.range_pct,
                            'return_pct': bar.return_pct
                        }
                        for bar in list(self.ohlc_bars)
                    ],
                    'statistics': stats,
                    'model_parameters': {
                        'vol_mean': self.regime_detector.vol_mean,
                        'vol_std': self.regime_detector.vol_std,
                        'z_threshold_low': self.regime_detector.vol_z_thresholds[0],
                        'z_threshold_high': self.regime_detector.vol_z_thresholds[1],
                        'calibration_count': self.regime_detector.calibration_count
                    }
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.add_alert("Data exported successfully", "success")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def process_queue(self):
        """Process data queue (runs in main thread)"""
        try:
            while True:
                msg_type, data = self.data_queue.get_nowait()
                
                if msg_type == 'price':
                    self.on_price_update(data)
                elif msg_type == 'regime_change':
                    self.on_regime_change(data)
                elif msg_type == 'alert':
                    self.add_alert(data['message'], data['type'])
                    
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.process_queue)
    
    def on_price_update(self, data):
        """Handle price update"""
        price = data['price']
        volume = data['volume']
        
        # Update price label
        self.price_label.config(text=f"{price:.2f}")
        
        # Update current bar
        if self.current_bar is None:
            self.current_bar = OHLCBar(datetime.now(), price)
            self.bar_start_time = datetime.now()
        else:
            self.current_bar.update(price, volume)
    
    def on_regime_change(self, data):
        """Handle regime change"""
        from_regime = data['from']
        to_regime = data['to']
        
        from_name = RobustRegimeDetector.REGIME_NAMES[from_regime]
        to_name = RobustRegimeDetector.REGIME_NAMES[to_regime]
        
        self.add_alert(f"Regime: {from_name} → {to_name}", "warning")
    
    def add_alert(self, message, alert_type="info"):
        """Add alert to alerts panel"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "success": "#3fb950",
            "warning": "#d29922",
            "error": "#f85149",
            "info": "#58a6ff"
        }
        
        self.alert_text.insert('1.0', f"[{timestamp}] {message}\n")
        self.alert_text.tag_add(alert_type, '1.0', '1.end')
        self.alert_text.tag_config(alert_type, foreground=colors.get(alert_type, "#c9d1d9"))
        
        # Limit lines
        lines = int(self.alert_text.index('end-1c').split('.')[0])
        if lines > 50:
            self.alert_text.delete('30.0', 'end')
    
    def update_chart(self):
        """Update chart periodically"""
        if not self.running:
            return
        
        try:
            self.draw_chart()
            self.update_statistics()
        except Exception as e:
            print(f"Chart update error: {e}")
        finally:
            self.root.after(500, self.update_chart)
    
    def draw_chart(self):
        """Draw OHLC chart with regime backgrounds"""
        self.ax.clear()
        
        bars = list(self.ohlc_bars) + ([self.current_bar] if self.current_bar else [])
        
        if not bars:
            self.ax.set_title('Waiting for data...', color='#c9d1d9')
            self.canvas.draw_idle()
            return
        
        # Calculate y-axis range
        all_prices = [b.low for b in bars] + [b.high for b in bars]
        y_min = min(all_prices) * 0.998
        y_max = max(all_prices) * 1.002
        
        # Draw regime backgrounds
        for i, bar in enumerate(bars):
            regime = bar.regime
            bg_rect = Rectangle((i - 0.5, y_min), 1, y_max - y_min,
                              facecolor=RobustRegimeDetector.REGIME_BG[regime],
                              alpha=0.4, zorder=0)
            self.ax.add_patch(bg_rect)
        
        # Draw candlesticks
        for i, bar in enumerate(bars):
            color = '#3fb950' if bar.close >= bar.open else '#f85149'
            edge = '#7ee787' if bar.close >= bar.open else '#ff7b72'
            
            # Body
            body_height = max(abs(bar.close - bar.open), y_max * 0.0001)
            body_bottom = min(bar.open, bar.close)
            
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, edgecolor=edge, linewidth=1.5,
                           alpha=0.9, zorder=2)
            self.ax.add_patch(rect)
            
            # Wicks
            self.ax.plot([i, i], [bar.low, body_bottom], color=edge, linewidth=1.5, zorder=1)
            self.ax.plot([i, i], [body_bottom + body_height, bar.high], color=edge, linewidth=1.5, zorder=1)
        
        # Styling
        self.ax.set_facecolor('#161b22')
        x_labels = [b.timestamp.strftime('%H:%M:%S') for b in bars]
        self.ax.set_xticks(range(len(bars)))
        self.ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlim(-0.5, max(self.max_bars - 0.5, len(bars) - 0.5))
        
        for spine in self.ax.spines.values():
            spine.set_color('#30363d')
        self.ax.grid(True, alpha=0.2, color='#30363d', linestyle='--')
        
        # Title
        if bars:
            regime = bars[-1].regime
            regime_name = RobustRegimeDetector.REGIME_NAMES[regime]
            self.ax.set_title(f'{self.current_symbol} | {regime_name} Regime | {len(bars)} bars',
                            color='#c9d1d9', fontsize=12, fontweight='bold')
        
        self.fig.tight_layout()
        self.canvas.draw_idle()
    
    def update_statistics(self):
        """Update statistics display"""
        bars = list(self.ohlc_bars) + ([self.current_bar] if self.current_bar else [])
        
        if not bars:
            return
        
        # Update bottom stats
        self.stat_labels['Bars'].config(text=str(len(bars)))
        
        regime = bars[-1].regime
        regime_name = RobustRegimeDetector.REGIME_NAMES[regime]
        self.stat_labels['Regime'].config(
            text=regime_name,
            fg=RobustRegimeDetector.REGIME_COLORS[regime]
        )
        
        # Calculate current bar's volatility and z-score
        current_vol = bars[-1].range_pct * 100
        self.stat_labels['Vol%'].config(text=f"{current_vol:.3f}")
        
        # Calculate z-score
        if self.regime_detector.vol_std > 0:
            z_score = (bars[-1].range_pct - self.regime_detector.vol_mean) / self.regime_detector.vol_std
            self.stat_labels['Z-Score'].config(text=f"{z_score:.2f}")
        
        stats = self.regime_detector.get_statistics()
        self.stat_labels['Switches'].config(text=str(stats.get('switches', 0)))
        self.stat_labels['Duration'].config(text=str(stats.get('current_duration', 0)))
        
        # Update statistics panel
        self.stats_text.delete('1.0', 'end')
        
        text = f"""REGIME DISTRIBUTION
{'=' * 32}
CALM:     {stats.get('calm_pct', 0):.1f}%
ACTIVE:   {stats.get('active_pct', 0):.1f}%
VOLATILE: {stats.get('volatile_pct', 0):.1f}%

DYNAMICS
{'=' * 32}
Total Switches: {stats.get('switches', 0)}
Current Duration: {stats.get('current_duration', 0)} bars

NORMALIZATION
{'=' * 32}
Vol Mean:  {stats.get('vol_mean', 0):.5f}
Vol Std:   {stats.get('vol_std', 0):.5f}

Z-SCORE THRESHOLDS
{'=' * 32}
CALM→ACTIVE:   {self.regime_detector.vol_z_thresholds[0]:.2f}
ACTIVE→VOLATILE: {self.regime_detector.vol_z_thresholds[1]:.2f}

AVG VOLATILITY BY REGIME
{'=' * 32}
"""
        
        for regime_id in [0, 1, 2]:
            key = f'avg_vol_{regime_id}'
            if key in stats:
                name = RobustRegimeDetector.REGIME_NAMES[regime_id]
                text += f"{name}: {stats[key]:.5f}\n"
        
        text += f"\nCALIBRATIONS: {self.regime_detector.calibration_count}\n"
        text += f"DATA POINTS: {len(self.regime_detector.volatilities)}\n"
        
        self.stats_text.insert('1.0', text)


def main():
    root = tk.Tk()
    app = MarketDashboard(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'running', False), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()