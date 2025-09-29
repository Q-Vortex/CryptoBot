import pandas as pd
import numpy as np
from binance.client import Client
import ta
from datetime import datetime, timedelta
import time

class MicroLiquidityHunter:
    def __init__(self, initial_balance=20, api_key=None, api_secret=None):
        # Инициализация клиента Binance (можно без ключей для публичных данных)
        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
        else:
            self.client = Client()
            
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.consecutive_losses = 0
        self.portfolio = {}
        
    def get_real_market_data(self, symbol, timeframe='1m', lookback=500):
        """Получаем РЕАЛЬНЫЕ данные с Binance"""
        try:
            klines = self.client.get_klines(
                symbol=symbol, 
                interval=timeframe, 
                limit=lookback
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy', 'taker_sell', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except Exception as e:
            print(f"Ошибка получения данных для {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Получаем текущую цену символа"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Ошибка получения цены для {symbol}: {e}")
            return None
    
    def wait_for_price_movement(self, symbol, entry_price, stop_loss, take_profit, signal, max_wait_minutes=60):
        """Ожидаем реального движения цены до тейк-профита или стоп-лосса"""
        print(f"   ⏳ Ожидание движения цены... (макс. {max_wait_minutes} минут)")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=max_wait_minutes)
        
        while datetime.now() < end_time:
            try:
                current_price = self.get_current_price(symbol)
                
                if current_price is None:
                    time.sleep(10)  # Ждем 10 секунд перед повторной попыткой
                    continue
                
                # Расчет текущего PnL
                if signal == 'BUY':
                    current_pnl_percent = (current_price - entry_price) / entry_price * 100
                else:
                    current_pnl_percent = (entry_price - current_price) / entry_price * 100
                
                # 🔥 ДОБАВЛЯЕМ PnL-СТОПЫ 🔥
                
                # 🚨 МАКСИМАЛЬНЫЙ УБЫТОК -5%
                if current_pnl_percent <= -5:
                    print(f"   🚨 СТОП ПО УБЫТКУ! PnL: {current_pnl_percent:.2f}%")
                    return current_price, 'PNL_STOP'
                
                # 🎯 МИНИМАЛЬНАЯ ПРИБЫЛЬ +1% (если хочешь гарантировать прибыль)
                target_pnl_percent = (take_profit / entry_price - 1) * 100 if signal == 'BUY' else (entry_price / take_profit - 1) * 100
                if current_pnl_percent >= 1 and current_pnl_percent < target_pnl_percent:
                    print(f"   🎯 РАННИЙ ТЕЙК! PnL: {current_pnl_percent:.2f}%")
                    return current_price, 'EARLY_TP'
                
                # Проверяем основные условия выхода
                if signal == 'BUY':
                    if current_price >= take_profit:
                        print(f"   ✅ ТЕЙК-ПРОФИТ достигнут! Цена: {current_price:.4f}")
                        return current_price, 'TP'
                    elif current_price <= stop_loss:
                        print(f"   ❌ СТОП-ЛОСС достигнут! Цена: {current_price:.4f}")
                        return current_price, 'SL'
                else:  # SELL
                    if current_price <= take_profit:
                        print(f"   ✅ ТЕЙК-ПРОФИТ достигнут! Цена: {current_price:.4f}")
                        return current_price, 'TP'
                    elif current_price >= stop_loss:
                        print(f"   ❌ СТОП-ЛОСС достигнут! Цена: {current_price:.4f}")
                        return current_price, 'SL'
                
                # Красивое отображение с обновлением строки
                print(f"   📊 Текущая цена: {current_price:.4f} | PnL: {current_pnl_percent:+.2f}%", end='\r')
                
                time.sleep(5)  # Проверяем каждые 5 секунд
                
            except Exception as e:
                print(f"   ⚠️ Ошибка при отслеживании цены: {e}")
                time.sleep(10)
        
        # Если время вышло, выходим по текущей цене
        final_price = self.get_current_price(symbol)
        if final_price is None:
            final_price = entry_price  # Если не удалось получить цену, используем входную
        
        print(f"   ⏰ Время вышло. Выход по цене: {final_price:.4f}")
        return final_price, 'TIME'
    
    def calculate_max_trade_size(self):
        """Рассчитываем максимальный размер сделки на основе баланса"""
        if self.initial_balance <= 20:
            max_trade = self.balance * 0.4  # 40% для микро-депозита
        elif self.initial_balance <= 50:
            max_trade = self.balance * 0.35  # 35% для маленького депозита
        else:
            max_trade = self.balance * 0.25  # 25% для стандартного депозита
        
        return max_trade
    
    def get_affordable_symbols(self, symbols):
        """Фильтруем пары по доступной цене на основе максимального размера сделки"""
        affordable_symbols = []
        max_trade_size = self.calculate_max_trade_size()
        
        print(f"   💰 Максимальный размер сделки: ${max_trade_size:.2f}")
        
        for symbol in symbols:
            try:
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    continue
                
                # Рассчитываем минимальную цену, которую можем позволить
                if current_price <= max_trade_size:
                    affordable_symbols.append(symbol)
                    print(f"   ✅ {symbol}: ${current_price:.4f}")
                else:
                    print(f"   ❌ {symbol}: ${current_price:.4f} (нужно ${max_trade_size:.2f}+)")
                    
            except Exception as e:
                print(f"   ⚠️  {symbol}: ошибка получения цены")
                
        return affordable_symbols
    
    def calculate_micro_trade_size(self, opportunity):
        """РАСЧЕТ РАЗМЕРА ДЛЯ МАЛЕНЬКОГО ДЕПОЗИТА"""
        # Базовый размер зависит от начального депозита
        if self.initial_balance <= 20:
            base_size = 5  # $5 для депозита $20
        elif self.initial_balance <= 50:
            base_size = 10  # $10 для депозита $50
        else:
            base_size = 20  # $20 для депозита $100+
        
        # МАРТИНГЕЙЛ для микро-депозита (менее агрессивный)
        if self.consecutive_losses > 0:
            martingale_multiplier = 1.5 ** min(self.consecutive_losses, 2)  # 1, 1.5, 2.25
            trade_size = base_size * martingale_multiplier
        else:
            trade_size = base_size
            
        # УМНОЖАЕМ на уверенность сигнала
        confidence_factor = opportunity['confidence'] / 50  # 0.8-1.6
        trade_size = trade_size * confidence_factor
        
        # Ограничиваем максимальный размер
        max_trade = self.calculate_max_trade_size()
        trade_size = min(trade_size, max_trade)
        
        # Минимальный размер $2 для микро-торговли
        trade_size = max(trade_size, 2)
        
        # Округляем до 2 знаков для красоты
        trade_size = round(trade_size, 2)
        
        return trade_size
    
    def execute_real_trade(self, opportunity):
        """Исполнение РЕАЛЬНОЙ демо-сделки с отслеживанием реальных цен"""
        if opportunity['signal'] == 'HOLD':
            return None
            
        # ДИНАМИЧЕСКИЙ РАЗМЕР для микро-депозита
        trade_size = self.calculate_micro_trade_size(opportunity)
        
        # ПРОВЕРКА ДОСТАТОЧНОСТИ СРЕДСТВ
        if trade_size > self.balance:
            print(f"❌ Недостаточно средств: нужно ${trade_size:.2f}, есть ${self.balance:.2f}")
            return None
        
        entry_price = opportunity['entry']
        stop_loss = opportunity['stop_loss']
        take_profit = opportunity['take_profit']
        
        # РЕАЛЬНЫЙ РАСЧЕТ КОЛИЧЕСТВА
        quantity = trade_size / entry_price
        
        print(f"🔹 МИКРО-СДЕЛКА: {opportunity['symbol']} {opportunity['signal']}")
        print(f"   💰 Размер: ${trade_size:.2f} | Вход: {entry_price:.4f}")
        print(f"   🛡️ Стоп-лосс: {stop_loss:.4f} | 🎯 Тейк-профит: {take_profit:.4f}")
        print(f"   📦 Количество: {quantity:.6f}")
        
        # ОЖИДАЕМ РЕАЛЬНОГО ДВИЖЕНИЯ ЦЕНЫ
        exit_price, exit_reason = self.wait_for_price_movement(
            opportunity['symbol'], entry_price, stop_loss, take_profit, opportunity['signal']
        )
        
        # Расчет PnL
        if opportunity['signal'] == 'BUY':
            pnl_percent = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_percent = (entry_price - exit_price) / entry_price * 100
        
        pnl_usd = (trade_size * pnl_percent) / 100
        
        # Обновляем счетчик проигрышей
        if pnl_usd < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # ОБНОВЛЯЕМ ДЕМО-БАЛАНС
        old_balance = self.balance
        self.balance += pnl_usd
        
        trade_result = {
            'symbol': opportunity['symbol'],
            'signal': opportunity['signal'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_size': trade_size,
            'quantity': quantity,
            'pnl_percent': pnl_percent,
            'pnl_usd': pnl_usd,
            'timestamp': datetime.now(),
            'result': exit_reason,
            'balance_before': old_balance,
            'balance_after': self.balance
        }
        
        self.trades.append(trade_result)
        
        return trade_result

    # ОСТАЛЬНЫЕ МЕТОДЫ ОСТАЮТСЯ ПРЕЖНИМИ
    def detect_liquidity_clusters(self, symbol, timeframe='1m', lookback=500):
        df = self.get_real_market_data(symbol, timeframe, lookback)
        if df is None:
            return {'supports': [], 'resistances': [], 'current_price': 0}
        
        df['price_range'] = (df['high'] + df['low']) / 2
        volume_profile = df.groupby('price_range')['volume'].sum()
        
        if len(volume_profile) == 0:
            return {'supports': [], 'resistances': [], 'current_price': float(df['close'].iloc[-1])}
        
        poc_level = volume_profile.idxmax()
        
        from sklearn.cluster import KMeans
        prices = df[['high', 'low']].values.flatten()
        prices = prices.reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=8, random_state=42)
        kmeans.fit(prices)
        
        clusters = kmeans.cluster_centers_.flatten()
        clusters = np.sort(clusters)
        
        current_price = float(df['close'].iloc[-1])
        
        supports = [c for c in clusters if c < current_price]
        resistances = [c for c in clusters if c > current_price]
        
        supports = sorted(supports, reverse=True)[:3]
        resistances = sorted(resistances)[:3]
        
        return {
            'supports': supports,
            'resistances': resistances,
            'poc': poc_level,
            'current_price': current_price
        }
    
    def analyze_market_structure(self, symbol):
        df = self.get_real_market_data(symbol, '1h', 100)
        if df is None:
            return {'trend': 'BULLISH', 'recent_high': 0, 'recent_low': 0, 'sma_20': 0, 'sma_50': 0}
        
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        current_sma_20 = df['sma_20'].iloc[-1]
        current_sma_50 = df['sma_50'].iloc[-1]
        
        trend = "BULLISH" if current_sma_20 > current_sma_50 else "BEARISH"
        
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        return {
            'trend': trend,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'sma_20': current_sma_20,
            'sma_50': current_sma_50
        }
    
    def calculate_aggressive_targets(self, symbol, current_price, signal, confidence):
        volatility_map = {
            'BTCUSDT': 0.025, 'ETHUSDT': 0.030, 'BNBUSDT': 0.035, 
            'ADAUSDT': 0.045, 'DOTUSDT': 0.040, 'LINKUSDT': 0.042,
            'LTCUSDT': 0.038, 'BCHUSDT': 0.043, 'XRPUSDT': 0.047, 
            'EOSUSDT': 0.050,
        }
        
        base_volatility = volatility_map.get(symbol, 0.035)
        confidence_boost = (confidence / 100) * 0.5
        volatility = base_volatility * (1 + confidence_boost)
        
        if signal == 'BUY':
            take_profit = current_price * (1 + volatility)
            stop_loss = current_price * (1 - volatility * 0.6)
        else:
            take_profit = current_price * (1 - volatility) 
            stop_loss = current_price * (1 + volatility * 0.6)
        
        return take_profit, stop_loss
    
    def find_trading_opportunity(self, symbol):
        try:
            liquidity = self.detect_liquidity_clusters(symbol)
            market_structure = self.analyze_market_structure(symbol)
            
            current_price = liquidity['current_price']
            nearest_support = liquidity['supports'][0] if liquidity['supports'] else None
            nearest_resistance = liquidity['resistances'][0] if liquidity['resistances'] else None
            
            support_distance = ((current_price - nearest_support) / current_price * 100) if nearest_support else None
            resistance_distance = ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None
            
            opportunity = {
                'symbol': symbol, 'current_price': current_price, 'trend': market_structure['trend'],
                'signal': 'HOLD', 'entry': None, 'stop_loss': None, 'take_profit': None, 'confidence': 0
            }
            
            if (nearest_support and support_distance <= 0.5 and market_structure['trend'] == 'BULLISH'):
                confidence = min(80, 100 - (support_distance * 100))
                take_profit, stop_loss = self.calculate_aggressive_targets(symbol, current_price, 'BUY', confidence)
                opportunity.update({
                    'signal': 'BUY', 'entry': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'confidence': confidence
                })
            elif (nearest_resistance and resistance_distance <= 0.5 and market_structure['trend'] == 'BEARISH'):
                confidence = min(80, 100 - (resistance_distance * 100))
                take_profit, stop_loss = self.calculate_aggressive_targets(symbol, current_price, 'SELL', confidence)
                opportunity.update({
                    'signal': 'SELL', 'entry': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'confidence': confidence
                })
            
            return opportunity
            
        except Exception as e:
            return None
    
    def scan_and_trade(self, symbols, max_cycles=10):
        """Сканируем рынок для МИКРО-ДЕПОЗИТА с реальными данными"""
        print(f"🚀 Запуск МИКРО-торговой сессии с РЕАЛЬНЫМИ данными")
        print(f"💰 Начальный демо-баланс: ${self.initial_balance:.2f}")
        print(f"🔄 Количество циклов: {max_cycles}")
        print("⚡ СТРАТЕГИЯ: АДАПТИВНЫЙ РАЗМЕР + МИКРО-МАРТИНГЕЙЛ + PnL-СТОПЫ")
        print("📊 РЕЖИМ: МИКРО-ДЕМО-СЧЕТ с РЕАЛЬНЫМИ ЦЕНАМИ")
        print("-" * 70)
        
        # ФИЛЬТРУЕМ ПАРЫ ПО ДОСТУПНОСТИ НА ОСНОВЕ БАЛАНСА
        print(f"\n🔍 Поиск доступных пар на основе баланса ${self.initial_balance:.2f}:")
        affordable_symbols = self.get_affordable_symbols(symbols)
        
        if not affordable_symbols:
            print("❌ Не найдено доступных пар для торговли!")
            return
            
        print(f"✅ Доступно пар: {len(affordable_symbols)}")
        
        for cycle in range(max_cycles):
            print(f"\n🔄 Цикл {cycle + 1}/{max_cycles}")
            print(f"🕒 {datetime.now().strftime('%H:%M:%S')} - Поиск сделок...")
            print(f"📊 Подряд проигрышей: {self.consecutive_losses}")
            print(f"💳 Текущий баланс: ${self.balance:.2f}")
            
            opportunities = []
            for symbol in affordable_symbols:  # Торгуем только доступные пары
                try:
                    opportunity = self.find_trading_opportunity(symbol)
                    if opportunity and opportunity['signal'] != 'HOLD':
                        opportunities.append(opportunity)
                    time.sleep(0.05)
                except Exception as e:
                    continue
            
            if opportunities:
                opportunities.sort(key=lambda x: x['confidence'], reverse=True)
                
                print(f"🎯 Найдено {len(opportunities)} сделок:")
                total_cycle_pnl = 0
                
                for opp in opportunities:
                    trade_result = self.execute_real_trade(opp)
                    if trade_result:
                        status = self.get_result_display(trade_result['result'])
                        pnl_display = f"{trade_result['pnl_usd']:+.2f}$ ({trade_result['pnl_percent']:+.2f}%)"
                        size_display = f"Размер: ${trade_result['trade_size']:.2f}"
                        print(f"   {opp['symbol']} {opp['signal']} | {pnl_display} | {size_display} | {status}")
                        total_cycle_pnl += trade_result['pnl_usd']
            else:
                print("📭 Подходящих сделок не найдено")
            
            print(f"💰 Сделок в цикле: {len(opportunities)} | Общий PnL цикла: ${total_cycle_pnl:+.2f}")
            print(f"💳 Новый баланс: ${self.balance:.2f}")
            
            if cycle < max_cycles - 1:
                print("⏳ Ожидание 30 секунд...")
                time.sleep(30)
    
    def get_result_display(self, result_type):
        """Возвращает отображаемое название результата"""
        display_map = {
            'TP': '✅ TP',
            'SL': '❌ SL',
            'TIME': '⏰ TIME', 
            'PNL_STOP': '🚨 PNL_STOP',
            'EARLY_TP': '🎯 EARLY_TP'
        }
        return display_map.get(result_type, result_type)
    
    def generate_final_report(self):
        """Генерация финального отчета для МИКРО-счета"""
        if not self.trades:
            print("\n📊 Нет данных для отчета")
            return
        
        df = pd.DataFrame(self.trades)
        
        total_trades = len(df)
        winning_trades = len(df[df['result'] == 'TP']) + len(df[df['result'] == 'EARLY_TP'])
        losing_trades = len(df[df['result'] == 'SL']) + len(df[df['result'] == 'PNL_STOP'])
        time_out_trades = len(df[df['result'] == 'TIME'])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl_usd = df['pnl_usd'].sum()
        total_pnl_percent = (total_pnl_usd / self.initial_balance) * 100
        
        avg_win = df[df['pnl_usd'] > 0]['pnl_usd'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl_usd'] < 0]['pnl_usd'].mean() if losing_trades > 0 else 0
        
        avg_trade_size = df['trade_size'].mean() if 'trade_size' in df.columns else 0
        
        print("\n" + "="*70)
        print("📊 ФИНАЛЬНЫЙ ОТЧЕТ - МИКРО-ДЕМО-СЧЕТ с PnL-СТОПАМИ")
        print("="*70)
        print(f"Всего сделок: {total_trades}")
        print(f"Выигрышных: {winning_trades} ({win_rate:.1f}%)")
        print(f"Проигрышных: {losing_trades}")
        print(f"По времени: {time_out_trades}")
        print(f"Средний размер сделки: ${avg_trade_size:.2f}")
        print(f"Средний выигрыш: ${avg_win:.2f}")
        print(f"Средний проигрыш: ${avg_loss:.2f}")
        print(f"Соотношение Win/Loss: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
        print(f"Начальный баланс: ${self.initial_balance:.2f}")
        print(f"Конечный баланс: ${self.balance:.2f}")
        print(f"Общий PnL: ${total_pnl_usd:+.2f} ({total_pnl_percent:+.2f}%)")
        
        # Статистика по типам закрытия
        result_counts = df['result'].value_counts()
        print(f"\n📈 Типы закрытия сделок:")
        for result_type, count in result_counts.items():
            print(f"   {self.get_result_emoji(result_type)} {result_type}: {count}")
        
        if total_pnl_usd > 0:
            print("🎉 МИКРО-торговля прибыльна!")
        else:
            print("💸 МИКРО-торговля убыточна")
        
        print("\n📈 Детали по сделкам:")
        for i, trade in enumerate(self.trades, 1):
            result_display = self.get_result_display(trade['result'])
            print(f"{i:2d}. {trade['symbol']} {trade['signal']} | "
                  f"Размер: ${trade['trade_size']:.2f} | "
                  f"Вход: {trade['entry_price']:.4f} | Выход: {trade['exit_price']:.4f} | "
                  f"PnL: {trade['pnl_usd']:+.2f}$ ({trade['pnl_percent']:+.2f}%) | {result_display}")
    
    def get_result_emoji(self, result_type):
        """Возвращает эмодзи для типа результата"""
        emoji_map = {
            'TP': '✅',
            'SL': '❌', 
            'TIME': '⏰',
            'PNL_STOP': '🚨',
            'EARLY_TP': '🎯'
        }
        return emoji_map.get(result_type, '📊')

# Запуск МИКРО-сессии с реальными данными
if __name__ == "__main__":
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT',
        'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XRPUSDT', 'EOSUSDT',
        'TRXUSDT', 'MATICUSDT', 'ATOMUSDT', 'FILUSDT', 'ETCUSDT'
    ]
    
    # Запускаем с любым депозитом - код сам адаптируется
    hunter = MicroLiquidityHunter(initial_balance=28.36)  # Или 50, или 100, или 1000
    hunter.scan_and_trade(symbols, max_cycles=5)
    hunter.generate_final_report()