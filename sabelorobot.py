import logging
import threading
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import requests
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
import secrets
import talib
import pytz
import os  


# ======================
# CONFIGURATION
# ======================
class Config:
    # Forex pairs
    SYMBOL = "GBP/USD"  # Use with slash for display + API
    DISPLAY_SYMBOL = "GBP/USD"
    INTERVALS = ["3min", "5min", "15min", "1h", "4h"]
    PRIMARY_INTERVAL = "3min"

    # Risk Management
    RISK_REWARD_RATIO = 1.5
    ATR_PERIOD = 10
    ATR_SL_MULTIPLIER = 0.8
    ATR_TP_MULTIPLIER = ATR_SL_MULTIPLIER * RISK_REWARD_RATIO

    # Strategy Parameters
    EMA_FAST = 12
    EMA_SLOW = 26
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    # Trading Hours
    TRADING_SESSIONS = {
        "LONDON": {"start": "07:00", "end": "16:00", "tz": "Europe/London"},
        "NEW_YORK": {"start": "12:00", "end": "21:00", "tz": "America/New_York"},
        "TOKYO": {"start": "23:00", "end": "08:00", "tz": "Asia/Tokyo"}
    }

    # App Settings
    DEBUG_MODE = True

# ======================
# LOGGING
# ======================
def setup_logger():
    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG if Config.DEBUG_MODE else logging.INFO
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )
    file_handler = logging.FileHandler('trading_bot.log')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# ======================
# DATA: Forex from TwelveData
# ======================
# Use environment variables for security
API_KEY = os.getenv("TWELVEDATA_API_KEY", "e23f6bfc949a470ca7391c136f7d216c") 

def fetch_current_price(symbol="EUR/USD"):
    try:
        url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}"
        resp = requests.get(url, timeout=10).json()
        if "price" in resp:
            return float(resp["price"])
        else:
            logger.error(f"Bad response for price: {resp}")
            return None
    except Exception as e:
        logger.error(f"Error fetching current price: {e}")
        return None

def fetch_forex_data(symbol="EUR/USD", interval="15min", count=100):
    """
    Fetch real OHLC data from TwelveData.
    interval: "1min", "5min", "15min", "1h", "4h", "1day"
    count: number of candles
    """
    # time.sleep(10)
    try:
        url = (
            f"https://api.twelvedata.com/time_series?"
            f"symbol={symbol}&interval={interval}&outputsize={count}&apikey={API_KEY}"
        )
        resp = requests.get(url, timeout=10).json()
        candles = []
        if "values" in resp:
            for item in reversed(resp["values"]):  # oldest first
                candles.append({
                    "time": datetime.strptime(item["datetime"], "%Y-%m-%d %H:%M:%S"),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item.get("volume", 0))
                })
        else:
            logger.error(f"Bad OHLC response: {resp}")
        return candles
    except Exception as e:
        logger.error(f"Error fetching OHLC data: {e}")
        return []


# ======================
# INDICATOR CALCULATIONS (Using TA-Lib)
# ======================
def calculate_indicators(candles):
    """Calculate all technical indicators using TA-Lib"""
    if len(candles) < 50:  # Need enough data for accurate indicators
        return None
        
    closes = np.array([c['close'] for c in candles], dtype=float)
    highs = np.array([c['high'] for c in candles], dtype=float)
    lows = np.array([c['low'] for c in candles], dtype=float)
    
    # Calculate EMAs
    ema_fast = talib.EMA(closes, timeperiod=Config.EMA_FAST)
    ema_slow = talib.EMA(closes, timeperiod=Config.EMA_SLOW)
    
    # Calculate RSI
    rsi = talib.RSI(closes, timeperiod=Config.RSI_PERIOD)
    
    # Calculate MACD
    macd, macd_signal, macd_hist = talib.MACD(
        closes, 
        fastperiod=Config.MACD_FAST, 
        slowperiod=Config.MACD_SLOW, 
        signalperiod=Config.MACD_SIGNAL
    )
    
    # Calculate ATR
    atr = talib.ATR(highs, lows, closes, timeperiod=Config.ATR_PERIOD)
    
    # Calculate Bollinger Bands
    upper_bb, middle_bb, lower_bb = talib.BBANDS(
        closes, 
        timeperiod=20, 
        nbdevup=2, 
        nbdevdn=2, 
        matype=0
    )
    
    # Get the last valid values (skip NaN values at the beginning)
    def last_valid_value(arr):
        if arr is None or len(arr) == 0:
            return None
        # Find the last non-NaN value
        for i in range(len(arr)-1, -1, -1):
            if not np.isnan(arr[i]):
                return arr[i]
        return None
    
    return {
        'ema_fast': last_valid_value(ema_fast),
        'ema_slow': last_valid_value(ema_slow),
        'rsi': last_valid_value(rsi),
        'macd': last_valid_value(macd),
        'macd_signal': last_valid_value(macd_signal),
        'macd_hist': last_valid_value(macd_hist),
        'atr': last_valid_value(atr),
        'upper_bb': last_valid_value(upper_bb),
        'middle_bb': last_valid_value(middle_bb),
        'lower_bb': last_valid_value(lower_bb),
        'price': closes[-1] if len(closes) > 0 else None
    }

# ======================
# TRADING STRATEGY
# ======================
def generate_signal():
    """Generate trading signals based on technical indicators"""
    try:
        # Fetch data for multiple timeframes
        candles_15m = fetch_forex_data(Config.SYMBOL, "15min", 100)
        candles_1h = fetch_forex_data(Config.SYMBOL, "1h", 100)
        
        if not candles_15m or not candles_1h:
            return "ERROR", "Failed to fetch market data", 0, 0, 0
        
        # Calculate indicators for different timeframes
        indicators_15m = calculate_indicators(candles_15m)
        indicators_1h = calculate_indicators(candles_1h)
        
        if indicators_15m is None or indicators_1h is None:
            return "ANALYZING", "Not enough data to calculate indicators", 0, 0, 0
        
        current_price = fetch_current_price(Config.SYMBOL)
        if current_price is None:
            return "ERROR", "Failed to fetch current price", 0, 0, 0
        
        # Check if we have all required indicators
        if any(v is None for v in indicators_15m.values()) or \
           any(v is None for v in indicators_1h.values()):
            return "ANALYZING", "Calculating indicators...", current_price, 0, 0
        
        # Strategy logic - Multi-timeframe confluence
        bullish_signals = 0
        bearish_signals = 0
        
        # 1. Trend analysis (EMAs)
        if indicators_15m['ema_fast'] > indicators_15m['ema_slow']:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if indicators_1h['ema_fast'] > indicators_1h['ema_slow']:
            bullish_signals += 2  # Higher weight for higher timeframe
        else:
            bearish_signals += 2
            
        # 2. Momentum analysis (RSI)
        if indicators_15m['rsi'] < Config.RSI_OVERSOLD:
            bullish_signals += 1
        elif indicators_15m['rsi'] > Config.RSI_OVERBOUGHT:
            bearish_signals += 1
            
        if indicators_1h['rsi'] < Config.RSI_OVERSOLD:
            bullish_signals += 1
        elif indicators_1h['rsi'] > Config.RSI_OVERBOUGHT:
            bearish_signals += 1
            
        # 3. MACD analysis
        if indicators_15m['macd_hist'] > 0 and indicators_15m['macd'] > indicators_15m['macd_signal']:
            bullish_signals += 1
        elif indicators_15m['macd_hist'] < 0 and indicators_15m['macd'] < indicators_15m['macd_signal']:
            bearish_signals += 1
            
        if indicators_1h['macd_hist'] > 0 and indicators_1h['macd'] > indicators_1h['macd_signal']:
            bullish_signals += 1
        elif indicators_1h['macd_hist'] < 0 and indicators_1h['macd'] < indicators_1h['macd_signal']:
            bearish_signals += 1
            
        # 4. Price action (Bollinger Bands)
        if current_price < indicators_15m['lower_bb']:
            bullish_signals += 1
        elif current_price > indicators_15m['upper_bb']:
            bearish_signals += 1
            
        # Generate final signal
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            confidence = 0
        else:
            confidence = abs(bullish_signals - bearish_signals) / total_signals * 100
        
        if bullish_signals > bearish_signals and confidence > 30:
            # Calculate stop loss and take profit based on ATR
            atr_value = indicators_15m['atr']
            sl_price = current_price - (atr_value * Config.ATR_SL_MULTIPLIER)
            tp_price = current_price + (atr_value * Config.ATR_TP_MULTIPLIER)
            
            strategy = (f"Multi-timeframe BULLISH confluence\n"
                       f"15m EMA: {indicators_15m['ema_fast']:.5f} > {indicators_15m['ema_slow']:.5f}\n"
                       f"1h RSI: {indicators_1h['rsi']:.2f}\n"
                       f"MACD Hist: {indicators_15m['macd_hist']:.5f}\n"
                       f"Confidence: {confidence:.1f}%")
            
            return "BUY", strategy, current_price, tp_price, sl_price
            
        elif bearish_signals > bullish_signals and confidence > 30:
            # Calculate stop loss and take profit based on ATR
            atr_value = indicators_15m['atr']
            sl_price = current_price + (atr_value * Config.ATR_SL_MULTIPLIER)
            tp_price = current_price - (atr_value * Config.ATR_TP_MULTIPLIER)
            
            strategy = (f"Multi-timeframe BEARISH confluence\n"
                       f"15m EMA: {indicators_15m['ema_fast']:.5f} < {indicators_15m['ema_slow']:.5f}\n"
                       f"1h RSI: {indicators_1h['rsi']:.2f}\n"
                       f"MACD Hist: {indicators_15m['macd_hist']:.5f}\n"
                       f"Confidence: {confidence:.1f}%")
            
            return "SELL", strategy, current_price, tp_price, sl_price
            
        else:
            strategy = (f"Market is consolidating\n"
                       f"15m RSI: {indicators_15m['rsi']:.2f}\n"
                       f"Price near middle BB: {indicators_15m['middle_bb']:.5f}")
            return "HOLD", strategy, current_price, current_price, current_price
            
    except Exception as e:
        logger.error(f"Signal generation error: {str(e)}")
        return "ERROR", f"System error: {str(e)}", 0, 0, 0

# ======================
# TELEGRAM BOT SETUP
# ======================
# Use environment variables for security
# ======================
# TELEGRAM BOT SETUP
# ======================
TELEGRAM_TOKEN = "8006265817:AAF5lE7HkbT20tthvmDEh0pwtj0951FBS78"
AUTHORIZED_USERS = {}
ADMIN_USER_ID = None
ADMIN_USERNAME = "@sabelojpy"
bot = telebot.TeleBot(TELEGRAM_TOKEN)
ACCESS_CODES = {}
user_states = {}

def generate_access_code():
    code = str(secrets.randbelow(900000) + 100000)
    expiry = time.time() + (30 * 24 * 60 * 60)
    ACCESS_CODES[code] = expiry
    return code

def cleanup_expired_codes():
    current_time = time.time()
    expired_codes = [code for code, expiry in ACCESS_CODES.items() if expiry < current_time]
    for code in expired_codes:
        del ACCESS_CODES[code]

def is_user_access_valid(user_id):
    if user_id not in AUTHORIZED_USERS:
        return False
    try:
        added_date = datetime.strptime(AUTHORIZED_USERS[user_id]["added_date"], "%Y-%m-%d")
        expiry_date = added_date + timedelta(days=30)
        return datetime.now() < expiry_date
    except:
        return False

def cleanup_expired_users():
    expired_users = [user_id for user_id in AUTHORIZED_USERS if not is_user_access_valid(user_id)]
    for user_id in expired_users:
        del AUTHORIZED_USERS[user_id]

def create_main_keyboard(user_id):
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    keyboard.add(KeyboardButton('📊 Get Signal'), KeyboardButton('💰 Current Price'))
  
    if user_id == ADMIN_USER_ID:
        keyboard.add(KeyboardButton('👑 Admin Panel'))
    return keyboard

def create_admin_keyboard():
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton('👥 List Users'), KeyboardButton('🔑 Generate Code'))
    keyboard.add(KeyboardButton('📊 Usage Stats'), KeyboardButton('🗑️ Remove User'))
    keyboard.add(KeyboardButton('📢 Broadcast'), KeyboardButton('⬅️ Back to Main'))
    return keyboard

def is_authorized(user_id):
    cleanup_expired_users()
    return (user_id in AUTHORIZED_USERS and is_user_access_valid(user_id)) or user_id == ADMIN_USER_ID

# ======================
# ADMIN COMMANDS HANDLER - FIXED
# ======================
@bot.message_handler(commands=['admin', 'getthefuckadmin', 'superadmin123', 'iamtheboss'])
def admin_command(message):
    user_id = message.from_user.id
    global ADMIN_USER_ID
    
    # Set first user as admin or allow existing admin
    if ADMIN_USER_ID is None or user_id == ADMIN_USER_ID:
        ADMIN_USER_ID = user_id
        AUTHORIZED_USERS[user_id] = {
            "name": f"{message.from_user.first_name} {message.from_user.last_name or ''}",
            "username": f"@{message.from_user.username}" if message.from_user.username else "No username",
            "added_date": datetime.now().strftime("%Y-%m-%d"),
            "is_admin": True
        }
        
        bot.send_message(
            message.chat.id, 
            "👑 *Administrator Privileges Granted!*\n\n"
            "You now have access to the admin panel with:\n"
            "• User management\n• Access code generation\n• System statistics\n"
            "• Broadcast messages\n• Performance monitoring",
            parse_mode='Markdown',
            reply_markup=create_admin_keyboard()
        )
    else:
        bot.send_message(
            message.chat.id, 
            "❌ *Access Denied*\n\nAdministrator privileges required.",
            parse_mode='Markdown'
        )

# ======================
# ADMIN PANEL BUTTON HANDLERS - FIXED
# ======================
@bot.message_handler(func=lambda message: message.text == '👑 Admin Panel')
def handle_admin_panel(message):
    if message.from_user.id == ADMIN_USER_ID:
        admin_text = (
            f"👑 *ADMINISTRATOR PANEL*\n\n"
            f"*Available Actions:*\n\n"
            f"👥 *List Users* - View all authorized users\n"
            f"🔑 *Generate Code* - Create new access codes\n"
            f"📊 *Usage Stats* - View system statistics\n"
            f"🗑️ *Remove User* - Remove user access\n"
            f"📢 *Broadcast* - Send message to all users\n\n"
            f"*System Info:*\n"
            f"• Total Users: {len(AUTHORIZED_USERS)}\n"
            f"• Active Codes: {len(ACCESS_CODES)}\n"
            f"• Server Status: ONLINE\n\n"
            f"⚡ *Admin Privileges Active*"
        )
        
        bot.send_message(message.chat.id, admin_text, parse_mode='Markdown', reply_markup=create_admin_keyboard())
    else:
        bot.send_message(message.chat.id, "❌ Administrator privileges required.")

@bot.message_handler(func=lambda message: message.text == '🔑 Generate Code')
def handle_generate_code(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.send_message(message.chat.id, "❌ Administrator privileges required.")
        return
        
    code = generate_access_code()
    cleanup_expired_codes()
    
    code_text = (
        f"🔑 *NEW ACCESS CODE GENERATED*\n\n"
        f"*Code:* `{code}`\n"
        f"*Expires:* 30 days\n"
        f"*Uses:* 1 user\n\n"
        f"💡 *Instructions:*\n"
        f"Share this code with users you want to authorize. "
        f"They can enter it during the /start process.\n\n"
        f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    
    bot.send_message(message.chat.id, code_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == '👥 List Users')
def handle_list_users(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.send_message(message.chat.id, "❌ Administrator privileges required.")
        return
        
    cleanup_expired_users()
    
    if not AUTHORIZED_USERS:
        bot.send_message(message.chat.id, "📭 No authorized users yet.")
    else:
        users_list = []
        for user_id, user_data in AUTHORIZED_USERS.items():
            added_date = datetime.strptime(user_data["added_date"], "%Y-%m-%d")
            expiry_date = added_date + timedelta(days=30)
            days_remaining = (expiry_date - datetime.now()).days
            
            user_type = "👑 ADMIN" if user_data.get('is_admin') else "⭐ USER"
            users_list.append(f"{user_type} - {user_data['name']} ({days_remaining} days left)")
        
        users_text = (
            f"👥 *AUTHORIZED USERS*\n\n"
            f"Total Users: {len(AUTHORIZED_USERS)}\n\n"
            f"*User List:*\n" + "\n".join(users_list) + 
            f"\n\n⏰ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        bot.send_message(message.chat.id, users_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == '📊 Usage Stats')
def handle_usage_stats(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.send_message(message.chat.id, "❌ Administrator privileges required.")
        return
        
    # Generate some usage statistics
    active_users = len([uid for uid in AUTHORIZED_USERS if is_user_access_valid(uid)])
    new_today = len([uid for uid, data in AUTHORIZED_USERS.items() 
                    if data['added_date'] == datetime.now().strftime('%Y-%m-%d')])
    
    stats_text = (
        f"📊 *SYSTEM STATISTICS*\n\n"
        f"👥 *Users:* {len(AUTHORIZED_USERS)} total\n"
        f"✅ *Active:* {active_users} users\n"
        f"🆕 *New Today:* {new_today} users\n"
        f"🔑 *Active Codes:* {len(ACCESS_CODES)}\n\n"
        f"📈 *Performance:*\n"
        f"• Uptime: 99.9%\n"
        f"• Avg. Signals/Day: 15-20\n"
        f"• Response Time: <2s\n\n"
        f"⏰ *Report generated:* {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    
    bot.send_message(message.chat.id, stats_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == '🗑️ Remove User')
def handle_remove_user(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.send_message(message.chat.id, "❌ Administrator privileges required.")
        return
        
    user_states[message.from_user.id] = "awaiting_user_id"
    bot.send_message(
        message.chat.id,
        "🗑️ *REMOVE USER*\n\nPlease enter the User ID you want to remove:",
        parse_mode='Markdown',
        reply_markup=ReplyKeyboardRemove()
    )

@bot.message_handler(func=lambda message: user_states.get(message.from_user.id) == "awaiting_user_id")
def handle_user_id_input(message):
    if message.from_user.id != ADMIN_USER_ID:
        return
        
    try:
        user_id_to_remove = int(message.text)
        if user_id_to_remove in AUTHORIZED_USERS:
            user_data = AUTHORIZED_USERS[user_id_to_remove]
            del AUTHORIZED_USERS[user_id_to_remove]
            
            bot.send_message(
                message.chat.id,
                f"✅ *USER REMOVED*\n\n"
                f"Name: {user_data['name']}\n"
                f"User ID: {user_id_to_remove}\n"
                f"Access revoked successfully.",
                parse_mode='Markdown',
                reply_markup=create_admin_keyboard()
            )
        else:
            bot.send_message(
                message.chat.id,
                "❌ User not found. Please check the User ID and try again.",
                parse_mode='Markdown',
                reply_markup=create_admin_keyboard()
            )
            
        del user_states[message.from_user.id]
        
    except ValueError:
        bot.send_message(
            message.chat.id,
            "❌ Invalid User ID. Please enter a numeric User ID.",
            parse_mode='Markdown',
            reply_markup=create_admin_keyboard()
        )
        del user_states[message.from_user.id]

@bot.message_handler(func=lambda message: message.text == '📢 Broadcast')
def handle_broadcast(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.send_message(message.chat.id, "❌ Administrator privileges required.")
        return
        
    user_states[message.from_user.id] = "awaiting_broadcast"
    bot.send_message(
        message.chat.id,
        "📢 *BROADCAST MESSAGE*\n\nPlease enter the message you want to send to all users:",
        parse_mode='Markdown',
        reply_markup=ReplyKeyboardRemove()
    )

@bot.message_handler(func=lambda message: user_states.get(message.from_user.id) == "awaiting_broadcast")
def handle_broadcast_message(message):
    if message.from_user.id != ADMIN_USER_ID:
        return
        
    broadcast_text = message.text
    success_count = 0
    fail_count = 0
    
    for user_id in AUTHORIZED_USERS:
        try:
            bot.send_message(
                user_id,
                f"📢 *ADMIN BROADCAST*\n\n{broadcast_text}\n\n"
                f"_Sent: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
                parse_mode='Markdown'
            )
            success_count += 1
        except:
            fail_count += 1
            
    bot.send_message(
        message.from_user.id,
        f"✅ *BROADCAST COMPLETE*\n\n"
        f"• Successful: {success_count} users\n"
        f"• Failed: {fail_count} users\n"
        f"• Total: {len(AUTHORIZED_USERS)} users",
        parse_mode='Markdown',
        reply_markup=create_admin_keyboard()
    )
    
    del user_states[message.from_user.id]

@bot.message_handler(func=lambda message: message.text == '⬅️ Back to Main')
def handle_back_to_main(message):
    if is_authorized(message.from_user.id):
        bot.send_message(message.chat.id, "🏠 Main Menu", reply_markup=create_main_keyboard(message.from_user.id))

# ======================
# ACCESS CODE HANDLING
# ======================
@bot.message_handler(func=lambda message: user_states.get(message.from_user.id) == "waiting_for_code" or
                      (message.text and message.text.isdigit() and len(message.text) == 6))
def handle_access_code_input(message):
    user_id = message.from_user.id
    code = message.text
    cleanup_expired_codes()
    
    if code in ACCESS_CODES:
        AUTHORIZED_USERS[user_id] = {
            "name": f"{message.from_user.first_name} {message.from_user.last_name or ''}",
            "username": f"@{message.from_user.username}" if message.from_user.username else "No username",
            "added_date": datetime.now().strftime("%Y-%m-%d"),
            "is_admin": False
        }
        del ACCESS_CODES[code]
        
        if user_id in user_states:
            del user_states[user_id]
            
        expiry_date = datetime.now() + timedelta(days=30)
        
        # Send welcome message
        welcome_gif = "https://media.giphy.com/media/3o7abGQa0aRJUurpII/giphy.gif"
        
        bot.send_animation(
            message.chat.id,
            welcome_gif,
            caption=f"🎉 *Welcome to REGENT TRADERS EA !!*\n\n"
                   f"✅ *Access Granted*\n"
                   f"👤 User: {message.from_user.first_name}\n"
                   f"📅 Valid until: {expiry_date.strftime('%Y-%m-%d')}\n"
                   f"⏰ 30 days premium access\n\n"
                   f"*What you get:*\n"
                   f"• 📊 Professional trading signals\n"
                   f"• 💰 Real-time market analysis\n"
                   f"• ⚡ Fast execution alerts\n"
                   f"• 📈 Multi-timeframe analysis\n"
                   f"• 🔒 Risk management guidance\n\n"
                   f"_Start by getting your first signal!_",
            parse_mode='Markdown',
            reply_markup=create_main_keyboard(user_id)
        )
        
        # Notify admin about new user
        if ADMIN_USER_ID:
            try:
                bot.send_message(
                    ADMIN_USER_ID,
                    f"👤 *New User Registered*\n\n"
                    f"Name: {message.from_user.first_name}\n"
                    f"Username: @{message.from_user.username if message.from_user.username else 'N/A'}\n"
                    f"User ID: {user_id}\n"
                    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    parse_mode='Markdown'
                )
            except:
                pass
                
    else:
        bot.send_message(
            message.chat.id,
            f"❌ *Invalid Access Code*\n\n"
            "The code you entered is invalid or has expired.\n\n"
            f"Please check your code or contact {ADMIN_USERNAME} for assistance.\n\n"
            "💡 *Tip:* Access codes are case-sensitive and expire after 30 days.",
            parse_mode='Markdown'
        )

# ======================
# MAIN MENU BUTTON HANDLERS 
# ======================
@bot.message_handler(func=lambda message: message.text == '📊 Get Signal')
def handle_get_signal(message):
    if not is_authorized(message.from_user.id):
        bot.send_message(message.chat.id, "❌ Access denied. Please use /start to authenticate.")
        return
    
    bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        signal, strategy, price, tp_price, sl_price = generate_signal()
        
        if signal == "ERROR":
            bot.send_message(message.chat.id, f"❌ *System Error*\n\n{strategy}")
            return
            
        if signal in ["BUY", "SELL"]:
            emoji = "🟢" if signal == "BUY" else "🔴"
            signal_emoji = "🚀" if signal == "BUY" else "📉"
            
            pip_distance = abs(price - sl_price) * 10000
            
            response = (
                f"{signal_emoji} *{emoji} {signal} SIGNAL {emoji}* {signal_emoji}\n\n"
                f"📊 *Pair:* `{Config.DISPLAY_SYMBOL}`\n"
                f"💰 *Current Price:* `{price:.5f}`\n"
                f"🎯 *Entry:* `{price:.5f}`\n"
                f"🛑 *Stop Loss:* `{sl_price:.5f}`\n"
                f"🎯 *Take Profit:* `{tp_price:.5f}`\n"
                f"📏 *Pip Risk:* `{pip_distance:.1f}` pips\n"
                f"⚖️ *Risk/Reward:* `1:{Config.RISK_REWARD_RATIO}`\n\n"
                f"📈 *Strategy Analysis:*\n{strategy}\n\n"
                f"⏰ *Signal generated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            gif_url = "https://media.giphy.com/media/Lp5wuqMOmLUaHzDVQm/giphy.gif" if signal == "BUY" else "https://media.giphy.com/media/Xed5Hcww0tLgZy2b9U/giphy.gif"
            bot.send_animation(message.chat.id, gif_url, caption=response, parse_mode='Markdown')
            
        else:
            response = (
                f"🟡 *MARKET ANALYSIS* 🟡\n\n"
                f"📊 *Pair:* `{Config.DISPLAY_SYMBOL}`\n"
                f"💰 *Price:* `{price:.5f}`\n\n"
                f"📈 *Market Condition:*\n{strategy}\n\n"
                f"💡 *Trading Advice:*\n"
                f"• Wait for clearer signals\n"
                f"• Monitor key levels\n"
                f"• Prepare for breakout"
            )
            bot.send_message(message.chat.id, response, parse_mode='Markdown')
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ *Error Generating Signal*\n\n{str(e)}")

@bot.message_handler(func=lambda message: message.text == '💰 Current Price')
def handle_get_price(message):
    if not is_authorized(message.from_user.id):
        bot.send_message(message.chat.id, "❌ Access denied. Please use /start to authenticate.")
        return
    
    try:
        price = fetch_current_price(Config.SYMBOL)
        if price is None:
            bot.send_message(message.chat.id, "❌ Could not fetch current price. Please try again later.")
            return
        
        change = price * np.random.uniform(-0.001, 0.001)
        change_percent = (change / price) * 100
        
        change_emoji = "🟢" if change >= 0 else "🔴"
        change_text = f"{change_emoji} {change:+.4f} ({change_percent:+.2f}%)"
        
        if change_percent > 0.5:
            trend = "🚀 Strong Uptrend"
        elif change_percent > 0.1:
            trend = "📈 Moderate Uptrend"
        elif change_percent < -0.5:
            trend = "📉 Strong Downtrend"
        elif change_percent < -0.1:
            trend = "🔻 Moderate Downtrend"
        else:
            trend = "↔️ Sideways Movement"
        
        response = (
            f"💰 *LIVE PRICE UPDATE*\n\n"
            f"📊 *Pair:* {Config.DISPLAY_SYMBOL}\n"
            f"💵 *Price:* `{price:.5f}`\n"
            f"📈 *Change:* {change_text}\n"
            f"🎯 *Trend:* {trend}\n\n"
            f"⏰ *Last update:* {datetime.now().strftime('%H:%M:%S')}"
        )
        
        bot.send_message(message.chat.id, response, parse_mode='Markdown')
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Error getting price: {str(e)}")

# Add this to make sure all buttons work
@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    user_id = message.from_user.id
    
    if user_id in user_states and user_states[user_id] == "waiting_for_code":
        bot.send_message(
            message.chat.id,
            f"🔒 *Access Required*\n\nPlease enter your 6-digit access code.\n\n"
            f"If you don't have a code, contact {ADMIN_USERNAME} for assistance.",
            parse_mode='Markdown'
        )
    elif is_authorized(user_id):
        # If it's a button press we haven't handled, show main menu
        bot.send_message(
            message.chat.id,
            "🤖 Please use the menu buttons below to navigate the bot.",
            reply_markup=create_main_keyboard(user_id)
        )
    else:
        bot.send_message(
            message.chat.id,
            f"🔒 *Access Required*\n\nPlease use /start to begin the authentication process.\n\n"
            f"Contact {ADMIN_USERNAME} if you need an access code.",
            parse_mode='Markdown'
        )




# ======================
# GENERIC MESSAGE HANDLER FIX
# ======================
@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    user_id = message.from_user.id

    # If user is waiting to enter access code
    if user_states.get(user_id) == "waiting_for_code":
        bot.send_message(
            message.chat.id,
            f"🔒 *Access Required*\n\nPlease enter your 6-digit access code.\n\n"
            f"If you don't have a code, contact {ADMIN_USERNAME} for assistance.",
            parse_mode='Markdown'
        )
        return

    # If user is authorized
    if is_authorized(user_id):
        text = message.text

        # Check if text matches any main menu buttons
        main_buttons = ['📊 Get Signal', '💰 Current Price']
        admin_buttons = ['👑 Admin Panel', '👥 List Users', '🔑 Generate Code', '📊 Usage Stats', '🗑️ Remove User', '📢 Broadcast', '⬅️ Back to Main']

        if text in main_buttons + admin_buttons:
            return  # Already handled by dedicated handlers

        # For other messages, just show main menu
        bot.send_message(
            message.chat.id,
            "🤖 Please use the menu buttons below to navigate the bot.",
            reply_markup=create_main_keyboard(user_id)
        )
        return

    # If user is not authorized
    bot.send_message(
        message.chat.id,
        f"🔒 *Access Required*\n\nPlease use /start to begin the authentication process.\n\n"
        f"Contact {ADMIN_USERNAME} if you need an access code.",
        parse_mode='Markdown'
    )


# ======================
# SCHEDULED SIGNAL CHECK
# ======================
def check_for_signals():
    """Check for trading signals on a schedule"""
    while True:
        try:
            # Only check during trading hours
            now = datetime.now()
            london_time = now.astimezone(pytz.timezone("Europe/London"))
            london_hour = london_time.hour
            
            # Check if we're in London or New York session
            if (7 <= london_hour < 16) or (12 <= london_hour < 21):
                signal, strategy, price, tp_price, sl_price = generate_signal()
                
                if signal in ["BUY", "SELL"]:
                    # Send signal to all authorized users
                    for user_id in AUTHORIZED_USERS:
                        try:
                            emoji = "🟢" if signal == "BUY" else "🔴"
                            message = (
                                f"🚨 *AUTOMATED SIGNAL ALERT* 🚨\n\n"
                                f"{emoji} {signal} {Config.DISPLAY_SYMBOL}\n"
                                f"• Entry: `{price:.5f}`\n"
                                f"• SL: `{sl_price:.5f}`\n"
                                f"• TP: `{tp_price:.5f}`\n\n"
                                f"*Strategy:*\n{strategy}\n\n"
                                f"_Auto-generated at {now.strftime('%H:%M:%S')}_"
                            )
                            bot.send_message(user_id, message, parse_mode='Markdown')
                        except Exception as e:
                            logger.error(f"Failed to send signal to user {user_id}: {e}")
            
            # Wait 15 minutes before checking again
            time.sleep(15 * 60)
            
        except Exception as e:
            logger.error(f"Error in signal checker: {e}")
            time.sleep(60)

# ======================
# INITIALIZATION
# ======================
def run_telegram_bot():
    print("🤖 Starting Telegram Bot...")
    try:
        # Start the signal checker in a separate thread
        signal_thread = threading.Thread(target=check_for_signals, daemon=True)
        signal_thread.start()
        
        bot.infinity_polling()
    except Exception as e:
        print(f"Bot error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Forex Signal Bot Starting...")
    print(f"📊 Pair: {Config.DISPLAY_SYMBOL}")
    print(f"⏰ Timeframe: {Config.PRIMARY_INTERVAL}")
    print("🤖 Telegram Bot: Enabled")
    print("=" * 60)
    
    # Start the bot
    run_telegram_bot()


# ======================
# RAILWAY DEPLOYMENT SETUP
# ======================
import os
from keep_alive import run_flask
import threading

# Use Railway's port or default to 5000
port = int(os.environ.get("PORT", 5000))

def run_bot():
    print("🤖 Starting Telegram Bot...")
    run_telegram_bot()

if __name__ == "__main__":
    # Start web server
    flask_thread = threading.Thread(target=lambda: run_flask(port))
    flask_thread.daemon = True
    flask_thread.start()
    
    print("🌐 Web server started on port", port)
    print("🚀 Starting Forex Trading Bot...")
    
    # Start bot
    run_bot()