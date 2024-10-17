import os
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2ODeepLearningEstimator
from pymongo import MongoClient
from apscheduler.schedulers.background import BackgroundScheduler
import pygad
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mean_squared_error
import time
# from google.colab import drive

import gym
from gym import spaces
from stable_baselines3 import PPO
from textblob import TextBlob
from hmmlearn import hmm
import shap
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from forex_python.converter import CurrencyRates
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Initialize H2O cluster
h2o.init(max_mem_size="4G")

# Load CSV files from a folder
def load_csv_files(folder_path):
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    with ThreadPoolExecutor() as executor:
        dataframes = list(executor.map(pd.read_csv, csv_files))
    # dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]
    return dataframes
data_dir = os.path.abspath('C:\\Users\\Baap\\Desktop\\trading bot\\trading_bot\\data')
min1_data = load_csv_files(os.path.join(data_dir, '1min'))
min15_data = load_csv_files(os.path.join(data_dir, '15min'))
min3_data = load_csv_files(os.path.join(data_dir, '5min'))

all_data = {
    '1min': min1_data,
    '15min': min15_data,
    '5min': min3_data
}

print("First 5 rows of 1-min data:")
print(min1_data[0].head())

# MongoDB setup
client = MongoClient('mongodb+srv://aimlvideos7:$xZj8T$SQc2.a88@testdata.loqpv.mongodb.net/')
db = client['demo']
collection = db['sample']

# Store results in MongoDB
def store_results_in_db(strategy, profit, trades, accuracy):
    results = {'strategy': strategy, 'profit': profit, 'trades': trades , 'accuracy':accuracy}
    collection.insert_one(results)
    print(f"Stored in DB: Strategy: {strategy}, Profit: {profit}, Trades: {trades}, accuracy: {accuracy}")

def fetch_live_data(currency_pair='EUR/USD'):
    c = CurrencyRates()
    try:
        rate = c.get_rate(currency_pair.split('/')[0], currency_pair.split('/')[1])
        return rate
    except Exception as e:
        print(f"Error fetching live Forex data: {e}")
        return None
    
    # Combine live and historical data
def combine_live_and_historical(live_rate, historical_data):
    historical_data['Live_Rate'] = live_rate
    combined_data = historical_data.copy()
    combined_data['Close'] = combined_data['Live_Rate'].combine_first(combined_data['Close'])
    return combined_data

def send_email(currency_pair, buy_price, sell_price, buy_time, sell_time, accuracy, bot_name, prediction):
    email_user = 'sagarkhemnar143@gmail.com'
    email_password = 'vbuh wjod dlcg wsag'  # Use your app-specific password
    email_send = 'coderd60@gmail.com'

    subject = f'Trade Signal for {currency_pair} - {bot_name} Bot'
    body = f"""
    Trade Signal for {currency_pair}:\n
    - **Bot Name**: {bot_name}
    - **Prediction**: {'Buy' if prediction == 1 else 'Hold'}
    - **Buy Price**: {buy_price}
    - **Sell Price**: {sell_price}
    - **Buy Time**: {buy_time}
    - **Sell Time**: {sell_time}
    - **Accuracy**: {accuracy * 100:.2f}%\n
    Please review the trade signal.
    """

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(email_user, email_password)
        text = msg.as_string()
        server.sendmail(email_user, email_send, text)
        server.quit()
        print(f"Email sent successfully with subject: {subject}")
    except Exception as e:
        print(f"Error sending email: {e}")

    

# def send_email(prediction):
#     email_user = 'sagarkhemnar143@gmail.com'
#     email_password = 'vbuh wjod dlcg wsag'  # Use your app-specific password
#     email_send = 'coderd60@gmail.com'

#     subject = 'EUR/USD Trade Signal Prediction'
#     body = f"The predicted signal for the next 5 minutes is: {'Buy' if prediction == 1 else 'Hold'}"

#     msg = MIMEMultipart()
#     msg['From'] = email_user
#     msg['To'] = email_send
#     msg['Subject'] = subject
#     msg.attach(MIMEText(body, 'plain'))

#     try:
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.starttls()  # Secure the connection
#         server.login(email_user, email_password)
#         text = msg.as_string()
#         server.sendmail(email_user, email_send, text)
#         server.quit()
#         print("Email sent successfully!")
#     except Exception as e:
#         print(f"Error sending email: {e}")


# Exponential Moving Average (EMA) trend bot
def ema_trend_bot(df, short_window=50, long_window=200):
    df['Short_EMA'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=long_window, adjust=False).mean()

    df['Signal'] = 0
    df.loc[df['Short_EMA'] > df['Long_EMA'], 'Signal'] = 1
    df.loc[df['Short_EMA'] < df['Long_EMA'], 'Signal'] = -1

    recent_signal = df['Signal'].iloc[-1]
    send_email(recent_signal)

    print("EMA Trend Bot signals:")
    print(df[['Short_EMA', 'Long_EMA', 'Signal']].tail())

    return df

# Scheduler to run the combined strategy every 30 minutes
def run_combined_strategy():
    live_rate = fetch_live_data()
    for df in all_data['1min']:
        combined_df = combine_live_and_historical(live_rate, df)
        ema_trend_bot(combined_df)

scheduler = BackgroundScheduler()
scheduler.add_job(run_combined_strategy, 'interval', minutes=30)
scheduler.start()

def train_model_parallel(df):
    df = ema_trend_bot(df)
    h2o_df = h2o.H2OFrame(df)
    dl_model = H2ODeepLearningEstimator()

    dl_model.train(x=['Open', 'High', 'Low', 'Volume'], y='Close', training_frame=h2o_df)

    # Save the model
    model_path = h2o.save_model(model=dl_model, path='data/models', force=True)
    print(f"Model saved at: {model_path}")
    return model_path

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(train_model_parallel, df) for df in min1_data]
    for future in as_completed(futures):
        print(f"Training completed. Model saved at: {future.result()}")

# Feature Engineering
def add_technical_indicators(df):
    # Adding indicators: SMA, EMA, RSI
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], period=14)

    # Adding new indicators: MACD, Bollinger Bands
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
    df['BB_Upper'], df['BB_Lower'], df['BB_Mid'] = compute_bollinger_bands(df['Close'])

    return df

# Compute MACD function
def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist

# Compute Bollinger Bands function
def compute_bollinger_bands(series, window=20, no_of_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * no_of_std)
    lower_band = rolling_mean - (rolling_std * no_of_std)
    return upper_band, lower_band, rolling_mean

# Compute RSI function
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

    # New features (Trend Analysis, Support/Resistance, etc.)
def trend_analysis(df):
    # ADX for trend strength
    df['ADX'] = compute_adx(df)
    df['Trend'] = 'Sideways'
    df.loc[df['ADX'] > 20, 'Trend'] = 'Uptrend'
    df.loc[df['ADX'] < 20, 'Trend'] = 'Downtrend'
    print(f"Trend Analysis:\n{df[['ADX', 'Trend']].tail()}")

    return df

    # Additional Trading Strategies
def ichimoku_trend_bot(df):
    # Calculate Ichimoku components
    df['Tenkan_Sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Kijun_Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou_Span'] = df['Close'].shift(-26)

    # Generate trading signals based on Ichimoku cloud
    df['Ichimoku_Signal'] = 0
    df.loc[df['Close'] > df['Senkou_Span_A'], 'Ichimoku_Signal'] = 1  # Buy signal
    df.loc[df['Close'] < df['Senkou_Span_B'], 'Ichimoku_Signal'] = -1  # Sell signal

    print("Ichimoku Trend Bot signals:")
    print(df[['Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span', 'Ichimoku_Signal']].tail())

    return df

def range_breakout_bot(df, lookback=20):
    df['Range_High'] = df['High'].rolling(window=lookback).max()
    df['Range_Low'] = df['Low'].rolling(window=lookback).min()

    df['Signal'] = 0
    df.loc[df['Close'] > df['Range_High'], 'Signal'] = 1  # Buy signal
    df.loc[df['Close'] < df['Range_Low'], 'Signal'] = -1  # Sell signal

    print("Range Breakout Bot signals:")
    print(df[['Range_High', 'Range_Low', 'Signal']].tail())

    return df

    # Performance Metrics Calculation
def calculate_performance_metrics(df):
    # Calculate performance metrics such as accuracy, profit, drawdown
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)

    total_return = df['Strategy_Return'].sum()
    num_trades = df['Signal'].diff().ne(0).sum() / 2  # Count the number of trades (Buy & Sell)
    accuracy = (df['Strategy_Return'] > 0).mean()  # Calculate accuracy

    print(f"Total Return: {total_return}, Number of Trades: {num_trades}, Accuracy: {accuracy}")
    return total_return, num_trades, accuracy
    

def compute_adx(df, period=14):
    # Calculate the ADX (Average Directional Index)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['DM+'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
    df['DM-'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
    df['TR'] = df['TR'].rolling(window=period).sum()
    df['DM+'] = df['DM+'].rolling(window=period).sum()
    df['DM-'] = df['DM-'].rolling(window=period).sum()
    df['DI+'] = 100 * (df['DM+'] / df['TR'])
    df['DI-'] = 100 * (df['DM-'] / df['TR'])
    df['DX'] = 100 * abs(df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-'])
    return df['DX'].rolling(window=period).mean()


    # Sentiment Analysis bot using TextBlob
def sentiment_analysis_bot(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0:
        sentiment = 'Positive'
    elif sentiment_score < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    print(f"Text: {text}\nSentiment: {sentiment} (Score: {sentiment_score})")
    return sentiment

# Execute all bots
def execute_trading_strategies(data_dict):
    for interval, dfs in data_dict.items():
        for df in dfs:
            print(f"Processing data for {interval} interval:")
            ema_trend_bot(df)
            ichimoku_trend_bot(df)
            trend_analysis(df)

# Run strategy execution in the background every minute
scheduler = BackgroundScheduler()
scheduler.add_job(execute_trading_strategies, 'interval', minutes=1, args=[all_data])
scheduler.start()

print("Trading bot running...")
    # New Bot with combined strategies
def combined_strategy_bot(df):
    df = ema_trend_bot(df)
    df = trend_analysis(df)
    df = add_technical_indicators(df)
    df = ichimoku_trend_bot(df)
    df = range_breakout_bot(df)

    print("Combined Strategy signals:")
    print(df[['Signal']].tail())

    return df

# Reinforcement Learning Environment
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index()
        self.current_step = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.df.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = 0  # Define your reward function
        info = {}
        if not done:
            obs = self.df.iloc[self.current_step].values
        else:
            obs = self.df.iloc[self.current_step].values
        return obs, reward, done, info

def train_rl_model(df):
    env = TradingEnv(df)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model


    # Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def fetch_news_headlines():
    # Placeholder for fetching data
    # Replace with actual data fetching logic
    return ["Market is bullish today!", "Economic slowdown expected."]

def add_sentiment(df):
    headlines = fetch_news_headlines()
    sentiments = [get_sentiment(headline) for headline in headlines]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    df['Sentiment'] = avg_sentiment
    return df

  # Regime-Switching Models
def detect_regimes(df):
    # Prepare data for HMM
    X = df[['Close', 'Volume']].values  # Features can be adjusted
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)
    model.fit(X)
    regimes = model.predict(X)
    df['Regime'] = regimes
    return df


    # Dynamic Position Sizing (Kelly Criterion)
def kelly_criterion(win_prob, avg_win, avg_loss):
    return win_prob - (1 - win_prob) / (avg_win / avg_loss)

def apply_kelly(df):
    # Placeholder calculations
    win_prob = 0.6  # Example win probability
    avg_win = df[df['Signal'] == 1]['Close'].pct_change().mean()
    avg_loss = -df[df['Signal'] == -1]['Close'].pct_change().mean()
    kelly_fraction = kelly_criterion(win_prob, avg_win, avg_loss)
    df['Position_Size'] = kelly_fraction
    return df

# Explainability with SHAP

def explain_model(model, df):
    # Assuming the model is an H2O model
    h2o_df = h2o.H2OFrame(df)
    shap_explainer = shap.Explainer(model)
    shap_values = shap_explainer(h2o_df)
    shap.summary_plot(shap_values, df)
    return shap_values


    # Volatility-Based Strategy Adaptation
def add_volatility_indicators(df):
    df['ATR'] = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
    df['Volatility_Signal'] = 0
    df.loc[df['ATR'] > df['ATR'].median(), 'Volatility_Signal'] = 1  # High volatility
    df.loc[df['ATR'] <= df['ATR'].median(), 'Volatility_Signal'] = -1  # Low volatility
    return df

# Transfer Learning and Meta-Learning
def transfer_model(source_model_path, target_currency_data):
    # Load source model
    source_model = h2o.load_model(source_model_path)

    # Transfer learning: Fine-tune the model on target data
    target_h2o_df = h2o.H2OFrame(target_currency_data)
    source_model.train(x=['Open', 'High', 'Low', 'Volume'], y='Close', training_frame=target_h2o_df, fine_tuning=True)

    # Save the fine-tuned model
    transfer_model_path = 'data/models/transfer_model.zip'
    h2o.save_model(model=source_model, path='data/models', force=True)
    print(f"Transfer Learning Model saved at: {transfer_model_path}")
    return transfer_model_path

    # Improved Backtesting Framework with Walk-Forward Optimization
def walk_forward_optimization(data, model_func):
    window_size = int(0.8 * len(data))
    train, test = data[:window_size], data[window_size:]
    model = model_func(train)
    predictions = model.predict(h2o.H2OFrame(test))
    mse = mean_squared_error(test['Close'].values, predictions.as_data_frame()['predict'].values)
    print(f"Walk-Forward MSE: {mse}")
    return mse

def improved_backtest_model(model_func, df):
    mse = walk_forward_optimization(df, model_func)
    return mse

# AI-Based Risk Management: Value at Risk (VaR)
def value_at_risk(returns, confidence_level=0.95):
    return np.percentile(returns, 100 * (1 - confidence_level))

def apply_var(df):
    df['Returns'] = df['Close'].pct_change()
    var_95 = value_at_risk(df['Returns'].dropna(), confidence_level=0.95)
    df['VaR_95'] = var_95
    return df



# Example: Apply combined strategy bot on 1-min data
for df in min1_data:
    df = combined_strategy_bot(df)

# Schedule the EMA trend bot to run every minute with additional features
def run_ema_trend():
    for df in all_data['1min']:
        df = ema_trend_bot(df)
        df = add_sentiment(df)
        df = detect_regimes(df)
        df = apply_kelly(df)
        df = add_volatility_indicators(df)
        df = apply_var(df)
        # Example profit, trades, and accuracy calculation
        profit = np.random.randint(500, 1500)
        trades = np.random.randint(10, 100)
        accuracy = np.random.uniform(0.5, 0.9)
        store_results_in_db('EMA Trend', profit, trades, accuracy)

# Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(run_ema_trend, 'interval', minutes=1)
scheduler.start()

# Keep the script running
try:
    while True:
        time.sleep(2)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()


# AutoML Integration with SHAP and Portfolio Optimization
def run_h2o_automl(df):
    target = 'Close'
    features = df.columns.tolist()
    features.remove(target)
    features.remove('Sentiment')
    features.remove('Regime')
    features.remove('Position_Size')
    features.remove('ATR')
    features.remove('Volatility_Signal')
    features.remove('VaR_95')
    # Add any additional features you want to include

    h2o_df = h2o.H2OFrame(df)
    h2o_df[target] = h2o_df[target].asfactor()  # Convert target to factor if classification

    train, test = h2o_df.split_frame(ratios=[0.8])

    aml = H2OAutoML(max_runtime_secs=7200, seed=1)
    aml.train(x=features, y=target, training_frame=train)

    # View leaderboard
    print(aml.leaderboard)

    # Evaluate performance on test data
    perf = aml.leader.model_performance(test)
    print(perf)

    # Explain model with SHAP
    shap_values = explain_model(aml.leader, df)

    return aml.leader

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_h2o_automl, df) for df in min1_data + min15_data + min3_data]
    for future in as_completed(futures):
        print(f"AutoML model training completed: {future.result()}")

# Backtest the model with Improved Framework
def backtest_model(model_func, df):
    mse = walk_forward_optimization(df, model_func)
    print(f"Walk-Forward Backtest MSE: {mse}")
    return mse

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(backtest_model, run_h2o_automl, df) for df in min1_data + min15_data + min3_data]
    for future in as_completed(futures):
        print(f"Walk-Forward Backtest completed. MSE: {future.result()}")

# Genetic algorithm for optimization
def fitness_function(solution, solution_idx):
    # Define your actual fitness calculation based on strategy performance
    return np.random.rand()  # Placeholder

ga_instance = pygad.GA(num_generations=100, num_parents_mating=5, fitness_func=fitness_function)
ga_instance.run()
optimized_solution = ga_instance.best_solution()

print("Best solution found by GA:")
print(optimized_solution)


# Example: Run Ichimoku bot
for df in min1_data:
    df = ichimoku_trend_bot(df)

# Example: Run Range Breakout bot
for df in min1_data:
    df = range_breakout_bot(df)

# Plotting results with additional indicators
def plot_signals(df):
    plt.figure(figsize=(15, 8))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['Short_EMA'], label='Short EMA', color='red')
    plt.plot(df['Long_EMA'], label='Long EMA', color='green')
    plt.plot(df['BB_Upper'], label='Bollinger Upper', color='cyan', linestyle='--')
    plt.plot(df['BB_Lower'], label='Bollinger Lower', color='cyan', linestyle='--')
    plt.scatter(df.index, df[df['Signal'] == 1]['Close'], label='Buy Signal', marker='^', color='green')
    plt.scatter(df.index, df[df['Signal'] == -1]['Close'], label='Sell Signal', marker='v', color='red')
    plt.title('Trading Signals with Technical Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example: Plot signals for the first 1-min dataframe
plot_signals(min1_data[0])
