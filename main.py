import numpy as np
import requests
import pandas as pd
import time
import random
import statsmodels.api as sm
from datetime import datetime

import subprocess
# from price_module import get_symbol_price

SYMBOL = 'ETHUSDT'
LIMIT = 20
INTERVAL = '1m'

def get_random_user_agent():
    os_version = f"Windows NT 10.0; Win64; x64"
    webkit_version = f"AppleWebKit/{random.randint(500, 600)}.{random.randint(10, 99)}"
    chrome_version = f"Chrome/{random.randint(100, 150)}.{random.randint(0, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
    safari_version = f"Safari/{random.randint(500, 600)}.{random.randint(10, 99)}"
    user_agent = f"Mozilla/5.0 ({os_version}) {webkit_version} (KHTML, like Gecko) {chrome_version} {safari_version}"
    return user_agent

def get_futures_klines(SYMBOL, INTERVAL, LIMIT):
    user_agent = get_random_user_agent()
    headers = {
        "User-Agent": user_agent,
    }
    endpoint = f'https://fapi.binance.com/fapi/v1/klines'
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": LIMIT
    }

    try:
        response = requests.get(url=endpoint, headers=headers, params=params)
        response.raise_for_status()  # Check the response status
        data = response.json()
    except requests.RequestException as e:
        print(f"Error getting klines data: {e}")
        return None

    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df = df.drop(['quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1)
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df

def get_symbol_price(SYMBOL, interval='1m', limit=1):
    headers = {
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "user-agent": get_random_user_agent(),
    }

    try:
        # Используем метод, соответствующий вашим требованиям
        url = f'https://fapi.binance.com/fapi/v1/klines?symbol={SYMBOL}&interval={interval}&limit={limit}'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Извлекаем текущую цену из ответа
        current_price = float(data[0][4])

        #Записываем текущую цену в файл price.txt
        with open('price.txt', 'a+') as price_file:
            current_time = time.strftime('%H:%M:%S')  # Изменено на сохранение только времени
            current_entry = f"{current_time} {current_price}\n"
            # Записываем новую запись в конец файла
            price_file.write(current_entry)
            # Сбрасываем указатель файла и читаем все записи
            price_file.seek(0)
            entries = price_file.readlines()
            # Проверяем количество записей
            if len(entries) >= 100:
                # Если записей больше 100, обрезаем список до последних 100 записей
                entries = entries[-100:]
            # Сбрасываем указатель файла в начало и перезаписываем все записи
            price_file.seek(0)
            price_file.truncate()
            # Записываем все записи обратно в файл
            price_file.writelines(entries)

        return current_price
    except requests.RequestException as e:
        print(f"Error getting price for symbol {SYMBOL}: {e}")
        return None

def indSlope(series, n):
    array_sl = [0 for _ in range(n - 1)]
    for j in range(n, len(series) + 1):
        y = series[j - n:j]
        x = np.array(range(n))
        x_sc = (x - x.min()) / (x.max() - x.min())
        y_sc = (y - y.min()) / (y.max() - y.min())
        x_sc = sm.add_constant(x_sc)
        model = sm.OLS(y_sc, x_sc)
        results = model.fit()
        array_sl.append(results.params.iloc[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(array_sl))))
    return np.array(slope_angle)

def indATR(source_DF, n):
    df = source_DF.copy()
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df.drop(['H-L', 'H-PC', 'L-PC'], axis=1, inplace=True)
    return df

def isLCC(DF, i):
    df = DF.copy()
    LCC = 0

    if df['close'][i] <= df['close'][i + 1] and df['close'][i] <= df['close'][i - 1] and df['close'][i + 1] > \
            df['close'][i - 1]:
        # найдено Дно
        LCC = i - 1;
    return LCC

def isHCC(DF, i):
    df = DF.copy()
    HCC = 0
    if df['close'][i] >= df['close'][i + 1] and df['close'][i] >= df['close'][i - 1] and df['close'][i + 1] < \
            df['close'][i - 1]:
        # найдена вершина
        HCC = i;
    return HCC

import datetime
def PrepareDF(DF, n=12, atr_period=14, slope_period=5):
    ohlc = DF.iloc[:, [0, 1, 2, 3, 4, 5]]
    ohlc.columns = ["date", "open", "high", "low", "close", "volume"]
    ohlc = ohlc.set_index('date')

    # Вычисляем максимум и минимум канала
    maxx, minn = getMaxMinChannel(ohlc, n)

    df = indATR(ohlc, atr_period).reset_index()
    df['slope'] = indSlope(df['close'], slope_period)
    df['channel_max'] = maxx  # Максимум канала
    df['channel_min'] = minn  # Минимум канала
    df['position_in_channel'] = (df['close'] - minn) / (maxx - minn)
    df = df.set_index('date')
    df = df.reset_index()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # Сохраняем slope в slope.txt
    # Сохраняем slope в slope.txt
    with open('slope.txt', 'a+') as slope_file:
        current_time = time.strftime('%H:%M:%S')
        current_entry = f"{current_time} {df['slope'].iloc[-1]}\n"
        # Записываем новую запись в конец файла
        slope_file.write(current_entry)
        # Сбрасываем указатель файла и читаем все записи
        slope_file.seek(0)
        entries = slope_file.readlines()
        # Проверяем количество записей
        if len(entries) >= 100:
            # Если записей больше 100, обрезаем список до последних 100 записей
            entries = entries[-100:]
        # Сбрасываем указатель файла в начало и перезаписываем все записи
        slope_file.seek(0)
        slope_file.truncate()
        # Записываем все записи обратно в файл
        slope_file.writelines(entries)
    # Сохраняем position_in_channel в position_in_channel.txt
    with open('position_in_channel.txt', 'a+') as pic_file:
        current_time = time.strftime('%H:%M:%S')
        current_entry = f"{current_time} {df['position_in_channel'].iloc[-1]}\n"
        # Записываем новую запись в конец файла
        pic_file.write(current_entry)
        # Сбрасываем указатель файла и читаем все записи
        pic_file.seek(0)
        entries = pic_file.readlines()
        # Проверяем количество записей
        if len(entries) >= 100:
            # Если записей больше 100, обрезаем список до последних 100 записей
            entries = entries[-100:]
        # Сбрасываем указатель файла в начало и перезаписываем все записи
        pic_file.seek(0)
        pic_file.truncate()
        # Записываем все записи обратно в файл
        pic_file.writelines(entries)

    return df

def getMaxMinChannel(DF, n, maxx=0, minn=None):
    if minn is None:
        minn = DF['low'].max()
    for i in range(1, min(n, len(DF))):
        if maxx < DF['high'].values[-i]:
            maxx = DF['high'].values[-i]
        if minn > DF['low'].values[-i]:
            minn = DF['low'].values[-i]
    return maxx, minn

def time_to_seconds(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def check_if_signal(SYMBOL, current_time, current_price):
    ohlc = get_futures_klines(SYMBOL, INTERVAL, LIMIT)

    if ohlc is not None:
        prepared_df = PrepareDF(ohlc)

        if 'position_in_channel' not in prepared_df.columns or 'date' not in prepared_df.columns:
            print("Error: Incorrect column names in the DataFrame.")
            return None

        signals = []
        i = len(prepared_df) - 1

        try:
            with open('position_in_channel.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    last_positions = [float(line.split()[1]) for line in lines[-3:]]
                    start_index_positions = len(lines) - 3
                    end_index_positions = len(lines)
                    print("Last positions:", last_positions)
                    print("Indexes:", list(range(start_index_positions, end_index_positions)))
                else:
                    print("Error: Insufficient data in position_in_channel.txt.")
                    return None

        except FileNotFoundError:
            print("Error: position_in_channel.txt not found.")
            return None

        try:
            with open('price.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    last_prices = [float(line.split()[1]) for line in lines[-3:]]
                    start_index_prices = len(lines) - 3
                    end_index_prices = len(lines)
                    print("Last prices:", last_prices)
                    print("Indexes:", list(range(start_index_prices, end_index_prices)))
                else:
                    print("Error: Insufficient data in price.txt.")
                    return None

        except FileNotFoundError:
            print("Error: price.txt not found.")
            return None

        if (
                len(last_positions) >= 99 and
                all(position > 0.85 for position in last_positions[97:99]) and
                last_positions[98] > 0.95 and
                last_positions[97] <= last_positions[98] and
                last_positions[98] >= last_positions[99] and
                last_prices[97] <= last_prices[98] and
                last_prices[98] >= current_price
        ):
            signals.append(('short', current_time))

        if (
                len(last_positions) >= 99 and
                all(position < 0.15 for position in last_positions[97:99]) and
                last_positions[98] < 0.05 and
                last_positions[97] >= last_positions[98] and
                last_positions[98] <= last_positions[99] and
                last_prices[97] >= last_prices[98] and
                last_prices[98] <= current_price
        ):
            signals.append(('long', current_time))

        return signals
    else:
        print("Error getting klines data. Check your internet connection.")
        return None

def main(SYMBOL):
    INTERVAL = '1m'  # Установите интервал на '1m'
    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        current_price = get_symbol_price(SYMBOL)  # Assuming get_symbol_price is defined
        print('current_price:', current_price)
        signals = check_if_signal(SYMBOL, current_time, current_price)

        if signals:
            print(f"Signal received: {signals}")
            print(f"Time: {current_time}")
            print(f"Current Symbol Price: {current_price}")

            with open("signal.txt", "a") as file:
                file.write(f"{current_time} - Signal: {signals}, Symbol Price: {current_price}\n")

        else:
            print("No signal at the moment.")
        time.sleep(60)  # Подождать 1 минуту (60 секунд)

if __name__ == "__main__":
    main(SYMBOL)