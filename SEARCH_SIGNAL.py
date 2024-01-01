import numpy as np
import requests
import pandas as pd
import datetime
import time
import random
import ta
import statsmodels.api as sm
from datetime import datetime
import subprocess
# from price_module import get_symbol_price

SYMBOL = 'ETHUSDT'
LIMIT = 100
INTERVAL = '1m'
signal_PSM = []


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
        current_price = round(float(data[0][4]),4)

        #Записываем текущую цену в файл price.txt
        with open('price.txt', 'a+') as price_file:
            current_time = time.strftime('%H:%M:%S')
            current_entry = f"{current_time} {current_price}\n"

            # Записываем новую запись в конец файла
            price_file.write(current_entry)
            # Сбрасываем указатель файла и читаем все записи
            price_file.seek(0)
            entries = price_file.readlines()
            # Проверяем количество записей
            if len(entries) >= 500:
                # Если записей больше 100, обрезаем список до последних 100 записей
                entries = entries[-500:]
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
    # print('slope angle =', slope_angle)
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
    print(LCC)
    return LCC

def isHCC(DF, i):
    df = DF.copy()
    HCC = 0
    if df['close'][i] >= df['close'][i + 1] and df['close'][i] >= df['close'][i - 1] and df['close'][i + 1] < \
            df['close'][i - 1]:
        # найдена вершина
        HCC = i;
    print(HCC)
    return HCC

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

    # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Сохраняем slope в slope.txt
    with open('slope.txt', 'a+') as slope_file:
        current_time = time.strftime('%H:%M:%S')
        current_entry = f"{current_time} { round( df['slope'].iloc[-1],4 ) }\n"
        print('Slope:', round( df['slope'].iloc[-1],4 ))
        # Записываем новую запись в конец файла
        slope_file.write(current_entry)
        # Сбрасываем указатель файла и читаем все записи
        slope_file.seek(0)
        entries = slope_file.readlines()
        # Проверяем количество записей
        if len(entries) >= 500:
            # Если записей больше 500, обрезаем список до последних 100 записей
            entries = entries[-500:]
        # Сбрасываем указатель файла в начало и перезаписываем все записи
        slope_file.seek(0)
        slope_file.truncate()
        # Записываем все записи обратно в файл
        slope_file.writelines(entries)
    # Сохраняем position_in_channel в position_in_channel.txt
    with open('position_in_channel.txt', 'a+') as pic_file:
        current_time = time.strftime('%H:%M:%S')
        current_entry = f"{current_time} { round( df['position_in_channel'].iloc[-1], 4)}\n"
        print('Position_in_channel:', round( df['position_in_channel'].iloc[-1], 4))
        # Записываем новую запись в конец файла
        pic_file.write(current_entry)
        # Сбрасываем указатель файла и читаем все записи
        pic_file.seek(0)
        entries = pic_file.readlines()
        # Проверяем количество записей
        if len(entries) >= 500:
            # Если записей больше 500, обрезаем список до последних 100 записей
            entries = entries[-500:]
        # Сбрасываем указатель файла в начало и перезаписываем все записи
        pic_file.seek(0)
        pic_file.truncate()
        # Записываем все записи обратно в файл
        pic_file.writelines(entries)
    # Сохраняем volume в volume.txt
    with open('volume.txt', 'a+') as volume_file:
        current_time = time.strftime('%H:%M:%S')
        current_entry = f"{current_time} { round(df['volume'].iloc[-1], 4)}\n"

        # Записываем новую запись в конец файла
        volume_file.write(current_entry)
        # Сбрасываем указатель файла и читаем все записи
        volume_file.seek(0)
        entries = volume_file.readlines()
        # Проверяем количество записей
        if len(entries) >= 500:
            # Если записей больше 500, обрезаем список до последних 100 записей
            entries = entries[-100:]
        # Сбрасываем указатель файла в начало и перезаписываем все записи
        volume_file.seek(0)
        volume_file.truncate()
        # Записываем все записи обратно в файл
        volume_file.writelines(entries)

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

def calculate_bollinger_bands(df, length=10, multiplier=2):
    # Calculate rolling mean and standard deviation
    df['rolling_mean'] = df['close'].rolling(window=length).mean()
    df['rolling_std'] = df['close'].rolling(window=length).std()

    # Calculate Bollinger Bands
    df['bb_upper'] = df['rolling_mean'] + (multiplier * df['rolling_std'])
    df['bb_lower'] = df['rolling_mean'] - (multiplier * df['rolling_std'])

    # Calculate Bollinger Bands percent
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

    # Drop NaN values introduced by rolling calculations
    df = df.dropna()

    # Print and return the current Bollinger Bands percent
    current_bb_percent = round(df['bb_percent'].iloc[-1] , 4)
    # print(f"Bollinger Bands Percent: {current_bb_percent}")
    return current_bb_percent

def get_symbol_price_with_bollinger_bands(SYMBOL, INTERVAL, LIMIT, bb_length=10, bb_multiplier=2):
    headers = {
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "user-agent": get_random_user_agent(),
    }
    try:
        # Получаем котировки
        df = get_futures_klines(SYMBOL, INTERVAL, LIMIT)

        # Если удалось получить данные
        if df is not None:
            # Выводим структуру полученного DataFrame
            # print(df)

            # Вычисляем Bollinger Bands и процент Bollinger Bands
            current_bb_percent = calculate_bollinger_bands(df, length=bb_length, multiplier=bb_multiplier)

            # Записываем текущую цену в файл price.txt
            with open('current_bb_percent.txt', 'a+') as price_file:
                current_time = time.strftime('%H:%M:%S')
                current_entry = f"{current_time} {current_bb_percent}\n"

                # Записываем новую запись в конец файла
                price_file.write(current_entry)
                # Сбрасываем указатель файла и читаем все записи
                price_file.seek(0)
                entries = price_file.readlines()
                # Проверяем количество записей
                if len(entries) >= 500:
                    # Если записей больше 100, обрезаем список до последних 100 записей
                    entries = entries[-500:]
                # Сбрасываем указатель файла в начало и перезаписываем все записи
                price_file.seek(0)
                price_file.truncate()
                # Записываем все записи обратно в файл
                price_file.writelines(entries)

            # Выводим текущее значение процента Bollinger Bands на печать
            # print(f"Текущее значение процента Bollinger Bands: {current_bb_percent}")
    except requests.RequestException as e:
        print(f"Ошибка при получении цены для символа {SYMBOL}: {e}")

def check_if_signal(SYMBOL, current_time, current_price):
    global signal_PSM
    ohlc = get_futures_klines(SYMBOL, INTERVAL, LIMIT)

    if ohlc is not None:
        prepared_df = PrepareDF(ohlc)

        if 'position_in_channel' not in prepared_df.columns or 'date' not in prepared_df.columns:
            print("Error: Incorrect column names in the DataFrame.")
            return None

        signals = []
        i = len(prepared_df) - 1

        try:
            with open('slope.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) >= 12:
                    last_slope = [float(line.split()[1]) for line in lines[-12:]]
                    start_index_positions = len(lines) - 12
                    end_index_positions = len(lines)
                    angle_slope = round (np.polyfit(range(len(last_slope)), last_slope, 1)[0], 4)

                    with open('angle_slope.txt', 'a+') as price_file:
                        current_entry = f"{current_time} {angle_slope}\n"
                        price_file.write(current_entry)
                        price_file.seek(0)
                        entries = price_file.readlines()
                        if len(entries) >= 500:
                            entries = entries[-500:]
                        price_file.seek(0)
                        price_file.truncate()
                        price_file.writelines(entries)

                        # print('Angle slope:', angle_slope)
                        # print("Last slope:", last_slope[-3:])

                if len(lines) >= 23:
                    last_slope = [float(line.split()[1]) for line in lines[-23:]]
                    start_index_positions = len(lines) - 23
                    end_index_positions = len(lines)
                    a_slope23 = round( np.polyfit(range(len(last_slope)), last_slope, 1)[0],4)

                    with open('a_slope23.txt', 'a+') as price_file:
                        current_entry = f"{current_time} {a_slope23}\n"
                        price_file.write(current_entry)
                        price_file.seek(0)
                        entries = price_file.readlines()
                        if len(entries) >= 500:
                            entries = entries[-500:]
                        price_file.seek(0)
                        price_file.truncate()
                        price_file.writelines(entries)

                        # print('Angle a_slope23:', a_slope23)

                else:
                    print("Error: Insufficient data in angle_slope.txt.")
                    return None
        except FileNotFoundError:
            print("Error: slope.txt not found.")
            return None
        try:
            with open('position_in_channel.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) >= 12:
                    last_positions = [float(line.split()[1]) for line in lines[-12:]]
                    start_index_positions = len(lines) - 12
                    end_index_positions = len(lines)
                    angle_positions = round (np.polyfit(range(len(last_positions)), last_positions, 1)[0], 4)

                    with open('angle_positions.txt', 'a+') as posit_file:
                        current_entry = f"{current_time} {angle_positions}\n"
                        posit_file.write(current_entry)
                        posit_file.seek(0)
                        entries = posit_file.readlines()
                        if len(entries) >= 500:
                            entries = entries[-500:]
                        posit_file.seek(0)
                        posit_file.truncate()
                        posit_file.writelines(entries)

                    # print('Angle positions:', angle_positions)
                    # print("Last positions:", last_positions[-3:])
                if len(lines) >= 23:
                    last_positions = [float(line.split()[1]) for line in lines[-23:]]
                    start_index_positions = len(lines) - 23
                    end_index_positions = len(lines)
                    a_pos23 = round (np.polyfit(range(len(last_positions)), last_positions, 1)[0],4)

                    with open('a_pos23.txt', 'a+') as posit_file:
                        current_entry = f"{current_time} {a_pos23}\n"
                        posit_file.write(current_entry)
                        posit_file.seek(0)
                        entries = posit_file.readlines()
                        if len(entries) >= 500:
                            entries = entries[-500:]
                        posit_file.seek(0)
                        posit_file.truncate()
                        posit_file.writelines(entries)

                    # print('Angle a_pos23:', a_pos23)

                else:
                    print("Error: Insufficient data in position_in_channel.txt.")
                    return None
        except FileNotFoundError:
            print("Error: position_in_channel.txt not found.")
            return None
        try:
            with open('price.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) >= 12:
                    last_prices = [float(line.split()[1]) for line in lines[-12:]]
                    start_index_prices = len(lines) -12
                    end_index_prices = len(lines)
                    angle_prices = round (np.polyfit(range(len(last_prices)), last_prices, 1)[0], 4)

                    with open('angle_price.txt', 'a+') as price_file:
                        current_entry = f"{current_time} {angle_prices}\n"
                        price_file.write(current_entry)
                        price_file.seek(0)
                        entries = price_file.readlines()
                        if len(entries) >= 500:
                            entries = entries[-500:]
                        price_file.seek(0)
                        price_file.truncate()
                        price_file.writelines(entries)

                    # print('Angle prices:', angle_prices)
                    # print("Last prices:", last_prices[-3:])

                    if len(lines) >= 23:
                        last_prices = [float(line.split()[1]) for line in lines[-23:]]
                        start_index_prices = len(lines) - 23
                        end_index_prices = len(lines)
                        a_prices23 = round (np.polyfit(range(len(last_prices)), last_prices, 1)[0],4)

                        with open('a_price_23.txt', 'a+') as price_file:
                            current_entry = f"{current_time} {a_prices23}\n"
                            price_file.write(current_entry)
                            price_file.seek(0)
                            entries = price_file.readlines()
                            if len(entries) >= 500:
                                entries = entries[-500:]
                            price_file.seek(0)
                            price_file.truncate()
                            price_file.writelines(entries)

                        # print('Angle prices 23:', a_prices23)

                else:
                    print("Error: Insufficient data in price.txt.")
                    return None
        except FileNotFoundError:
            print("Error: price.txt not found.")
            return None
        try:
            with open('current_bb_percent.txt', 'r') as f:
                lines = f.readlines()

                if len(lines) >= 12:
                    last_bb_percent = [float(line.split()[1]) for line in lines[-12:]]
                    start_index_prices = len(lines) -12
                    end_index_prices = len(lines)
                    angle_bb_percent = round (np.polyfit(range(len(last_bb_percent)), last_bb_percent, 1)[0],4)

                    with open('angle_bb_percent.txt', 'a+') as bb_file:
                        current_entry = f"{current_time} {angle_bb_percent}\n"
                        bb_file.write(current_entry)
                        bb_file.seek(0)
                        entries = bb_file.readlines()
                        if len(entries) >= 500:
                            entries = entries[-500:]
                        bb_file.seek(0)
                        bb_file.truncate()
                        bb_file.writelines(entries)

                    # print('angle_bb_percent prices:', angle_bb_percent)

                    if len(lines) >= 23:
                        last_bb_percent = [float(line.split()[1]) for line in lines[-23:]]
                        start_index_prices = len(lines) - 23
                        end_index_prices = len(lines)
                        a_bb_23 = round (np.polyfit(range(len(last_bb_percent)), last_bb_percent, 1)[0],4)

                        with open('angle_bb_23.txt', 'a+') as bb_file:
                            current_entry = f"{current_time} {a_bb_23}\n"
                            bb_file.write(current_entry)
                            bb_file.seek(0)
                            entries = bb_file.readlines()
                            if len(entries) >= 500:
                                entries = entries[-500:]
                            bb_file.seek(0)
                            bb_file.truncate()
                            bb_file.writelines(entries)

                        # print('angle_bb_23 prices:', a_bb_23)
                    # print("Last bb_percent:", last_bb_percent[-3:])

                else:
                    print("Error: Insufficient data in angle_bb_percent.txt.")
                    return None

                # Проверка, что есть минимум 3 записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    bb_percent_1 = float(entry_1[1])
                    bb_percent_2 = float(entry_2[1])
                    bb_percent_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: current_bb_percent.txt not found.")
        except Exception as e:
            print(f"Error reading current_bb_percent.txt: {e}")
        try:
            with open('volume.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    volume_1 = float(entry_1[1])
                    volume_2 = float(entry_2[1])
                    volume_3 = float(entry_3[1])

                if len(lines) >= 17:
                    last_positions = [float(line.split()[1]) for line in lines[-17:]]
                    start_index_positions = len(lines) - 17
                    end_index_positions = len(lines)
                    a_vol17 = abs (round (np.polyfit(range(len(last_positions)), last_positions, 1)[0], 4))

                    with open('a_vol17.txt', 'a+') as posit_file:
                        current_entry = f"{current_time} {a_vol17}\n"
                        posit_file.write(current_entry)
                        posit_file.seek(0)
                        entries = posit_file.readlines()
                        if len(entries) >= 500:
                            entries = entries[-500:]
                        posit_file.seek(0)
                        posit_file.truncate()
                        posit_file.writelines(entries)

        except FileNotFoundError:
            print("Error: volume.txt not found.")
        except Exception as e:
            print(f"Error reading volume.txt: {e}")
        try:
            with open('angle_slope.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    angle_slope_1 = float(entry_1[1])
                    angle_slope_2 = float(entry_2[1])
                    angle_slope_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: angle_slope.txt not found.")
        except Exception as e:
            print(f"Error reading angle_slope.txt: {e}")
        try:
            with open('angle_positions.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    angle_positions_1 = float(entry_1[1])
                    angle_positions_2 = float(entry_2[1])
                    angle_positions_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: angle_positions.txt not found.")
        except Exception as e:
            print(f"Error reading angle_positions.txt: {e}")
        try:
            with open('angle_price.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    angle_prices_1 = float(entry_1[1])
                    angle_prices_2 = float(entry_2[1])
                    angle_prices_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: angle_price.txt not found.")
        except Exception as e:
            print(f"Error reading angle_price.txt: {e}")
        try:
            with open('angle_bb_percent.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    angle_bb_1 = float(entry_1[1])
                    angle_bb_2 = float(entry_2[1])
                    angle_bb_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: angle_bb_percent.txt not found.")
        except Exception as e:
            print(f"Error reading angle_bb_percent.txt: {e}")
        try:
            with open('a_price_23.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    a_pr23_1 = float(entry_1[1])
                    a_pr23_2 = float(entry_2[1])
                    a_pr23_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: a_price_23.txt not found.")
        except Exception as e:
            print(f"Error reading a_price_23.txt: {e}")
        try:
            with open('angle_bb_23.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    a_bb23_1 = float(entry_1[1])
                    a_bb23_2 = float(entry_2[1])
                    a_bb23_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: angle_bb_23.txt not found.")
        except Exception as e:
            print(f"Error reading angle_bb_23.txt: {e}")
        try:
            with open('a_pos23.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    a_pos23_1 = float(entry_1[1])
                    a_pos23_2 = float(entry_2[1])
                    a_pos23_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: a_pos23.txt not found.")
        except Exception as e:
            print(f"Error reading a_pos23.txt: {e}")
        try:
            with open('a_vol17.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    a_vol17_1 = float(entry_1[1])
                    a_vol17_2 = float(entry_2[1])
                    a_vol17_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: a_vol17.txt not found.")
        except Exception as e:
            print(f"Error reading a_vol17.txt: {e}")

        last_vol3 = round ((volume_3 + volume_2 + volume_1),4)
        print('Last_vol3:',last_vol3,'/ ',volume_3,'/ ' ,volume_2 , '/ ',volume_1)
        all_angle23 = round ((a_slope23 + a_pos23 + a_prices23 + a_bb_23), 4)
        # print('Suma all angle23:', all_angle23)

        all_price = round ((angle_prices_1 + a_pr23_1)/2, 4)
        with open('all_price.txt', 'a+') as file:
            current_entry = f"{current_time} {all_price}\n"
            file.write(current_entry)
            file.seek(0)
            entries = file.readlines()
            if len(entries) >= 500:
                entries = entries[-500:]
            file.seek(0)
            file.truncate()
            file.writelines(entries)

            print('Angle all_price:', all_price)
        try:
            with open('all_price.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    all_price_1 = float(entry_1[1])
                    all_price_2 = float(entry_2[1])
                    all_price_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: all_price.txt not found.")
        except Exception as e:
            print(f"Error reading all_price.txt: {e}")

        all_pos = round ((angle_positions + a_pos23)/2, 4)
        with open('all_pos.txt', 'a+') as file:
            current_entry = f"{current_time} {all_pos}\n"
            file.write(current_entry)
            file.seek(0)
            entries = file.readlines()
            if len(entries) >= 500:
                entries = entries[-500:]
            file.seek(0)
            file.truncate()
            file.writelines(entries)

            print('Angle all_pos:', all_pos)
        try:
            with open('all_pos.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    all_pos_1 = float(entry_1[1])
                    all_pos_2 = float(entry_2[1])
                    all_pos_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: all_pos.txt not found.")
        except Exception as e:
            print(f"Error reading all_pos.txt: {e}")

        all_bb = round ((angle_bb_1 + a_bb_23) /2, 4)
        with open('all_bb.txt', 'a+') as file:
            current_entry = f"{current_time} {all_bb}\n"
            file.write(current_entry)
            file.seek(0)
            entries = file.readlines()
            if len(entries) >= 500:
                entries = entries[-500:]
            file.seek(0)
            file.truncate()
            file.writelines(entries)

            print('Angle all_bb:', all_bb)
        try:
            with open('all_bb.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    all_bb_1 = float(entry_1[1])
                    all_bb_2 = float(entry_2[1])
                    all_bb_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: all_bb.txt not found.")
        except Exception as e:
            print(f"Error reading all_bb.txt: {e}")

        all_a = round ((angle_slope + all_angle23)/2 , 4)
        with open('all_a.txt', 'a+') as price_file:
            current_entry = f"{current_time} {all_a}\n"
            price_file.write(current_entry)
            price_file.seek(0)
            entries = price_file.readlines()
            if len(entries) >= 500:
                entries = entries[-500:]
            price_file.seek(0)
            price_file.truncate()
            price_file.writelines(entries)

            print('Angle all_a:', all_a)
        try:
            with open('all_a.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    all_a_1 = float(entry_1[1])
                    all_a_2 = float(entry_2[1])
                    all_a_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: all_a.txt not found.")
        except Exception as e:
            print(f"Error reading all_a.txt: {e}")

        all_sum = round ((all_price + all_pos + all_bb + all_a), 4)
        with open('all_sum.txt', 'a+') as s_file:
            current_entry = f"{current_time} {all_sum}\n"
            s_file.write(current_entry)
            s_file.seek(0)
            entries = s_file.readlines()
            if len(entries) >= 500:
                entries = entries[-500:]
            s_file.seek(0)
            s_file.truncate()
            s_file.writelines(entries)

            print('Angle all_sum:', all_sum)
        try:
            with open('all_sum.txt', 'r') as f:
                lines = f.readlines()
                # Проверка, что есть минимум две записи
                if len(lines) >= 3:
                    # Получаем последние две записи
                    entry_1 = lines[-1].split()
                    entry_2 = lines[-2].split()
                    entry_3 = lines[-3].split()
                    # Преобразование в числа
                    all_sum_1 = float(entry_1[1])
                    all_sum_2 = float(entry_2[1])
                    all_sum_3 = float(entry_3[1])
        except FileNotFoundError:
            print("Error: all_sum.txt not found.")
        except Exception as e:
            print(f"Error reading all_sum.txt: {e}")

        try:
            with open("signal.txt", "r") as file:
                content = file.readlines()
                last_signals = [line.strip().split(", ") for line in content]
                # print(last_signals)
            received_signals = [signal[0].split(": ")[1].replace("'", "").replace("[", "").replace("]", "") for signal in last_signals]
            signal_PSM = received_signals[-10:]
            print(signal_PSM)

        except FileNotFoundError:
            print("Error: signal.txt not found.")
            signal_PSM = []
        except Exception as e:
            print(f"Error reading signal.txt: {e}")
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if ((all_price_3 < 0 and all_price_2 > 0) and all_price_1 < -1 ):
            signals.append('Pr_1')
        if ((all_price_3 < 0 and all_price_2 > 0) and all_price_1 > 1 ):
            signals.append('Pr_2')
        if ((all_price_3 > all_price_2 < all_price_1) and all_price_2 < -0.3):
            signals.append('Pr_3')
        if ((all_price_3 < all_price_2 > all_price_1) and all_price_2 > 0.3):
            signals.append('Pr_4')

        if ((all_pos_3 > 0 and all_pos_2 < 0) and all_pos_1 < -0.01):
            signals.append('Pos_1')
        if ((all_pos_3 < 0 and all_pos_2 > 0) and all_pos_1 > 0.01):
            signals.append('Pos_2')
        if ((all_pos_3 > all_pos_2 < all_pos_1) and all_pos_2 < -0.03):
            signals.append('Pos_3')
        if ((all_pos_3 < all_pos_2 > all_pos_1) and all_pos_2 > 0.03):
            signals.append('Pos_4')

        if ((all_bb_3 > 0 and all_bb_2 < 0) and all_bb_1 < -1):
            signals.append('BB_1')
        if ((all_bb_3 < 0 and all_bb_2 > 0) and  all_bb_1 > 1 ):
            signals.append('BB_2')
        if ((all_bb_3 > all_bb_2 < all_bb_1) and all_bb_2 < -3):
            signals.append('BB_3')
        if ((all_bb_3 < all_bb_2 > all_bb_1) and all_bb_2 > 3 ):
            signals.append('BB_4')

        if ((all_a_3 > 0 and all_a_2 < 0) and all_a_1 < -2):
            signals.append('ASL_1')
        if ((all_a_3 < 0 and all_a_2 > 0) and all_a_1 > 2):
            signals.append('ASL_2')
        if ((all_a_3 > all_a_2 < all_a_1) and all_a_2 < -4):
            signals.append('ASL_3')
        if ((all_a_3 < all_a_2 > all_a_1) and all_a_2 > 4):
            signals.append('ASL_4')

        if ((all_sum_3 > 0 and all_sum_2 < 0) and all_sum_1 < -1):
            signals.append('Sum1')
        if ((all_sum_3 < 0 and all_sum_2 > 0) and all_sum_1 > 1):
            signals.append('Sum2')
        if ((all_sum_3 > all_sum_2 < all_sum_1) and all_sum_2 < -8):
            signals.append('Sum3')
        if ((all_sum_3 < all_sum_2 > all_sum_1) and all_sum_2 > 8):
            signals.append('Sum4')

        if ((a_bb23_3 > a_bb23_2 < a_bb23_1) and a_bb23_2 < -2.5):
            signals.append('BB23_1')
        if ((a_bb23_3 < a_bb23_2 > a_bb23_1) and a_bb23_2 > 2.5):
            signals.append('BB23_2')

        if ((a_vol17_3 > a_vol17_2 < a_vol17_1) and a_vol17_2 < 10):
            signals.append('VOL17_1')
        if ((a_vol17_3 < a_vol17_2 > a_vol17_1) and a_vol17_2 > 50):
            signals.append('VOL17_2')

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if all_price_1 > -0.2 and \
            (  all(sig in signal_PSM for sig in ['Pos_3', 'ASL_3', 'Sum3'])\
            or all(sig in signal_PSM for sig in ['BB_1', 'ASL_1', 'Sum1'])\
            or all(sig in signal_PSM for sig in ['Pr_3', 'Pos_3', 'BB_3', 'ASL_3', 'Sum3'])\
            or all(sig in signal_PSM for sig in ['Pos_3', 'BB_3', 'ASL_3', 'Sum3', 'VOL17_1']) \
                ):
            print("UP")
            signals.append('UP')

        if  all_price_1 < 0.25 and \
            ( all(sig in signal_PSM for sig in ['Pos_4', 'BB_4', 'ASL_4', 'Sum4']) \
            or all(sig in signal_PSM for sig in ['ASL_4', 'Sum4', 'BB23_2']) \
            or all(sig in signal_PSM for sig in ['VOL17_1', 'Pos_4', 'ASL_4', 'BB23_2', 'ASL_4', 'Pr_4']) \
            or all(sig in signal_PSM for sig in ['VOL17_1', 'Pos_4', 'ASL_4', 'BB23_2', 'ASL_4', 'Pr_4']) \
            or all(sig in signal_PSM for sig in ['VOL17_1', 'Pos_4', 'ASL_4', 'BB23_2', 'ASL_4', 'Pr_4']) \
            or all(sig in signal_PSM for sig in ['VOL17_1', 'Pos_4', 'ASL_4', 'BB23_2', 'ASL_4', 'Pr_4']) \
            or all(sig in signal_PSM for sig in ['VOL17_1', 'Pos_4', 'ASL_4', 'BB23_2', 'ASL_4', 'Pr_4']) \
            or all(sig in signal_PSM for sig in ['BB_2', 'Sum2']) \
                ):
            print("DW")
            signals.append('DW')
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            # Отслеживание состояний
            all_price = signals.get('all_price', [])
            all_pos = signals.get('all_pos', [])
            all_bb = signals.get('all_bb', [])
            all_a = signals.get('all_a', [])
            all_sum = signals.get('all_sum', [])
            a_bb23 = signals.get('a_bb23', [])
            a_vol17 = signals.get('a_vol17', [])

            # Условие для ЛОНГ
            if (
                    all_price[-1] > 0.2 and
                    all_price[-1] > all_price[-2] and
                    all_pos[-1] > all_pos[-2] and
                    all_sum[-1] > 3 and
                    all_sum[-1] > all_sum[-2] and
                    a_bb23[-1] > -0.5 and
                    a_bb23[-1] > a_bb23[-2] and
                    a_vol17[-1] > 3
            ):
                decision = 'LG1'



# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#         state = "WAITING"
#         last_price = current_price
#         last_position = last_positions [-1]
#
#         for signal in signal_PSM:
#             if state == "WAITING":
#                 if "Pos_1" in signal and "BB_1" in signal and "ASL_1" in signal and "Sum1" in signal:
#                     state = "LONG_ENTRY"
#                 elif "Pr_3" in signal and "Pos_3" in signal and "BB_3" in signal and "ASL_3" in signal and "Sum3" in signal:
#                     state = "SHORT_ENTRY"
#             elif state == "LONG_ENTRY":
#                 if last_price [-2] > last_price [-1]:
#                     # Logic to enter long position
#                     state = "WAITING"
#             elif state == "SHORT_ENTRY":
#                 if last_price [-2] < last_price [-1]:
#                     # Logic to enter short position
#                     state = "WAITING"
#
#             elif state == "LONG_EXIT":
#                 if last_price [-2] < last_price [-1]:
#                     # Logic to exit long position
#                     state = "WAITING"
#
#             elif state == "SHORT_EXIT":
#                 if last_price[-1] > last_price [-1]:
#                     # Logic to exit short position
#                     state = "WAITING"


        return signals
    else:
        print("Error getting klines data. Check your internet connection.")
        return None

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#         if (
#                 ((all_price_3 > all_price_1 or all_price_2 > all_price_1) and
#                 ((all_sum_3 > all_sum_2 < all_sum_1) and all_sum_2 < -5) and
#                 (angle_bb_1 < -2 or angle_bb_1 < -2) and
#                 (all_sum_2 < all_sum_1) and
#                 (all_sum_3 < -5 or all_sum_2 < -5 or all_sum_1 < -5)) and
#                 (last_positions [-2] < 0.2 or last_positions [-1] < 0.2)
#         ):
#             signals.append(('LONG', current_time))
#
#         if (
#                 ((all_sum_3 > 9 or all_sum_2 > 9 or all_sum_1 > 9) or
#                 ((all_sum_3 < all_sum_1 or all_sum_2 < all_sum_1) and all_sum_1 > 8)) and
#                 (all_price_3 > 0.1 or  all_price_2 > 0.1 or all_price_1 > 0.1) and
#                 (all_pos_3 > 0.01 or all_pos_2 > 0.01 or all_pos_1 > 0.01) and
#                 (all_bb_3 > 2 or all_bb_2 > 2 or all_bb_1 > 2) and
#                 (all_a_3 > 2 or all_a_2 > 2 or all_a_1 > 2)
#                 # (last_prices [-2] < last_prices [-1])  # под вопросом
#
#         ):
#             signals.append(('LONG_X', current_time))
#
#         # if (
#         #         (all_sum_2 > 10 or all_sum_2 > 10 or all_sum_1 > 10) or
#         #         (volume_3 < 100 or volume_2 < 100 or volume_1 < 100) or
#         #         (all_price_3 > 0.22 or all_price_2 > 0.22 or all_price_1 > 0.22) or
#         #         (all_pos_3 > 0.05 or all_pos_2 > 0.05 or all_pos_1 > 0.05) or
#         #         (all_bb_3 > 4 or all_bb_2 > 4 or all_bb_1 > 4) or
#         #         (all_a_3 > 7 or all_a_2 > 7 and all_a_1 > 7) or
#         #         (last_slope [-3] > 45 or last_slope [-2] > 45 or last_slope [-1] > 45)
#         # ):
#         #     signals.append(('LONG_X', current_time))
#
#         if (
#                 ((((all_price_3 < all_price_2 > all_price_1) and
#                 (all_price_2 > 0.4 or all_price_1 > 0.4)) and
#                 (angle_bb_3 > 3 or angle_bb_2 > 3 or angle_bb_1 > 3)) and
#                 (last_prices [-2] < last_prices [-1])) or
#
#                 (((all_sum_3 < all_sum_2 > all_sum_1) and (all_sum_2 > 8 or all_sum_1 > 8) and
#                 (all_price_2 > 0.4 or all_price_1 > 0.4) and
#                 (angle_bb_3 > 6 or angle_bb_2 > 6 or angle_bb_1 > 6 )) and
#                 (last_prices [-2] < last_prices [-1])) or
#
#                 ((angle_bb_3 < angle_bb_2 > angle_bb_1 ) and
#                 (angle_bb_3 > 6 or angle_bb_2 > 6 or angle_bb_1 > 6 )) and
#                 (all_price_2 > 0.4 or all_price_1 > 0.4) and
#                 (last_prices[-2] < last_prices[-1])
#         ):
#             signals.append(('SHORT', current_time))
#
#         if (
#                 (all_sum_3 < -9 or all_sum_2 < -9 or all_sum_1 > -9) and
#                 (all_price_3 < -0.1 or  all_price_2 < -0.1 or all_price_1 < -0.1) and
#                 (all_pos_3 < -0.01 or all_pos_2 < -0.01 or all_pos_1 < -0.01) and
#                 (all_bb_3 < -2 or all_bb_2 < -2 or all_bb_1 < -2) and
#                 (all_a_3 < -2 or all_a_2 < -2 or all_a_1 < -2) and
#                 (volume_3 < volume_2 < volume_1)
#                 # (last_slope [-3] < last_slope [-2] < last_slope [-1]) and
#         ):
#             signals.append(('SHORT_X', current_time))
#
#         if (
#                 ((last_slope[-3] > 45 or last_slope[-2] > 45) and last_slope[-1] > 45) and
#                 ((last_positions [-3] > 0.95 or last_positions [-2] > 0.95) and last_positions [-1] < 0.95) and
#                 (last_vol3 < 1000)
#         ):
#             signals.append(('UP', current_time))
#
#         if (
#                 ((last_slope[-3] < -45 or last_slope[-2] < -45) and last_slope[-1] > -45) and
#                 ((last_positions[-3] < 0.05 or last_positions[-2] < 0.05) and last_positions[-1] > 0.05) and
#                 (last_vol3 > 2000)
#         ):
#             signals.append(('DOWN', current_time))

        # if (
        #         (all_sum_2 < 2 or all_sum_2 < 2 or all_sum_1 < 2) or
        #         (volume_3 > 1500 or volume_2 > 1500 or volume_1 > 1500) or
        #         (all_price_3 < - 0.1 or all_price_2 < - 0.1 or all_price_1 < - 0.1) or
        #         (all_pos_3 < -0.02 or all_pos_2 < -0.02 or all_pos_1 < -0.02) or
        #         (all_bb_3 < -4 or all_bb_2 < -4 or all_bb_1 < -4) or
        #         (all_a_3 < -7 or all_a_2 < -7 and all_a_1 < -7) or
        #         (last_slope [-3] < -45 or last_slope [-2] < -45 or last_slope [-1] < -45)
        # ):
        #     signals.append(('SHORT_X', current_time))

        #
        # if (
        #         # (last_slope[-3] < -46 or last_slope[-2] < -46 or last_slope[-1] < -46) and
        #         (last_positions[-3] > 0.95 or last_positions[-2] > 0.95 or last_positions[-1] > 0.95) and
        #
        #         (all_pos_3 > all_pos_1 and all_pos_2 > all_pos_1 and all_pos_1 < -0.4) and
        #         (all_a_3 < -3 or all_a_2 < -3 and all_a_1 < -3) and
        #         (all_bb_3 < -3 or all_bb_2 < -3 or all_bb_1 < -3)
        #         # (volume_3 > volume_1 or volume_2 > volume_1)
        # ):
        #     signals.append(('SHORT', current_time))

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # if (
        #         (last_slope[-3] < -44.5 or last_slope[-2] < -44.5 or last_slope[-1] < -44.5) and
        #         (last_positions [-3] < 0.05 or last_positions [-2] < 0.05 or last_positions [-1] < 0.05) and
        #         (volume_3 < 100 or volume_2 < 100 or volume_1 < 100)
        # ):
        #     signals.append(('L_down', current_time))
        #
        # if (
        #         (last_slope[-3] < -46 or last_slope[-2] < -46 or last_slope[-1] < -46) and
        #         (last_positions [-3] < 0.05 or last_positions [-2] < 0.05 or last_positions [-1] < 0.05) and
        #         (volume_3 > 1500 or volume_2 > 1500 or volume_1 > 1500) and
        #         (all_sum_3 > all_sum_2 < all_sum_1)
        # ):
        #     signals.append(('L_up', current_time))
        #
        # if (
        #         (last_slope[-3] > 45 or last_slope[-2] > 45 or last_slope[-1] > 45) and
        #         (last_positions [-3] > 0.95 or last_positions [-2] > 0.95 or last_positions [-1] > 0.95) and
        #         (volume_3 < 100 or volume_2 < 100 or volume_1 < 100) and
        #         (all_sum_3 < all_sum_2 > all_sum_1)
        # ):
        #     signals.append(('S_down', current_time))
        #
        # if (
        #         (last_slope[-3] > 44.5 or last_slope[-2] > 44.5 or last_slope[-1] > 44.5) and
        #         (last_positions [-3] > 0.95 or last_positions [-2] > 0.95 or last_positions [-1] > 0.95) and
        #         (volume_3 > 1500 or volume_2 > 1500 or volume_1 > 1500)
        # ):
        #     signals.append(('S_up', current_time))
# ////////////////////////////////////////////////////////////////////////////////////////
#         return signals
#     else:
#         print("Error getting klines data. Check your internet connection.")
#         return None

# signals = [line.split('- Signal: ')[1].split(',')[0].strip() for line in lines]

def main(SYMBOL):
    INTERVAL = '1m'
    current_bb_percent = 0  # Инициализация с каким-то начальным значением
    while True:
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = get_symbol_price(SYMBOL)
        print('Current_price:', current_price)

        # decision = decide_trade()
        # print(f"Решение: {decision}")

        # Переменная current_bb_percent инициализируется перед использованием
        current_bb_percent = get_symbol_price_with_bollinger_bands(SYMBOL, INTERVAL, LIMIT, bb_length=10, bb_multiplier=2)
        signals = check_if_signal(SYMBOL, current_time, current_price)
        if signals:
            print(f"Signal received: {signals}")
            print(f"Time: {current_time}")
            print(f"Current Symbol Price: {current_price}")
            print(f"Current current_bb_percent: {current_bb_percent}")
            with open("signal.txt", "a+") as file:
                file.write(f"{current_time} - Signal: {signals}, Symbol Price: {current_price}\n")
                # current_entry = f"{current_time} {signals}\n"
                # file.write(current_entry)
                file.seek(0)
                entries = file.readlines()
                if len(entries) >= 300:
                    entries = entries[-300:]
                file.seek(0)
                file.truncate()
                file.writelines(entries)
        else:
            print("No signal at the moment.")
        # Пример использования

        # analyze_signals_and_write()

        time.sleep(60)

if __name__ == "__main__":
    main(SYMBOL)

# def main():
#     INTERVAL = '1m'  # Установите интервал на '1m'
#     while True:
#         signal = check_if_signal(SYMBOL)
#         if signal:
#             print(f"Signal received: {signal}")
#             with open("signal.txt", "w") as file:
#                 file.write(signal)
#         else:
#             print("No signal at the moment.")
#         time.sleep(60)  # Подождать 1 минуту (60 секунд)
#
# if __name__ == "__main__":
#     main()






