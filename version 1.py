import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
import copy
import time
import random

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from futures_sign import send_signed_request, send_public_request
from cred import KEY, SECRET

import logging

logging.basicConfig(filename='bot.log', level=logging.INFO)

client = Client(KEY, SECRET)
SYMBOL = 'ETHUSDT'
LIMIT = 150
INTERVAL = '1m'
LEVERAGE = 10
TOTAL_CAF = 0.25
stop_percent = 0.005

POINTER = str(random.randint(1000, 9999))
URL = f'https://binance.com/fapi/v1/klines?symbol={SYMBOL}&limit={LIMIT}&interval={INTERVAL}'


def get_random_user_agent():
    os_version = f"Windows NT 10.0; Win64; x64"
    webkit_version = f"AppleWebKit/{random.randint(500, 600)}.{random.randint(10, 99)}"
    chrome_version = f"Chrome/{random.randint(100, 150)}.{random.randint(0, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
    safari_version = f"Safari/{random.randint(500, 600)}.{random.randint(10, 99)}"
    user_agent = f"Mozilla/5.0 ({os_version}) {webkit_version} (KHTML, like Gecko) {chrome_version} {safari_version}"
    return user_agent


def get_futures_klines(SYMBOL, LIMIT):
    user_agent = get_random_user_agent()
    headers = {
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "user-agent": user_agent,
    }
    try:
        response = requests.get(url=URL, headers=headers)
        response.raise_for_status()  # Check the response status
        data = response.json()
    except requests.RequestException as e:
        print(f"Error getting klines data: {e}")
        return None

    df = pd.DataFrame(data)
    df = pd.DataFrame(response.json())
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'd1', 'd2', 'd3', 'd4', 'd5']
    df = df.drop(['d1', 'd2', 'd3', 'd4', 'd5'], axis=1)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df


def get_symbol_price(SYMBOL):
    user_agent = get_random_user_agent()
    headers = {
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "user-agent": user_agent,
    }
    try:
        url = f'https://api.binance.com/api/v3/ticker/price?symbol={SYMBOL}'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except requests.RequestException as e:
        print(f"Error getting price for symbol {SYMBOL}: {e}")
        return None


def get_max_position(client, SYMBOL, TOTAL_CAF, LEVERAGE):
    symbol_price = get_symbol_price(SYMBOL)
    wallet = client.futures_account()
    balance = wallet.get('totalWalletBalance')

    if balance is None:
        print("totalWalletBalance is not available in the wallet.")
        return None

    balance = round(float(balance), 3)

    if symbol_price is not None:
        maxposition = round((balance * TOTAL_CAF * LEVERAGE) / symbol_price, 3)
        return maxposition
    else:
        print(f"Unable to calculate max position for {SYMBOL}.")
        return None


def open_position(SYMBOL, s_l, quantity_l):
    sprice = get_symbol_price(SYMBOL)
    stop_price = 0

    if s_l == 'long':
        stop_price = str(round(sprice * (1 + stop_percent), 4))
    elif s_l == 'short':
        stop_price = str(round(sprice * (1 - stop_percent), 4))

    if stop_price:
        params = {
            "symbol": SYMBOL,
            "side": "BUY" if s_l == 'long' else "SELL",  # Обратите внимание на смену "BUY" и "SELL"
            "type": "LIMIT",
            "quantity": str(quantity_l),
            "timeInForce": "GTC",
            "price": stop_price
        }

        response = send_signed_request('POST', '/fapi/v1/order', params)
        # Возможно, вам также потребуется обработка ответа или логгирование


def close_position(SYMBOL, s_l, quantity_l):
    sprice = get_symbol_price(SYMBOL)
    stop_price = 0

    if s_l == 'long':
        stop_price = str(round(sprice * (1 - stop_percent), 4))
    elif s_l == 'short':
        stop_price = str(round(sprice * (1 + stop_percent), 4))

    if stop_price:
        params = {
            "symbol": SYMBOL,
            "side": "SELL" if s_l == 'long' else "BUY",
            "type": "LIMIT",
            "quantity": str(quantity_l),
            "timeInForce": "GTC",
            "price": stop_price
        }

        response = send_signed_request('POST', '/fapi/v1/order', params)
        # Возможно, вам также потребуется обработка ответа или логгирование


def get_opened_positions(client, SYMBOL):
    try:
        status = client.futures_account()
        positions = pd.DataFrame(status['positions'])
        position_amt = positions[positions['symbol'] == SYMBOL]['positionAmt'].astype(float).tolist()[0]
        leverage = int(positions[positions['symbol'] == SYMBOL]['leverage'].iloc[0])
        entryprice = positions[positions['symbol'] == SYMBOL]['entryPrice']
        profit = float(status['totalUnrealizedProfit'])
        balance = round(float(status['totalWalletBalance']), 2)
        if position_amt > 0:
            pos = "long"
        elif position_amt < 0:
            pos = "short"
        else:
            pos = ""
        return ([pos, position_amt, profit, leverage, balance, round(float(entryprice.iloc[0]), 3), 0])

    except Exception as e:
        print(f"Error getting opened positions: {e}")
        return ["", 0, 0, 0, 0, 0, 0]


def check_and_close_orders(client, SYMBOL):  # проверяет наличие открытых ордеров для указанного символа на бирже
    try:
        open_orders = client.futures_get_open_orders(symbol=SYMBOL)

        if open_orders:
            client.futures_cancel_all_open_orders(symbol=SYMBOL)
            return True  # Вернем True, если были открытые ордера и они были закрыты
    except Exception as e:
        print(f"Error checking and closing orders for {SYMBOL}: {e}")

    return False  # Вернем False, если не было открытых ордеров или их не удалось закрыть


def indSlope(series, n):
    array_sl = [j * 0 for j in range(n - 1)]
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
    df_temp = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    return df_temp  # не менял хотя просит отказаться от temp


def isLCC(DF, i):
    df = DF.copy()
    LCC = 0
    if df['close'][i] <= df['close'][i + 1] and df['close'][i] <= df['close'][i - 1] and df['close'][i + 1] > \
            df['close'][i - 1]:
        LCC = i - 1;
    return LCC


def isHCC(DF, i):
    df = DF.copy()
    HCC = 0
    if df['close'][i] >= df['close'][i + 1] and df['close'][i] >= df['close'][i - 1] and df['close'][i + 1] < \
            df['close'][i - 1]:
        HCC = i;
    return HCC


def PrepareDF(DF, n=12):
    ohlc = DF.iloc[:, [0, 1, 2, 3, 4, 5]]
    ohlc.columns = ["date", "open", "high", "low", "close", "volume"]
    ohlc = ohlc.set_index('date')

    # Вычисляем максимум и минимум канала
    maxx = 0
    minn = ohlc['low'].max()
    for i in range(1, n):
        if maxx < ohlc['high'].iloc[-i]:
            maxx = ohlc['high'].iloc[-i]
        if minn > ohlc['low'].iloc[-i]:
            minn = ohlc['low'].iloc[-i]

    df = indATR(ohlc, 14).reset_index()
    df['slope'] = indSlope(df['close'], 5)
    df['channel_max'] = maxx  # Максимум канала
    df['channel_min'] = minn  # Минимум канала
    df['position_in_channel'] = (df['close'] - minn) / (maxx - minn)
    df = df.set_index('date')
    df = df.reset_index()
    return df


def check_if_signal(SYMBOL):
    ohlc = get_futures_klines(SYMBOL, 100)
    prepared_df = PrepareDF(ohlc)
    signal = ""

    i = 98

    if isLCC(prepared_df, i - 1) > 0:
        if prepared_df['position_in_channel'][i - 1] < 0.5:
            if prepared_df['slope'][i - 1] < -20:
                signal = 'long'
    if isHCC(prepared_df, i - 1) > 0:
        if prepared_df['position_in_channel'][i - 1] > 0.5:
            if prepared_df['slope'][i - 1] > 20:
                signal = 'short'
    return signal


def prt(message):
    print(POINTER + ': ' + message)


def main(counterr, client, SYMBOL, TOTAL_CAF, LEVERAGE):
    position_closed = False  # Флаг для отслеживания закрытия позиции
    while True:
        try:
            maxposition = get_max_position(client, SYMBOL, TOTAL_CAF, LEVERAGE)
            position = get_opened_positions(client, SYMBOL)
            open_sl = position[0]

            if open_sl == "":  # no position
                prt('Нет открытых позиций')
                # close all stop loss orders
                check_and_close_orders(client, SYMBOL)
                signal = check_if_signal(SYMBOL)

                if signal == 'long':
                    open_position(SYMBOL, 'long', maxposition)
                elif signal == 'short':
                    open_position(SYMBOL, 'short', maxposition)
            else:
                entry_price = position[5]  # enter price
                current_price = get_symbol_price(SYMBOL)

                fees = client.get_trade_fee(symbol=SYMBOL)
                maker_fee = float(fees[0]['makerCommission'])
                taker_fee = float(fees[0]['takerCommission'])

                opening_commission = entry_price * position[1] * float(maker_fee) / 2
                closing_commission = current_price * position[1] * float(taker_fee) / 2

                if open_sl == 'long':
                    take_profit = round(
                        (current_price - position[5]) * position[1] - (opening_commission + closing_commission), 5)
                    max_profit = take_profit  # Инициализируем максимальную прибыль текущей целевой прибылью
                    while True:
                        current_price = get_symbol_price(SYMBOL)
                        target_profit = round(
                            (current_price - position[5]) * position[1] - (opening_commission + closing_commission), 5)

                        if target_profit >= max_profit:
                            # Обновляем максимальную прибыль и продолжаем следить
                            max_profit = target_profit
                            time.sleep(40)
                        else:
                            # Если целевая прибыль уменьшается, закрываем позицию и выводим прибыль
                            position = get_opened_positions(client, SYMBOL)
                            if position[0] == 'long':
                                close_position(SYMBOL, 'long', abs(position[1]))
                                position_closed = True
                                break  # Выход из цикла

                if position[0] == 'short':
                    take_profit = round(
                        (position[5] - current_price) * position[1] - (opening_commission + closing_commission), 5)
                    max_profit = take_profit  # Инициализируем максимальную прибыль текущей целевой прибылью
                    while True:
                        current_price = get_symbol_price(SYMBOL)
                        target_profit = round(
                            (position[5] - current_price) * position[1] - (opening_commission + closing_commission), 5)

                        if target_profit >= max_profit:
                            max_profit = target_profit
                            time.sleep(40)
                        else:
                            position = get_opened_positions(client, SYMBOL)
                            if position[0] == 'short':
                                close_position(SYMBOL, 'short', abs(position[1]))
                                position_closed = True
                                break  # Выход из цикла

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Continuing...")

        if position_closed:
            position_closed = False  # Сбрасываем флаг
            continue  # Перезапускаем main

        #  time.sleep(60)


starttime = time.time()
timeout = time.time() + 60 * 60 * 12  # 60 seconds times 60 meaning the script will run for 12 hr
counterr = 1

if __name__ == "__main__":
    while True:
        main(counterr, client, SYMBOL, TOTAL_CAF, LEVERAGE)
        counterr += 1
        time.sleep(60)




