import requests
import pandas as pd
import time
import random
import threading
import argparse

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from futures_sign import send_signed_request, send_public_request
from cred import KEY, SECRET

client = Client(KEY, SECRET)
SYMBOL = 'ETHUSDT'
LIMIT = 150
INTERVAL = '1m'
LEVERAGE = 10
TOTAL_CAF = 0.1
stop_percent = 0.001

# fees = client.get_trade_fee(symbol=SYMBOL)
# maker_fee = float(fees[0]['makerCommission'])
# taker_fee = float(fees[0]['takerCommission'])

counterr = 0
POINTER = str(random.randint(1000, 9999))
URL = f'https://binance.com/fapi/v1/klines?symbol={SYMBOL}&limit={LIMIT}&interval={INTERVAL}'

def get_random_user_agent():
    os_version = f"Windows NT 10.0; Win64; x64"
    webkit_version = f"AppleWebKit/{random.randint(500, 600)}.{random.randint(10, 99)}"
    chrome_version = f"Chrome/{random.randint(100, 150)}.{random.randint(0, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
    safari_version = f"Safari/{random.randint(500, 600)}.{random.randint(10, 99)}"
    user_agent = f"Mozilla/5.0 ({os_version}) {webkit_version} (KHTML, like Gecko) {chrome_version} {safari_version}"
    return user_agent

def get_symbol_price(SYMBOL):
    headers = {
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "user-agent": get_random_user_agent(),
    }
    INTERVAL = '1m'
    try:
        # Используем метод, соответствующий вашим требованиям
        url = f'https://fapi.binance.com/fapi/v1/klines?symbol={SYMBOL}&interval={INTERVAL}&limit=1'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Извлекаем текущую цену из ответа
        current_price = float(data[0][4])
        return current_price
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

    print(f"balance: {balance}")
    print(f"TOTAL_CAF: {TOTAL_CAF}")
    print(f"symbol_price: {symbol_price}")

    if symbol_price is not None:
        maxposition = round((balance * TOTAL_CAF * LEVERAGE) / symbol_price, 3)
        return maxposition
    else:
        print(f"Unable to calculate max position for {SYMBOL}.")
        return None

def open_position(SYMBOL, s_l, maxposition, close_position=False):
    sprice = get_symbol_price(SYMBOL)
    stop_price = 0

    if s_l == 'long':
        stop_price = str(round(sprice * (1 + stop_percent), 4))
    elif s_l == 'short':
        stop_price = str(round(sprice * (1 - stop_percent), 4))

    if stop_price:
        # Try placing a batch order with a limit order
        params = {
            "batchOrders": [
                {
                    "symbol": SYMBOL,
                    "side": "BUY" if s_l == 'long' else "SELL",
                    "type": "LIMIT",
                    "quantity": str(maxposition),
                    "timeInForce": "GTC",
                    "price": stop_price,
                    "closePosition": close_position
                }
            ]
        }

        try:
            response = send_signed_request('POST', '/fapi/v1/order', params)
            print(response)
        except Exception as e:
            print(f"Error placing limit order: {e}")

            # If it's an insufficient balance error, try placing a market order with maxposition
            if "INSUFFICIENT_BALANCE" in str(e).upper():
                print("Insufficient funds. Trying to place a market order with maxposition.")
                market_params = {
                    "symbol": SYMBOL,
                    "side": "BUY" if s_l == 'long' else "SELL",
                    "type": "MARKET",
                    "quantity": str(maxposition),
                }
                market_response = send_signed_request('POST', '/fapi/v1/order', market_params)
                print(market_response)
            else:
                print(f"Unexpected error: {e}. Could not place market order.")

# def open_position(SYMBOL, s_l, maxposition, close_position=False):
#     sprice = get_symbol_price(SYMBOL)
#     stop_price = 0
#
#     if s_l == 'long':
#         stop_price = str(round(sprice * (1 + stop_percent), 4))
#     elif s_l == 'short':
#         stop_price = str(round(sprice * (1 - stop_percent), 4))
#
#     if stop_price:
#         params = {
#             "batchOrders": [
#                 {
#                     "symbol": SYMBOL,
#                     "side": "BUY" if s_l == 'long' else "SELL",
#                     "type": "LIMIT",
#                     "quantity": str(maxposition),
#                     "timeInForce": "GTC",
#                     "price": stop_price,
#                     "closePosition": close_position
#                     # Set closePosition to True if you want to close the position when the order is triggered
#                 }
#             ]
#         }
#
#         response = send_signed_request('POST', '/fapi/v1/order', params)
#         print(response)

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
        print(response)

def get_opened_positions(client, SYMBOL):
    try:
        status = client.futures_account()
        positions = pd.DataFrame(status.get('positions', []))
        position_data = positions[positions['symbol'] == SYMBOL]

        if position_data.empty:
            return ["", 0, 0, 0, 0, 0, 0]

        position_amt = position_data['positionAmt'].astype(float).iloc[0]
        leverage = int(position_data['leverage'].iloc[0])
        entryprice = position_data['entryPrice'].iloc[0]
        profit = float(status.get('totalUnrealizedProfit', 0))
        balance = round(float(status.get('totalWalletBalance', 0)), 2)

        pos = ""
        if position_amt > 0:
            pos = "long"
        elif position_amt < 0:
            pos = "short"

        return [pos, position_amt, profit, leverage, balance, round(float(entryprice), 3), 0]

    except Exception as e:
        print(f"Error getting opened positions: {e}")
        return ["", 0, 0, 0, 0, 0, 0]

def check_and_close_orders(client, SYMBOL):  #проверяет наличие открытых ордеров для указанного символа на бирже
    try:
        open_orders = client.futures_get_open_orders(symbol=SYMBOL)

        if open_orders:
            client.futures_cancel_all_open_orders(symbol=SYMBOL)
            return True  # Вернем True, если были открытые ордера и они были закрыты
    except Exception as e:
        print(f"Error checking and closing orders for {SYMBOL}: {e}")

    return False  # Вернем False, если не было открытых ордеров или их не удалось закрыть

def get_opened_positions(client,SYMBOL):
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

def long_position_handler(SYMBOL, maxposition, random_delay, client, position_closed):
    while True:
        current_price = get_symbol_price(SYMBOL)
        position = get_opened_positions(client, SYMBOL)
        entry_price = position[5]  # enter price

        fees = client.get_trade_fee(symbol=SYMBOL)
        maker_fee = float(fees[0]['makerCommission'])
        taker_fee = float(fees[0]['takerCommission'])

        opening_commission = entry_price * position[1] * float(maker_fee) / 2
        closing_commission = current_price * position[1] * float(taker_fee) / 2

        take_profit = round((current_price - position[5]) * position[1] - (opening_commission + closing_commission), 5)
        max_profit = take_profit  # Инициализируем максимальную прибыль текущей целевой прибылью
        target_profit = round(position[2] - (opening_commission + closing_commission), 5)

        parser = argparse.ArgumentParser()
        parser.add_argument("--signal", help="Received signal from SEARCH_SIGNAL.py")
        args = parser.parse_args()

        if target_profit >= max_profit or args.signal == 'long':
            # Обновляем максимальную прибыль и продолжаем следить
            max_profit = target_profit
            time.sleep(random_delay)
        else:
            # Если целевая прибыль уменьшается, закрываем позицию и выводим прибыль
            if position[0] == 'long':
                close_position(SYMBOL, 'long', abs(position[1]))
                position_closed = True
                break  # Выход из цикла

def short_position_handler(SYMBOL, maxposition, random_delay, client, position_closed):
    while True:
        current_price = get_symbol_price(SYMBOL)
        position = get_opened_positions(client, SYMBOL)
        entry_price = position[5]  # enter price

        fees = client.get_trade_fee(symbol=SYMBOL)
        maker_fee = float(fees[0]['makerCommission'])
        taker_fee = float(fees[0]['takerCommission'])

        opening_commission = entry_price * position[1] * float(maker_fee) / 2
        closing_commission = current_price * position[1] * float(taker_fee) / 2

        take_profit = round((entry_price - current_price) * position[1] - (opening_commission + closing_commission), 5)
        max_profit = take_profit  # Инициализируем максимальную прибыль текущей целевой прибылью
        target_profit = round(position[2] - (opening_commission + closing_commission), 5)

        parser = argparse.ArgumentParser()
        parser.add_argument("--signal", help="Received signal from SEARCH_SIGNAL.py")
        args = parser.parse_args()

        if target_profit >= max_profit or args.signal == 'short':
            # Обновляем максимальную прибыль и продолжаем следить
            max_profit = target_profit
            time.sleep(random_delay)
        else:
            # Если целевая прибыль уменьшается, закрываем позицию и выводим прибыль
            if position[0] == 'short':
                close_position(SYMBOL, 'short', abs(position[1]))
                position_closed = True
                break  # Выход из цикла

def prt(message):
    # telegram message
    print(POINTER + ': ' + message)

def main(counterr, client, SYMBOL, TOTAL_CAF, LEVERAGE):
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", help="Received signal from SEARCH_SIGNAL.py")
    args = parser.parse_args()

    position_closed = False  # Флаг для отслеживания закрытия позиции
    random_delay = random.randint(50, 60)

    try:
        maxposition = get_max_position(client, SYMBOL, TOTAL_CAF, LEVERAGE)
        position = get_opened_positions(client, SYMBOL)
        open_sl = position[0]

        if open_sl == "":  # no position
            prt('Нет открытых позиций')
            # close all stop loss orders
            # check_and_close_orders(client, SYMBOL)
            # signal = check_if_signal(SYMBOL)
            # Обработка сигнала
            if args.signal:
                print(f"Received signal from SEARCH_SIGNAL.py: {args.signal}")

            if args.signal == 'long':
                open_position(SYMBOL, 'long', maxposition)
            elif args.signal == 'short':
                open_position(SYMBOL, 'short', maxposition)
        else:
            if open_sl == 'long':
                # Запустить внутренний цикл в отдельном потоке
                long_position_thread = threading.Thread(target=long_position_handler, args=(SYMBOL, maxposition, random_delay, client, position_closed))
                long_position_thread.start()
            elif open_sl == 'short':
                short_position_thread = threading.Thread(target=short_position_handler, args=(SYMBOL, maxposition, random_delay, client, position_closed))
                short_position_thread.start()

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Continuing...")

starttime = time.time()
timeout = time.time() + 60 * 60 * 12  # 60 seconds times 60 meaning the script will run for 12 hr
counterr = 1

while time.time() <= timeout:
    try:
        prt("script continue running at "+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main(counterr, client, SYMBOL, TOTAL_CAF, LEVERAGE)
        counterr=counterr+1
        if counterr>5:
            counterr=1
        time.sleep(60 - ((time.time() - starttime) % 60.0)) # 1 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\KeyboardInterrupt. Stopping.')
        exit()


# def close_position(SYMBOL, s_l, quantity_l):
#     sprice = get_symbol_price(SYMBOL)
#     stop_price = 0
#
#     if s_l == 'long':
#         stop_price = str(round(sprice * (1 - stop_percent), 4))
#     elif s_l == 'short':
#         stop_price = str(round(sprice * (1 + stop_percent), 4))
#
#     if stop_price:
#         params = {
#             "symbol": SYMBOL,
#             "side": "SELL" if s_l == 'long' else "BUY",
#             "type": "LIMIT",
#             "quantity": str(quantity_l),
#             "timeInForce": "GTC",
#             "price": stop_price
#         }
#
#         response = send_signed_request('POST', '/fapi/v1/order', params)
#         print(response)