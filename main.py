
import krakenex
import numpy as np
import time
import csv
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

#By default, lookback is set to 50, and the max that the bot can spend per buy order is $33.0.

class KrakenBot:
    def __init__(self, api_key, api_secret, pair='XXBTZUSD', lookback=50, max_usd_per_order=33, sell_percentage=0.25, stop_loss_pct=0.10, csv_file='trading_log.csv'):
        self.api = krakenex.API(key=api_key, secret=api_secret)
        self.pair = pair
        self.lookback = lookback
        self.max_usd_per_order = max_usd_per_order
        self.sell_percentage = sell_percentage
        self.stop_loss_pct = stop_loss_pct
        self.csv_file = csv_file
        self.usd_balance = 0
        self.btc_balance = 0
        self.initialize_csv()
#This initializes the CSV save file.
    def initialize_csv(self):
        try:
            with open(self.csv_file, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Predicted Price', 'Close Price', 'Difference', 'Action', 'Volume', 'Expenditure', 'Profit', 'Total USD Balance', 'Total BTC Balance'])
        except FileExistsError:
            pass
#This gets data. Total data is calculated by lookback times interval. So right now, the program looks back over 50 1 minute intervals to get data for the ARIMA algo.
    def fetch_data(self):
        ohlc = self.api.query_public('OHLC', {'pair': self.pair, 'interval': 1})
        if ohlc.get('error'):
            print(f"Error fetching OHLC data: {ohlc['error']}")
            return None
        return ohlc['result'][self.pair]
#Does the actual ARIMA calculation and returns the next predicted price.
    def calculate_arima(self, data):
        closes = np.array([float(d[4]) for d in data])
        model = ARIMA(closes, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast[0]
#Gets both BTC and USD balances.
    def get_balance(self):
        balance = self.api.query_private('Balance')
        if balance.get('error'):
            print(f"Error fetching balance: {balance['error']}")
            return 0, 0
        usd_balance = float(balance['result'].get('ZUSD', 0))
        btc_balance = float(balance['result'].get('XXBT', 0))
        return usd_balance, btc_balance
#Places order, whether it's a sell or buy.
    def place_order(self, order_type, volume):
        response = self.api.query_private('AddOrder', {
            'pair': self.pair,
            'type': order_type,
            'ordertype': 'market',
            'volume': volume
        })
        if response.get('error'):
            print(f"Error placing order: {response['error']}")
            return None
        return response['result']
#Calculates stoploss for each buy.
    def place_stop_loss(self, stop_price, volume):
        stop_price = round(stop_price, 1)
        response = self.api.query_private('AddOrder', {
            'pair': self.pair,
            'type': 'sell',
            'ordertype': 'stop-loss',
            'price': stop_price,
            'volume': volume
        })
        if response.get('error'):
            print(f"Error placing stop-loss order: {response['error']}")
            return None
        return response['result']
#Logs each trade into the CSV file.
    def log_trade(self, predicted_price, last_close, action, volume, expenditure, profit):
        difference = predicted_price - last_close
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        usd_balance, btc_balance = self.get_balance()
        self.usd_balance = usd_balance
        self.btc_balance = btc_balance
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, predicted_price, last_close, difference, action, volume, expenditure, profit, usd_balance, btc_balance])
#Manages the order in which the program runs.
    def run(self):
        while True:
            data = self.fetch_data()
            if data is None:
                print("No data fetched, exiting.")
                return

            data = data[-self.lookback:]
            predicted_price = self.calculate_arima(data)
            last_close = float(data[-1][4])
            difference = abs(predicted_price - last_close)

            print(f"Last close price: {last_close}, Predicted price: {predicted_price}, Difference: {difference}")
            #Difference is the size between the predicted price and actual price, if it's below 15 it's considered a weak prediction and no action should be taken.
            if difference > 15:
                #If the price was lower than predicted, buy.
                if predicted_price > last_close:
                    usd_balance, _ = self.get_balance()
                    print(f"USD Balance: {usd_balance}")
                    if usd_balance > 0:
                        max_volume = self.max_usd_per_order / last_close
                        volume = min(max_volume, usd_balance / last_close)
                        if volume > 0.0001:
                            order_result = self.place_order('buy', volume)
                            if order_result:
                                print(f"Buying BTC: {volume} at {last_close}")
                                expenditure = volume * last_close
                                stop_price = round(last_close * (1 - self.stop_loss_pct), 1)
                                stop_loss_result = self.place_stop_loss(stop_price, volume)
                                if stop_loss_result:
                                    print(f"Placed stop-loss order at {stop_price}")
                                self.log_trade(predicted_price, last_close, 'buy', volume, expenditure, 0)
                #If the price was higher than predicted, sell.
                elif predicted_price < last_close:
                    _, btc_balance = self.get_balance()
                    print(f"BTC Balance: {btc_balance}")
                    if btc_balance > 0.0001:
                        portion_volume = btc_balance * self.sell_percentage
                        volume_to_sell = max(min(portion_volume, btc_balance), 0.0001)
                        if volume_to_sell > 0.0001:
                            print(f"Selling BTC: {volume_to_sell} at {last_close}")
                            order_result = self.place_order('sell', volume_to_sell)
                            if order_result:
                                profit = volume_to_sell * last_close
                                self.log_trade(predicted_price, last_close, 'sell', volume_to_sell, 0, profit)
                            else:
                                print("Failed to place sell order.")

            print("Waiting for the next cycle...\n\n")
            #Program sleeps for 60 minutes between cycles.
            time.sleep(60 * 60)

#Main executes the program and gets keys from the user.
if __name__ == "__main__":
    api_key = input('Enter API Key (Public): ')
    api_secret = input('Enter Secret Key: ')
    bot = KrakenBot(api_key, api_secret)
    bot.run()
