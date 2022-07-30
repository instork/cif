import argparse
import contextlib
import os
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from torchvision import datasets, transforms
from base import BaseDataLoader
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
import datetime as dt
import numpy as np
import pandas as pd

ALIAS = {
    'btc': 'USDT-BTC',
    'eth':'USDT-ETH',
}

AGG_OPTION = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'market': 'first'}

DATETIME_FORMAT = '%Y/%m/%d'


class CoinDataLoader(BaseDataLoader):
    """
    Bitcoin and Ethereum price and news data loading
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                db_name='test_db', target='btc', start='2020/01/01', end='2021/06/01'):
        self.client = MongoClient(self.load_mongo_url())
        self.db = self.load_database(db_name)
        self.start = dt.datetime.strptime(start, DATETIME_FORMAT)
        self.end = dt.datetime.strptime(end, DATETIME_FORMAT)
        self.dataset = self.load_coin_data(target)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def load_mongo_url(self) -> str:
        load_dotenv(os.path.join(ROOT, 'data/.env'))
        user = os.getenv("MONGODB_USER")
        pwd = os.getenv("MONGODB_PWD")
        host = os.getenv("MONGODB_HOST")
        port = os.getenv("MONGODB_PORT")
        return f'mongodb://{user}:{pwd}@{host}:{port}'

    def load_database(self, db_name: str) -> Database:
        return self.client[db_name]

    def etz_timezone(self, timestamp, time_diff=-5) -> dt.datetime:
        return timestamp + dt.timedelta(hours=time_diff)
        # return timestamp.tz_localize('UTC').tz_convert('US/Eastern')

    def load_coin_data(self, target: str) -> pd.DataFrame:
        price_df = self.load_price(target)
        news_df = self.load_news()
        return news_df.merge(price_df, left_on='etz_time', right_on='etz_time')

    def load_price(self, target: str) -> pd.DataFrame:
        price_df = pd.DataFrame(list(self.db[ALIAS[target]].find(
                    {"utc_time":{"$gte":self.start, "$lt":self.end+dt.timedelta(1)}})))

        price_df['etz_time'] = price_df['utc_time'].apply(lambda x: self.etz_timezone(x))
        price_df = price_df[['etz_time','opening_price','high_price','low_price','trade_price','candle_acc_trade_volume','market']]
        price_df.columns = ['etz_time','open','high','low','close','volume','market']
        price_df['etz_time'] = price_df['etz_time'].apply(lambda x: x.date())

        price_df = price_df.groupby('etz_time').agg(AGG_OPTION)
        etz_period = (price_df.index>=self.start.date())&(price_df.index<self.end.date())
        price_df = price_df[etz_period].reset_index()
        return price_df

    def load_news(self) -> pd.DataFrame:
        news_df = pd.DataFrame(list(self.db['news'].find(
                    {"etz_time":{"$gte":self.start, "$lt":self.end}})))

        news_df.drop(['_id'], axis=1, inplace=True)
        news_df['etz_time'] = news_df['etz_time'].apply(lambda x: x.date())
        return news_df

    def close_client(self):
        self.client.close()


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
