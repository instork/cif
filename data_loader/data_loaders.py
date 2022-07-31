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
from typing import Union
import datetime as dt
import numpy as np
import pandas as pd

ALIAS = {
    'market': lambda x: 'USDT-'+str(x).upper(),
    'news': lambda x: str(x).upper()+'-News',
}

AGG_OPTION = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'market': 'first'}

DATETIME_FORMAT = '%Y/%m/%d'
DEFAULT_START = '2020/01/01'
DEFAULT_END = dt.datetime.today()


class CoinDataLoader(BaseDataLoader):
    """
    Dataloader for Bitcoin and Ethereum prices with news data
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END):
        self.data_dir = data_dir
        self.set_datetime(start, end)
        self.dataset = self.load_dataframe()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def set_datetime(self, start: Union[dt.datetime,str], end: Union[dt.datetime,str]):
        """
        Set start/end date with format like 2020/01/01 or using datetime object.
        """
        self.start = dt.datetime.strptime(start, DATETIME_FORMAT) if isinstance(start,str) else start
        self.end = dt.datetime.strptime(end, DATETIME_FORMAT) if isinstance(end,str) else end

    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir)
        df['news'] = df['news'].apply(lambda x: eval(x))
        return df


class NewsDataLoader(BaseDataLoader):
    """
    Dataloader for Bitcoin and Ethereum prices with news data
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END):
        self.data_dir = data_dir
        self.set_datetime(start, end)
        self.dataset = self.load_dataframe()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def set_datetime(self, start: Union[dt.datetime,str], end: Union[dt.datetime,str]):
        """
        Set start/end date with format like 2020/01/01 or using datetime object.
        """
        self.start = dt.datetime.strptime(start, DATETIME_FORMAT) if isinstance(start,str) else start
        self.end = dt.datetime.strptime(end, DATETIME_FORMAT) if isinstance(end,str) else end

    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir)
        df['news'] = df['news'].apply(lambda x: eval(x))
        return df


class MongoDataLoader(BaseDataLoader):
    """
    MongoDB Dataloader for coin data.
    You can limit start/end date with format like 2020/01/01 or using datetime object.
    """
    def __init__(self, batch_size=64, shuffle=True, validation_split=0.0, num_workers=1, db_name='test_db', target='btc',
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END):
        self.set_datetime(start, end)
        self.dataset = self.load_mongo_data(db_name, target)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def set_datetime(self, start: Union[dt.datetime,str], end: Union[dt.datetime,str]):
        """
        Set start/end date with format like 2020/01/01 or using datetime object.
        """
        self.start = dt.datetime.strptime(start, DATETIME_FORMAT) if isinstance(start,str) else start
        self.end = dt.datetime.strptime(end, DATETIME_FORMAT) if isinstance(end,str) else end

    def load_mongo_data(self, db_name: str, target: str) -> pd.DataFrame:
        """
        Load coin dataframe from MongoDB, requires data/.env file.
        """
        client = MongoClient(self._load_mongo_url())
        db = client[db_name]
        price_df = self._load_price(db, target)
        news_df = self._load_news(db, target)
        client.close()
        return news_df.merge(price_df, left_on='etz_time', right_on='etz_time')

    def save_mongo_data(self, path: str, include_index=False):
        """
        Save coin dataframe to csv file.
        """
        self.dataset.to_csv(path, index=include_index)

    def _load_mongo_url(self) -> str:
        load_dotenv(os.path.join(ROOT, 'data/.env'))
        user = os.getenv("MONGODB_USER")
        pwd = os.getenv("MONGODB_PWD")
        host = os.getenv("MONGODB_HOST")
        port = os.getenv("MONGODB_PORT")
        return f'mongodb://{user}:{pwd}@{host}:{port}'

    def _etz_timezone(self, timestamp, time_diff=-5) -> dt.datetime:
        return timestamp + dt.timedelta(hours=time_diff)
        # return timestamp.tz_localize('UTC').tz_convert('US/Eastern')

    def _load_price(self, db: Database, target: str) -> pd.DataFrame:
        price_df = pd.DataFrame(list(db[ALIAS['market'](target)].find(
                    {"utc_time":{"$gte":self.start, "$lt":self.end+dt.timedelta(1)}})))

        price_df['etz_time'] = price_df['utc_time'].apply(lambda x: self._etz_timezone(x))
        price_df = price_df[['etz_time','opening_price','high_price','low_price','trade_price','candle_acc_trade_volume','market']]
        price_df.columns = ['etz_time','open','high','low','close','volume','market']
        price_df['etz_time'] = price_df['etz_time'].apply(lambda x: x.date())

        price_df = price_df.groupby('etz_time').agg(AGG_OPTION)
        etz_period = (price_df.index>=self.start.date())&(price_df.index<self.end.date())
        price_df = price_df[etz_period].reset_index()
        return price_df

    def _load_news(self, db: Database, target: str) -> pd.DataFrame:
        news_df = pd.DataFrame(list(db['news'].find(
                    {"etz_time":{"$gte":self.start, "$lt":self.end}})))

        news_df = news_df[['etz_time',ALIAS['news'](target)]]
        news_df.columns = ['etz_time','news']
        news_df['etz_time'] = news_df['etz_time'].apply(lambda x: x.date())
        return news_df


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
