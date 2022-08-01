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
from typing import Union, List
import datetime as dt
import numpy as np
import pandas as pd
import re

ALIAS = {
    'market': lambda x: 'USDT-'+str(x).upper(),
    'news': lambda x: str(x).upper()+'-News',
}

AGG_OPTION = {
    'open': lambda x: list(x), # 'first'
    'high': lambda x: list(x), # 'max'
    'low': lambda x: list(x), # 'min'
    'close': lambda x: list(x), # 'last'
    'volume': lambda x: list(x), # 'sum'
    'market': 'first'}

DATETIME_FORMAT = '%Y-%m-%d'
DEFAULT_START = '2020-01-01'
DEFAULT_END = dt.datetime.today()
strptime = lambda datetime: dt.datetime.strptime(datetime, DATETIME_FORMAT) if isinstance(datetime,str) else datetime


class CoinDataLoader(BaseDataLoader):
    """
    Dataloader interface for coin data
    """

    def set_datetime(self, start: Union[dt.datetime,str], end: Union[dt.datetime,str]):
        """
        Set start/end date with format like 2020-01-01 or using datetime object.
        """
        self.start = strptime(start)
        self.end = strptime(end)

    def load_dataframe(self, data_dir: str, columns: List[str]) -> pd.DataFrame:
        """
        Load .csv file from data_dir with selected columns.
        Value types in dataframe are expected only list or datetime.
        """
        df = pd.read_csv(data_dir)[columns]
        for column in columns:
            df[column] = df[column].apply(lambda x: strptime(x) if re.match('^\d+-\d+-\d+$',x) else eval(x))
        return df

    def label_dataframe(self, df: pd.DataFrame, problem_type: str, drop_price=True) -> pd.DataFrame:
        """
        Label by one day later fluctuation percentage.
        [multi_label_classification]: -4%, -2%, 2%, 4%
        [single_label_classification]: Up, Down
        """
        open_price = df['open'].apply(lambda x: x[0])
        close_price = df['close'].apply(lambda x: x[-1])
        df['target'] = (((close_price - open_price) / open_price) * 100.).shift(-1)
        df.dropna(how='any', inplace=True)
        df['target'] = self._classify_labels(df['target'], problem_type)
        df = df.drop(['open','close'], axis=1) if drop_price else df
        return df

    def _classify_labels(self, target: pd.Series, problem_type) -> pd.Series:
        if problem_type == 'multi_label_classification':
            return np.where(target <= -4., '-4%',
                            np.where(target < 0., '-2%',
                            np.where(target < 4., '+2%', '+4%')))
        elif problem_type == 'single_label_classification':
            return np.where(target < 0., 'Down', 'Up')
        else:
            raise Exception('Unexpected problem type entered!')


class NewsDataLoader(CoinDataLoader):
    """
    Dataloader for Bitcoin and Ethereum news data
    [multi_label_classification]: -4%, -2%, 2%, 4%
    [single_label_classification]: Up, Down
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END,
                problem_type='multi_label_classification'):
        self.set_datetime(start, end)
        self.dataset = self.load_dataframe(data_dir, ['etz_time','news','open','close'])
        self.dataset = self.label_dataframe(self.dataset, problem_type)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def preprocess(self):
        pass


class OhlcDataLoader(CoinDataLoader):
    """
    Dataloader for Bitcoin and Ethereum price data
    [multi_label_classification]: -4%, -2%, 2%, 4%
    [single_label_classification]: Up, Down
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END,
                problem_type='multi_label_classification'):
        self.set_datetime(start, end)
        self.dataset = self.load_dataframe(data_dir, ['etz_time','open','high','low','close','volume'])
        self.dataset = self.label_dataframe(self.dataset, problem_type, drop_price=False)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def preprocess(self):
        pass


class MongoDataLoader(BaseDataLoader):
    """
    MongoDB Dataloader for coin data.
    You can limit start/end date with format like 2020-01-01 or using datetime object.
    """
    def __init__(self, batch_size=64, shuffle=True, validation_split=0.0, num_workers=1, db_name='test_db', target='btc',
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END):
        self.set_datetime(start, end)
        self.dataset = self.load_mongo_data(db_name, target)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def set_datetime(self, start: Union[dt.datetime,str], end: Union[dt.datetime,str]):
        """
        Set start/end date with format like 2020-01-01 or using datetime object.
        """
        self.start = dt.datetime.strptime(start, DATETIME_FORMAT) if isinstance(start,str) else start
        self.end = dt.datetime.strptime(end, DATETIME_FORMAT) if isinstance(end,str) else end

    def load_mongo_data(self, db_name: str, target: str) -> pd.DataFrame:
        """
        Load coin dataframe from remote MongoDB, requires 'data/.env' file.
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
