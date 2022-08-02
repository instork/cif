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
DATA_DIR = os.path.join(ROOT, 'data')

from datasets import Dataset
from torch.utils.data import Dataset as BaseDataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from typing import Callable, Union, Dict, List
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

MAX_LEN = 256
VALID_SPLIT = 0.1


class CoinDataset(BaseDataset):
    """
    ### Base dataset for coin data
    """
    def __init__(self, data: str, columns: List[str], problem_type='multi_label_classification',
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END, **kwargs):
        self.set_datetime(start, end)
        self.problem_type = problem_type
        self.dataset = self.load_dataframe(data, columns)
        self.dataset = self.label_dataframe(self.dataset)
        self.id2label = {idx:label for idx, label in enumerate(self.dataset.target.unique())}
        self.label2id = {label:idx for idx, label in enumerate(self.dataset.target.unique())}

    def set_datetime(self, start: Union[dt.datetime,str], end: Union[dt.datetime,str]):
        """
        #### Set start/end date with format like 2020-01-01 or using datetime object.
        """
        self.start = strptime(start) if start else strptime(DEFAULT_START)
        self.end = strptime(end) if end else strptime(DEFAULT_END)

    def load_dataframe(self, data: str, columns: List[str]) -> pd.DataFrame:
        """
        #### Load .csv file from data directory with selected columns.
        Value types in dataframe are expected only list or datetime.
        """
        df = pd.read_csv(os.path.join(DATA_DIR, data))[columns]
        for column in columns:
            df[column] = df[column].apply(lambda x: strptime(x) if re.match('^\d+-\d+-\d+$',x) else eval(x))
        return df[(df['etz_time']>=self.start)&(df['etz_time']<=self.end)]

    def label_dataframe(self, df: pd.DataFrame, drop_price=True) -> pd.DataFrame:
        """
        ### Label by one day later fluctuation percentage.
        [multi_label_classification]: -4%, -2%, 2%, 4%
        [single_label_classification]: Up, Down
        """
        open_price = df['open'].apply(lambda x: x[0])
        close_price = df['close'].apply(lambda x: x[-1])
        df['target'] = (((close_price - open_price) / open_price) * 100.).shift(-1)
        df.dropna(how='any', inplace=True)
        df['target'] = self._classify_labels(df['target'])
        df = df.drop(['open','close'], axis=1) if drop_price else df
        return df

    def preprocess(self, df: pd.DataFrame, **kwargs) -> Dataset:
        raise NotImplementedError

    def _classify_labels(self, target: pd.Series) -> pd.Series:
        if self.problem_type == 'multi_label_classification':
            return np.where(target <= -4., '-4%',
                            np.where(target < 0., '-2%',
                            np.where(target < 4., '+2%', '+4%')))
        elif self.problem_type == 'single_label_classification':
            return np.where(target < 0., 'Down', 'Up')
        else:
            raise ValueError('Unexpected problem type entered!')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> pd.Series:
        return self.dataset.iloc[index]


class NewsDataset(CoinDataset):
    """
    ### Dataset for Bitcoin and Ethereum news data
    [multi_label_classification]: -4%, -2%, 2%, 4%
    [single_label_classification]: Up, Down
    """
    def __init__(self, data='btc.csv', problem_type='multi_label_classification', preprocessed=False,
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END, **kwargs):
        news_columns = ['etz_time','news','open','close']
        self.problem_type = problem_type
        self.dataset = pd.DataFrame()
        super().__init__(data, news_columns, problem_type, start, end, **kwargs)
        self.dataset = self.preprocess(self.dataset, **kwargs) if preprocessed else self.dataset

    def preprocess(self, df: pd.DataFrame, model_path: str, max_len=MAX_LEN,
                    valid_split=VALID_SPLIT, labeled=True, **kwargs) -> Dataset:
        """
        #### Preprocess news data with tokenization and label encoding.
        """
        old_columns = df.columns.tolist()
        df = self._unlist_news(df)

        dataset = Dataset.from_pandas(df)
        dataset = dataset.train_test_split(valid_split)
        dataset = dataset.map(self._tokenize(model_path, max_len), batched=True)
        dataset = dataset.map(self._label_encoding) if labeled else dataset
        dataset = dataset.remove_columns(old_columns)
        dataset.set_format('torch')
        return dataset

    def _unlist_news(self, df: pd.DataFrame) -> pd.DataFrame:
        df_unlisted = df.iloc[[]].copy()
        for idx in range(len(df)):
            df_repeated = pd.concat([df.iloc[[idx]]]*len(df.iloc[idx].news))
            df_repeated.news = df_repeated.iloc[0].news
            df_unlisted = pd.concat([df_unlisted, df_repeated])
        return df_unlisted

    def _tokenize(self, model_path: str, max_len: int) -> Callable[[Dataset],BatchEncoding]:
        tokenizer = AutoTokenizer.from_pretrained(model_path, problem_type=self.problem_type)
        return lambda examples: tokenizer(examples['news'], max_length=max_len, padding='max_length', truncation=True)

    def _label_encoding(self, examples: Dataset) -> Dict[str,np.ndarray]:
        if self.problem_type == 'multi_label_classification':
            return {'labels':np.eye(len(self.label2id))[self.label2id[examples['target']]]}
        elif self.problem_type == 'single_label_classification':
            return {'labels':self.label2id[examples['target']]}


class OhlcDataset(CoinDataset):
    """
    ### Dataset for Bitcoin and Ethereum price data
    [multi_label_classification]: -4%, -2%, 2%, 4%
    [single_label_classification]: Up, Down
    """
    def __init__(self, data='btc.csv', problem_type='multi_label_classification', preprocessed=False,
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END, **kwargs):
        ohlc_columns = ['etz_time','open','high','low','close','volume']
        self.problem_type = problem_type
        self.dataset = pd.DataFrame()
        super().__init__(data, ohlc_columns, problem_type, start, end, **kwargs)

    def preprocess(self):
        pass


class MongoDataset(BaseDataset):
    """
    ### MongoDB Dataset for coin data
    You can limit start/end date with format like 2020-01-01 or using datetime object.
    """
    def __init__(self, db_name='test_db', target='btc',
                start: Union[dt.datetime,str]=DEFAULT_START, end: Union[dt.datetime,str]=DEFAULT_END):
        self.set_datetime(start, end)
        self.dataset = self.load_mongo_data(db_name, target)

    def set_datetime(self, start: Union[dt.datetime,str], end: Union[dt.datetime,str]):
        """
        #### Set start/end date with format like 2020-01-01 or using datetime object.
        """
        self.start = dt.datetime.strptime(start, DATETIME_FORMAT) if isinstance(start,str) else start
        self.end = dt.datetime.strptime(end, DATETIME_FORMAT) if isinstance(end,str) else end

    def load_mongo_data(self, db_name: str, target: str) -> pd.DataFrame:
        """
        #### Load coin dataframe from remote MongoDB, requires 'data/mongo_env'.
        """
        client = MongoClient(self._load_mongo_url())
        db = client[db_name]
        price_df = self._load_price(db, target)
        news_df = self._load_news(db, target)
        client.close()
        return news_df.merge(price_df, left_on='etz_time', right_on='etz_time')

    def save_mongo_data(self, path: str, include_index=False):
        """
        #### Save coin dataframe to csv file.
        """
        self.dataset.to_csv(path, index=include_index)

    def _load_mongo_url(self) -> str:
        load_dotenv(os.path.join(DATA_DIR, 'mongo_env'))
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

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> pd.Series:
            return self.dataset.iloc[index]
