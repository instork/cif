import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

from dataset import NewsDataset, OhlcDataset
from transformers import EarlyStoppingCallback
from model.model import Transformers
from model.metric import compute_metrics
from trainer import CoinTrainer

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def temp_news_main(config: ConfigParser):
    logger = config.get_logger('train')

    # setup dataset instances
    dataset = NewsDataset(**config['dataset'], **config['model'])
    train_dataset = dataset.dataset['train']
    eval_dataset = dataset.dataset['test']
    config['model']['id2label'] = dataset.id2label
    config['model']['label2id'] = dataset.label2id

    # # build model architecture, then print to console
    transformers = Transformers(**config['dataset'], **config['model'])
    logger.info(transformers.model)

    # GPU training for CUDA and MPS
    device = torch.device('cuda') if torch.cuda.is_available() else \
             torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    transformers.model.to(device)

    # # build trainer
    config['trainer']['output_dir'] = config._save_dir
    config['trainer']['logging_dir'] = config._log_dir
    del config['trainer']['save_dir']
    trainer = CoinTrainer(train_dataset, eval_dataset, transformers.model, compute_metrics,
                            callbacks=[EarlyStoppingCallback(3)], **config['trainer'])

    trainer.train()


def main(config: ConfigParser):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)', dest='config')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)', dest='resume')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)', dest='device')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    temp_news_main(config)
    # main(config)
