import configparser
import logging
from datetime import datetime
from pathlib import Path
from itertools import count
import os
import argparse

import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data.dataset import CUBDataset, BUFGDataset
from model.macnn import MACNNFCN, MACNNLoss
from util import AccumLoss, images_denorm, plot_attn_maps, plot_conf_mat, compute_metrics, \
    compute_feat_shape, plot_attn_maps_or_images


def train_func(config, n_loss, e_name=None, log_name=None):
    """
    MACNN training function

    Args:
        config (str): path to the configuration file
        n_loss (int): number of losses terms (for visualization)
        e_name (str): experiment namegt
        log_name (str): log file name
    """
    if e_name is None:
        e_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if log_name is None:
        log_name = 'exp.log'

    # config parse
    parser = configparser.ConfigParser()
    parser.read(config)
    data_options = parser['data']
    lparams = parser['optimizer']
    net_config = parser['net']
    hparams = parser['parameter']

    # setup paths
    save_dir = Path(data_options.get('save_dir')).expanduser()
    exp_save_dir = save_dir / e_name
    log_dir = exp_save_dir / 'log'
    backup_dir = exp_save_dir / 'backup'
    log_file = log_dir / log_name
    for d in [save_dir, exp_save_dir, log_dir, backup_dir]:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)

    # setup logger
    log_config = {'level': logging.INFO,
                  'format': '{asctime:s} {levelname:<8s} {filename:<12s} : {message:s}',
                  'datefmt': '%Y-%m-%d %H:%M:%S',
                  'filename': log_file,
                  'filemode': 'w',
                  'style': '{'}
    logging.basicConfig(**log_config)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    cfmt = logging.Formatter('{asctime:s} : {message:s}',
                             style='{', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(cfmt)
    logger = logging.getLogger(__name__)
    logger.addHandler(console)
    # write raw config to logger
    with open(config) as cfg:
        logger.info('###### Config file content ######\n')
        for line in cfg:
            logger.info(line.strip())
        logger.info('#' * 8 + '\n')
    logger.info('The backup, final weights and the log will be saved to '
                '... {}'.format(exp_save_dir))

    # parameters
    save_interval = hparams.getint('save_interval')
    batch_size = hparams.getint('batch_size')
    print_batch = hparams.getint('print_batch')
    N = net_config.getint('num_attn')
    input_size = np.fromstring(net_config.get('input_size'), sep=',', dtype=int).tolist()
    num_class = net_config.getint('num_class')

    # writer setup
    writer = SummaryWriter(log_dir)
    global_step = count()

    # random seed for reproducability
    seed = hparams.getint('seed', fallback=1234)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # input transform


    # device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('The script is working on {}'.format(device))

    # load dataset
    kwargs = {'num_workers': 2, 'pin_memory': True}
    if data_options.get('data_name') == 'CUB':
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_set = CUBDataset(data_options.get('data_root'), data_options.get('meta_dir'),
                           'train', transform)
        test_set = CUBDataset(data_options.get('data_root'), data_options.get('meta_dir'),
                          'test', transform)
    elif data_options.get('data_name') == 'mini':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        train_set = MiniImageNet(phase='train')
        test_set = train_set
    elif data_options.get('data_name') == 'BU':
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_set = BUFGDataset(data_options.get('data_root'), train=True, transform=transform)
        test_set = BUFGDataset(data_options.get('test_root'), train=False, transform=transform)
    else:
        print("no data named ",data_options.get('data_name'))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, **kwargs)
    logger.info('Data set loaded!')

    # create model & loss function
    model = MACNNFCN(net_config.get('backbone'), N, num_class)
    model = model.to(device)
    loss_weights = np.fromstring(hparams.get('loss_weights'), dtype=float, sep=',')
    loss_func = MACNNLoss(N, compute_feat_shape(tuple(input_size), net_config.get('backbone')),
                          hparams.getfloat('margin'), loss_weights)
    logger.info('Model created!')

    # set optimizer
    epochs = lparams.getint('epochs')
    lr = lparams.getfloat('lr')
    decay = lparams.getfloat('decay')
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=decay)
    step = lparams.getint('exp_step')
    gamma = lparams.getfloat('gamma')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

    losses = AccumLoss(n_loss)

    def vis_losses(losses, legends, step):
        """
        Visualize the losses in tensorboard

        Args:
            losses (Iterable): All the losses for visualization
            legends (Iterable): The legend/label for each loss in `losses`
            step (int): global step
        """
        ldict = {k: v for k, v in zip(legends, losses)}
        writer.add_scalars('Train/Losses', ldict, step)
        writer.flush()

    def vis_attn_maps(data, Ms, n, tag, step):
        """
        Visualize the attention maps with original images

        Args:
            data (Tensor): the input data
            Ms (Tensor): the output attention maps
            n (int): number of visualized examples
            tag (str): tag for tensorboard
            step (int): global step
        """
        if n >= data.shape[0]:
            data_vis = data.detach().cpu().numpy()
            maps_vis = Ms.detach().cpu().numpy()
        else:
            data_vis = data.detach().cpu().numpy()[:n]
            maps_vis = Ms.detach().cpu().numpy()[:n]
        data_vis = images_denorm(data_vis)
        # transpose to numpy image format
        data_vis = data_vis.transpose((0, 2, 3, 1))
        attn_fig = plot_attn_maps(data_vis, maps_vis)
        writer.add_figure(tag, attn_fig, step)
        image_fig = plot_attn_maps_or_images(data_vis, maps_vis)
        writer.add_figure(tag, image_fig, step)

    def vis_all_metrics(metrics_train, metrics_test, epoch, class_names, vis_per_class=False):
        """
        Visualize all the metrics

        Args:
            metrics_train (Iterable): list of evaluating metrics for training set
            metrics_test (Iterable): list of evaluating metrics for test set
            epoch (int): current epoch for global step
            class_names (Iterable): list of all class names
            vis_per_class (bool): if True, visualize the accuracy for each class
        """
        metric_names = ['Acc', 'Class-Avg-Acc']
        for i, name in enumerate(metric_names):
            metric = {'train': metrics_train[i], 'test': metrics_test[i]}
            writer.add_scalars('Accuracy/{}'.format(name), metric, epoch + 1)

        if vis_per_class:
            for j in range(num_class):
                per_class = {'train': metrics_train[2][j], 'test': metrics_test[2][j]}
                writer.add_scalars('Per_Class_Acc/Class_{:d}'.format(j), per_class, epoch + 1)

        conf_mat_fig_train = plot_conf_mat(metrics_train[3], class_names,
                                           fig_size=(17, 15), label_font_size=5,
                                           show_color_bar=True)
        writer.add_figure('Conf Mat/train', conf_mat_fig_train, epoch + 1)
        conf_mat_fig_test = plot_conf_mat(metrics_test[3], class_names,
                                          fig_size=(17, 15), label_font_size=5,
                                          show_color_bar=True)
        writer.add_figure('Conf Mat/test', conf_mat_fig_test, epoch + 1)
        writer.flush()

    @torch.no_grad()
    def evalulate(epoch):
        model.eval()

        def sub_eval(data_loader):
            gts = []
            preds = []
            for idx, (data, target) in enumerate(data_loader):
                data = data.to(device)
                pred = model.predict(data)
                preds.append(pred.cpu().numpy())
                gts.append(target.numpy())
            preds = np.concatenate(preds, axis=0)
            gts = np.concatenate(gts)
            return preds, gts

        # eval on train
        preds_train, gts_train = sub_eval(train_loader)
        metrics_train = compute_metrics(preds_train, gts_train, num_class)
        logger.info('Train metrics: acc: {:.3%}, class-avg-acc: {:.3%}'.format(
            metrics_train[0], metrics_train[1]))

        # eval on test

        preds_test, gts_test = sub_eval(test_loader)
        metrics_test = compute_metrics(preds_test, gts_test, num_class)
        logger.info('Test Metrics: acc: {:.3%}, class-avg-acc: {:.3%}'.format(
            metrics_test[0], metrics_test[1]))

        vis_all_metrics(metrics_train, metrics_test, epoch, np.arange(num_class), False)
        #vis_all_metrics(metrics_train, epoch, np.arange(num_class), False)

    def train():
        for epoch in range(epochs):
            logger.info('------ Epoch {} starts ------'.format(epoch + 1))
            model.train()

            losses.clear_arr()
            logger.info('lr: {:.5f}'.format(optimizer.param_groups[0]['lr']))
            writer.add_scalar('Train/learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)

            for idx, (data, target, *_) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                Ms, pred_logits = model(data)
                loss = loss_func(Ms, pred_logits, target)
                loss.backward()
                losses.accum(list(map(lambda x: x.item(), loss_func.losses)))
                optimizer.step()

                if (idx + 1) % print_batch == 0:
                    gstep = next(global_step)

                    # logging the loss
                    losses.cut()
                    loss_avg = losses.average(print_batch)
                    logger.info('Epoch: {:>3d}, batch: [{}/{} ({:.0%})],'
                                'average loss: Dis: {:.4f}, Div: {:.4f}, '
                                'Cls: {:.4f}'.format(
                                    epoch + 1, idx + 1, len(train_loader),
                                    (idx + 1) / len(train_loader),
                                    loss_avg[0], loss_avg[1], loss_avg[2]))

                    # visualize the loss
                    vis_losses(loss_avg, ['Dis', 'Div', 'Cls'], gstep)

                    # visualize the attention maps
                    vis_attn_maps(data, Ms, 4, 'Train/Attention Maps', gstep)

                    # reallocation
                    losses.update()

            if (epoch + 1) % save_interval == 0:
                torch.save(model.state_dict(), backup_dir / 'epoch_{}.weight'.format(epoch + 1))
                logger.info('Backup weights saved!')
                logger.info('#' * 8)

            # evaluate
            logger.info('### Testing... ###')
            evalulate(epoch)
            logger.info('#' * 8)

            scheduler.step()

        # final save
        torch.save(model.state_dict(), exp_save_dir / 'final.weight')
        logger.info('Final model weights saved!')
        logger.info('#' * 12)
        return model

    train()


if __name__ == '__main__':
    #config = 'cfgs/CUB_FG.cfg'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', default='BU_FG.cfg')
    opt = parser.parse_args()
    config = opt.config_path
    #config = 'cfgs/mini.cfg'
    n_loss = 3
    e_name = 'test-BU'
    train_func(config, n_loss, e_name)
