import os, time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import argparse, time, logging
import matplotlib.pyplot as plt

import numpy as np
import mxnet as mx
import math

from mxnet import gluon, nd
from mxnet import autograd as ag
from darknet3C import DarkNet

from gluoncv.utils import makedirs, TrainingHistory



# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')

    # Training
    parser.add_argument('--batch-size', type=int, default=128, help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=4, help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='darknet', help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-workers', dest='num_workers', default=16, type=int, help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0, help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60', help='epochs at which learning rate decays. default is 40,60.')

    parser.add_argument('--save-period', type=int, default=5, help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params', help='directory of saved models')
    parser.add_argument('--resume-from', type=str, help='resume training from the model')
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd, nag')

    # Resume
    parser.add_argument('--resume-epoch', type=int, default=0, help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='', help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='', help='path of trainer state to load from.')

    # Dataset
    parser.add_argument('--dataset-path', type=str, default='ITW_rec', help='root folder should be .mxnet/datasets/')
    parser.add_argument('--img-size', type=int, default=224, help='image size')
    parser.add_argument('--crop-ratio', type=float, default=0.875, help='Crop ratio during validation. default is 0.875')

    # Logging
    parser.add_argument('--log-interval', type=int, default=10, help='Number of batches to wait before logging.')
    parser.add_argument('--save-plot-dir', type=str, default='.', help='the path to save the history plot')
    parser.add_argument('--logging-file', type=str, default='train_darknet3C.log', help='name of training log file')
    parser.add_argument('--model-name', type=str, default='darknet3C', help='model name for save exp')
    opt = parser.parse_args()
    return opt

def get_data_rec(opt, ctx):
    dataset_path = opt.dataset_path
    batch_size = opt.batch_size
    img_size= opt.img_size
    num_workers = opt.num_workers
    rec_path = os.path.join('/home/seungtaek/.mxnet/datasets/', dataset_path)

    # Augmentation Params
    jitter_param = 0.4
    lighting_param = 0.1
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(img_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(rec_path, 'train.rec'),
        path_imgidx=os.path.join(rec_path, 'train.idx'),
        preprocess_threads=num_workers,
        data_shape=(3, img_size, img_size),
        batch_size=batch_size,
        shuffle=True,

        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
        rand_mirror=True,
        random_resized_crop=True,
        max_aspect_ratio=4. / 3.,
        min_aspect_ratio=3. / 4.,
        max_random_area=1,
        min_random_area=0.08,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
    )

    val_data = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(rec_path, 'val.rec'),
        path_imgidx=os.path.join(rec_path, 'val.idx'),
        preprocess_threads=num_workers,
        data_shape=(3, img_size, img_size),
        batch_size=batch_size,
        shuffle=False,

        resize=resize,
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )

    return train_data, val_data, batch_fn

def main():
    now = time.localtime()
    now = f'{now.tm_year}{now.tm_mon:0>2}{now.tm_mday:0>2}{now.tm_hour:0>2}{now.tm_min:0>2}{now.tm_sec:0>2}'

    opt = parse_args()
    model_name = opt.model_name
    filehandler = logging.FileHandler(f'log/{model_name}_{now}.log')
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)

    batch_size = opt.batch_size
    classes = 5 # {sunny, cloudy, foggy, rain, snow}
    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    print(context)

    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

    # model define
    net = DarkNet(num_classes=classes, input_size=opt.img_size)

    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx = context)
    optimizer = opt.optimizer

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0

    plot_path = opt.save_plot_dir

    # load dataset
    train_data, val_data, batch_fn = get_data_rec(opt, context)

    def test(ctx, val_data):
        val_data.reset()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        metric = mx.metric.Accuracy()

        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            outputs = [net(X) for X in data]
            val_loss = [loss_fn(yhat, y) for yhat, y in zip(outputs, label)]

            #val_loss = (val_loss + sum([l.sum().asscalar() for l in val_loss])) / (i + 1)
            metric.update(label, outputs)

        return metric.get() # [0], metric.get()[1], val_loss

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]

        if opt.resume_params == '':
            net.initialize(mx.init.Xavier(), ctx=ctx)
        else:
            net.load_parameters(opt.resume_params, ctx=ctx)

        trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})

        if opt.resume_states != '':
            trainer.load_states(opt.resume_states)


        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        history_acc = TrainingHistory(['train-acc', 'val-acc', 'train-loss'])

        lr_decay_count = 0
        iteration = 0
        best_val_score = 0
        max_batch = 0

        for epoch in range(opt.resume_epoch, opt.num_epochs):
            tic = time.time()
            train_data.reset()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            btic = time.time()

            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate*lr_decay)
                lr_decay_count += 1

            for i, batch in enumerate(train_data):
                data, label = batch_fn(batch, ctx)

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_metric.update(label, output)

                iteration += 1
                train_loss = (train_loss + sum([l.sum().asscalar() for l in loss])) / iteration


                if opt.log_interval and not (i + 1) % opt.log_interval:
                    if i > max_batch: max_batch = i
                    train_metric_name, train_metric_score = train_metric.get()
                    logger.info('Epoch[%d/%d] Batch [%d/%d]\tSpeed: %f samples/sec\ttrain_loss=%f\t%s=%f\tlr=%f' % (
                        epoch, opt.num_epochs, i, max_batch, batch_size * opt.log_interval / (time.time() - btic),
                        train_loss, train_metric_name, train_metric_score, trainer.learning_rate))
                    btic = time.time()

            name, train_acc = train_metric.get()
            name, val_acc = test(ctx, val_data)

            history_acc.update([train_acc, val_acc, train_loss])
            history_acc.plot(save_path=f'graphs/{model_name}_{now}_{epoch:0>3}_acc_history.png')

            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters(f'params/{model_name}_{now}_{epoch:0>3}(best).params')
                trainer.save_states(f'params/{model_name}_{now}_{epoch:0>3}(best).states')

            logging.info('[Epoch %d] train=%f val=%f time: %f' %
                (epoch, train_acc, val_acc, time.time()-tic))

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                net.save_parameters(f'params/{model_name}_{now}_{epoch:0>3}.params')
                trainer.save_states(f'params/{model_name}_{now}_{epoch:0>3}.states')

        if save_period and save_dir:
            net.save_parameters(f'params/{model_name}_{now}_{epoch-1:0>3}.params')
            trainer.save_states(f'params/{model_name}_{now}_{epoch-1:0>3}.states')

    train(context)

if __name__ == '__main__':
    main()