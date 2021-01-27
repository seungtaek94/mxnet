import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"

import argparse, time, logging

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from darknet import DarkNet
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms, datasets

import gluoncv as gcv
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')

    # Training
    parser.add_argument('--batch-size', type=int, default=64,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=2,
                        help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='darknet',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=16, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')

    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='sgd, nag')

    # Dataset
    parser.add_argument('--dataset-path', type=str, default='ITW_rec',
                        help='root folder should be .mxnet/datasets/')
    parser.add_argument('--img-size', type=int, default=224,
                        help='image size')


    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--save-plot-dir', type=str, default='.',
                        help='the path to save the history plot')
    parser.add_argument('--logging-file', type=str, default='train_ITW.log',
                        help='name of training log file')
    opt = parser.parse_args()
    return opt

def get_data_rec(opt, ctx):
    dataset_path = opt.dataset_path
    batch_size = opt.batch_size
    img_size= opt.img_size
    num_workers = opt.num_workers
    rec_path = os.path.join('/home/seungtaek/.mxnet/datasets/', dataset_path)

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
        shuffle=True
    )

    val_data = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(rec_path, 'val.rec'),
        path_imgidx=os.path.join(rec_path, 'val.idx'),
        preprocess_threads=num_workers,
        data_shape=(3, img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    return train_data, val_data, batch_fn

def main():
    opt = parse_args()

    filehandler = logging.FileHandler(opt.logging_file)
    streamhandler = logging.StreamHandler()

    #logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)


    logger.info(opt)

    batch_size = opt.batch_size
    classes = 5 # {sunny, cloudy, foggy, rain, snow}
    model_name = 'darknet53'

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    print(context)
    num_workers = opt.num_workers

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
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.Xavier(), ctx=ctx)

        trainer = gluon.Trainer(net.collect_params(), optimizer,
                                {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        train_history = TrainingHistory(['training-error', 'validation-error'])

        lr_decay_count = 0
        iteration = 0
        best_val_score = 0
        max_batch = 0

        for epoch in range(epochs):
            tic = time.time()
            train_data.reset()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            #num_batch = len(train_data)
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
                    logger.info('Epoch[%d/%d] Batch [%d/%d]\tSpeed: %f samples/sec\tloss=%f\t%s=%f\tlr=%f' % (
                        epoch, opt.num_epochs, i, max_batch, batch_size * opt.log_interval / (time.time() - btic),
                        train_loss, train_metric_name, train_metric_score, trainer.learning_rate))
                    btic = time.time()


            #train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            name, val_acc = test(ctx, val_data)
            train_history.update([1-acc, 1-val_acc])
            train_history.plot(save_path='%s/%s_history.png'%(plot_path, model_name))

            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters('%s/%.4f-ITW-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))
                trainer.save_states('%s/%.4f-ITW-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

            logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
                (epoch, acc, val_acc, train_loss, time.time()-tic))

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                net.save_parameters('%s/ITW-%s-%d.params'%(save_dir, model_name, epoch))
                trainer.save_states('%s/ITW-%s-%d.params'%(save_dir, model_name, epoch))

        if save_period and save_dir:
            net.save_parameters('%s/ITW-%s-%d.params'%(save_dir, model_name, epochs-1))
            trainer.save_states('%s/ITW-%s-%d.params'%(save_dir, model_name, epochs-1))

    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()