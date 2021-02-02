import os, time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import argparse, time, logging

import numpy as np
import mxnet as mx
import math

from mxnet import gluon, nd
from mxnet import autograd as ag
from cspdarknet53 import DarkNet

from gluoncv.utils import makedirs, TrainingHistory


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')

    # Training
    parser.add_argument('--batch-size', type=int, default=128, help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=0, help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='darknet',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-workers', dest='num_workers', default=16, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=150, help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='80,120',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--label-smoothing', type=bool, default=False,
                        help='use label smoothing or not in training. default is false.')

    parser.add_argument('--save-period', type=int, default=5, help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params', help='directory of saved models')
    parser.add_argument('--resume-from', type=str, help='resume training from the model')
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd, nag')

    # Resume
    parser.add_argument('--resume-epoch', type=int, default=0, help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='', help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='', help='path of trainer state to load from.')

    # Dataset
    parser.add_argument('--dataset-path', type=str, default='ITW_rec_origin',
                        help='root folder should be .mxnet/datasets/')
    parser.add_argument('--img-size', type=int, default=224, help='image size')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')

    # Logging
    parser.add_argument('--log-interval', type=int, default=10, help='Number of batches to wait before logging.')
    parser.add_argument('--save-plot-dir', type=str, default='.', help='the path to save the history plot')
    parser.add_argument('--logging-file', type=str, default='train_cspdarknet53.log', help='name of training log file')
    parser.add_argument('--model-name', type=str, default='cspdarknet53', help='model name for save exp')
    opt = parser.parse_args()
    return opt


class Trainer():
    def __init__(self, opt, net, ctx):
        super(Trainer, self).__init__()
        
        self.opt = opt
        self.net = net
        self.ctx = ctx
    
    def get_data_rec(self):
        dataset_path = self.opt.dataset_path
        batch_size = self.opt.batch_size
        img_size = self.opt.img_size
        num_workers = self.opt.num_workers
        rec_path = os.path.join('/home/seungtaek/.mxnet/datasets/', dataset_path)

        # Augmentation Params
        jitter_param = 0.4
        lighting_param = 0.1
        crop_ratio = self.opt.crop_ratio if self.opt.crop_ratio > 0 else 0.875
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
    
        
    def val(self):
        val_data.reset()
        metric = mx.metric.Accuracy()

        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)

        return metric.get()
        
    def run(self):
        now = time.localtime()
        now = f'{now.tm_year}{now.tm_mon:0>2}{now.tm_mday:0>2}{now.tm_hour:0>2}{now.tm_min:0>2}{now.tm_sec:0>2}'

        
        model_name = self.opt.model_name
        filehandler = logging.FileHandler(f'log/{model_name}_{now}.log')
        streamhandler = logging.StreamHandler()

        logger = logging.getLogger('')
        logger.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)
        logger.info(self.opt)

        lr_decay = self.opt.lr_decay
        lr_decay_epoch = [int(i) for i in self.opt.lr_decay_epoch.split(',')] + [np.inf]

        # model define
        net = self.net

        if self.opt.resume_from:
            net.load_parameters(opt.resume_from, ctx=ctx)
            
        optimizer = self.opt.optimizer

        save_period = self.opt.save_period
        if self.opt.save_dir and save_period:
            save_dir = self.opt.save_dir
            makedirs(save_dir)
        else:
            save_dir = ''
            save_period = 0

        plot_path = self.opt.save_plot_dir

        # load dataset
        train_data, val_data, batch_fn = self.get_data_rec()
        
        if isinstance(self.ctx, mx.Context):
            self.ctx = [self.ctx]

        if self.opt.resume_params == '':
            net.initialize(mx.init.Xavier(), ctx=ctx)
        else:
            net.load_parameters(self.opt.resume_params, ctx=ctx)

        trainer = gluon.Trainer(net.collect_params(), optimizer,
                                {'learning_rate': self.opt.lr, 'wd': self.opt.wd, 'momentum': self.opt.momentum})

        if self.opt.resume_states != '':
            trainer.load_states(self.opt.resume_states)

        # Sparse_label == True  : Label should be integer
        # Sparse_label == False : Label should contaion probability distribution
        if self.opt.label_smoothing:
            loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
        else:
            loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)

        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        history_acc = TrainingHistory(['train-acc', 'val-acc', 'train-loss'])

        lr_decay_count = 0
        iteration = 0
        best_val_score = 0
        max_batch = 0

        for epoch in range(self.opt.resume_epoch, self.opt.num_epochs):
            tic = time.time()
            train_data.reset()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            btic = time.time()
            estimate_time = 0.0

            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate * lr_decay)
                lr_decay_count += 1

            for i, batch in enumerate(train_data):
                data, label = batch_fn(batch, ctx)

                if self.opt.label_smoothing:
                    hard_label = label
                    label = smooth(label, classes)

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

                for l in loss:
                    l.backward()
                trainer.step(batch_size)

                if self.opt.label_smoothing:
                    train_metric.update(hard_label, output)
                else:
                    train_metric.update(label, output)

                iteration += 1
                train_loss = (train_loss + sum([l.sum().asscalar() for l in loss])) / iteration

                training_Speed = batch_size * self.opt.log_interval / (time.time() - btic)


                if self.opt.log_interval and not (i + 1) % self.opt.log_interval:
                    if i > max_batch: max_batch = i
                    train_metric_name, train_metric_score = train_metric.get()
                    logger.info(
                        'Epoch[%d/%d] Batch [%d/%d]\tSpeed: %f samples/sec\ttrain_loss=%f\t%s=%f\tlr=%f' % (
                            epoch, self.opt.num_epochs, i, max_batch, training_Speed, train_loss, train_metric_name,
                            train_metric_score, trainer.learning_rate))
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
                         (epoch, train_acc, val_acc, time.time() - tic))

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                net.save_parameters(f'params/{model_name}_{now}_{epoch:0>3}.params')
                trainer.save_states(f'params/{model_name}_{now}_{epoch:0>3}.states')

        if save_period and save_dir:
            net.save_parameters(f'params/{model_name}_{now}_{epoch - 1:0>3}.params')
            trainer.save_states(f'params/{model_name}_{now}_{epoch - 1:0>3}.states')

if __name__ == '__main__':
    opt = parse_args()
    
    batch_size = opt.batch_size
    classes = 5  # {sunny, cloudy, foggy, rain, snow}
    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    print(ctx)
    
    net = DarkNet(num_classes=5, input_size=opt.img_size)
    trainer = Trainer(opt, net, ctx)
    
    trainer.run()
