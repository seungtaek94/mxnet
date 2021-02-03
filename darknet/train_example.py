import argparse
from cspdarknet53 import DarkNet
import mxnet as mx
from trainer import Trainer

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