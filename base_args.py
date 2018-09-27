import argparse

from src.utils.logger import Logger


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(
            formatter_class=lambda prog: CustomFormatter(prog,
                                                         max_help_position=40),
            *args, **kwargs)


base_parser = argparse.ArgumentParser(description='', add_help=False)


data_args = base_parser.add_argument_group('Data proportions',
                                           'Data proportions to use when '
                                           'training and validating')

data_args.add_argument('-tdp', '--train_data_proportion', type=float,
                       default=1.0, metavar='',
                       help='Proportion of the training data to use.')

data_args.add_argument('-ddp', '--dev_data_proportion', type=float,
                       default=1.0, metavar='',
                       help='Proportion of the validation data to use.')

training_args = base_parser.add_argument_group('Training Hyperparameters',
                                               'Hyperparameters specific to '
                                               'the training procedure, and '
                                               'unrelated to the NN '
                                               'architecture')

training_args.add_argument('--epochs', type=int, default=10, metavar='',
                           help='Number of epochs for training')

training_args.add_argument('--batch_size', type=int, default=64, metavar='',
                           help='Size of the minibatch to use for training and'
                                'validation')

optim_args = base_parser.add_argument_group('Optimizer parameters')

optim_args.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        metavar='',
                        help='Initial learning rate')

optim_choices = ['sgd', 'adagrad', 'adadelta', 'adam', 'rmsprop']
optim_args.add_argument('--optim', type=str, default='adam',
                        choices=optim_choices,
                        help='Optimizer to use')

optim_args.add_argument('-gc', '--grad_clipping', type=float, default=5.0,
                        metavar='',
                        help='Gradients are clipped to this value each '
                             'time step is called by the optimizer')

optim_args.add_argument('-lrd', '--learning_rate_decay', type=float,
                        default=1.0, metavar='',
                        help='Factor by which current learning rate is '
                             'multiplied each time it is decayed')

optim_args.add_argument('-sda', '--start_decay_at', type=int,
                        default=2, metavar='',
                        help='Epoch at which the learning rate decay should '
                             'begin')

optim_args.add_argument('-de', '--decay_every', type=int, default=2,
                        metavar='',
                        help='Frequency in epochs by which the learning rate '
                             'should be decayed')

misc_args = base_parser.add_argument_group('Miscellaneous')

misc_args.add_argument('--write_mode', type=str,
                       choices=Logger.WRITE_MODES, default='NONE',
                       help='Mode for saving hyperparameters and results')

misc_args.add_argument('--save_model', action='store_true',
                       help='Whether to save the best and last model '
                            'checkpoints')

misc_args.add_argument('--no_cuda', action='store_true',
                       help='Force the use of the cpu even if a gpu is '
                            'available')

misc_args.add_argument('--log_interval', type=int, default=200, metavar='',
                       help='Number of iterations between training loss '
                            'loggings')

misc_args.add_argument('--seed', type=int, default=42, metavar='',
                       help='Random seed to be used by torch initializations')
