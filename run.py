import os

from glob import glob

import torch
import colored_traceback
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.corpus.corpus import ClassificationCorpus

from src.utils.logger import Logger
from src.utils.ops import np_softmax

from src.train import Trainer
from src.optim.optim import OptimWithDecay
from src import config

from src.models.classifier import Classifier

from src.layers.pooling import PoolingLayer

from base_args import base_parser, CustomArgumentParser

colored_traceback.add_hook(always=True)


base_parser.description = ''

arg_parser = CustomArgumentParser(parents=[base_parser],
                                  description='Deep Contextualized Word Representations for detecting Irony and Sarcasm')

arg_parser.add_argument('--corpus', type=str, required=True,
                        choices=list(config.corpora_dict.keys()),
                        help='Name of the corpus to use.')

arg_parser.add_argument('--lstm_hidden_size', type=int, default=2048,
                        help='Hidden dimension size for the word-level LSTM')

arg_parser.add_argument('--sent_enc_layers', type=int, default=1,
                        help='Number of layers for the word-level LSTM')

arg_parser.add_argument('--force_reload', action='store_true',
                        help='Whether to reload pickles or not (makes the '
                        'process slower, but ensures data coherence)')

arg_parser.add_argument('--pooling_method', type=str, default='max',
                        choices=PoolingLayer.POOLING_METHODS,
                        help='Pooling scheme to use as raw sentence '
                             'representation method.')

arg_parser.add_argument('--sent_enc_dropout', type=float, default=0.0,
                        help='Dropout between sentence encoding lstm layers. '
                             'and after the sent enc lstm. 0 means no dropout.')

arg_parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout applied to layers and final MLP. 0 means no dropout.')

arg_parser.add_argument('--model_hash', type=str, default=None,
                        help='Hash of the model to load, can be a partial hash')

arg_parser.add_argument('--lowercase', '-lc', action='store_true',
                        help='Whether to lowercase data or not. WARNING: '
                             'REMEBER TO CLEAR THE CACHE BY PASSING '
                             '--force_reload or deleting .cache')

arg_parser.add_argument("--test", action="store_true",
                        help="Run this script in test mode")

arg_parser.add_argument('--training_schedule',
                        default='decay',
                        choices=['decay', 'decay-nie'],
                        help='Schedule to alter learning rate during training')


def main():

    hp = arg_parser.parse_args()


    logger = Logger(hp, model_name='elmo4irony', write_mode=hp.write_mode)
    if hp.write_mode != 'NONE':
        logger.write_hyperparams()

    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed_all(hp.seed)  # silently ignored if there are no GPUs

    CUDA = False
    if torch.cuda.is_available() and not hp.no_cuda:
        CUDA = True

    USE_POS = False

    corpus = ClassificationCorpus(config.corpora_dict, hp.corpus,
                                  force_reload=hp.force_reload,
                                  train_data_proportion=hp.train_data_proportion,
                                  dev_data_proportion=hp.dev_data_proportion,
                                  batch_size=hp.batch_size,
                                  lowercase=hp.lowercase,
                                  use_pos=USE_POS)

    if hp.model_hash:
        experiment_path = os.path.join(config.RESULTS_PATH, hp.model_hash + '*')
        ext_experiment_path = glob(experiment_path)
        assert len(ext_experiment_path) == 1, 'Try providing a longer model hash'
        ext_experiment_path = ext_experiment_path[0]
        model_path = os.path.join(ext_experiment_path, 'best_model.pth')
        model = torch.load(model_path)

    else:
        # Define some specific parameters for the model
        num_classes = len(corpus.label2id)
        batch_size = corpus.train_batches.batch_size
        hidden_sizes = hp.lstm_hidden_size
        model = Classifier(num_classes,
                           batch_size,
                           hidden_sizes=hidden_sizes,
                           use_cuda=CUDA,
                           pooling_method=hp.pooling_method,
                           batch_first=True,
                           dropout=hp.dropout,
                           sent_enc_dropout=hp.sent_enc_dropout,
                           sent_enc_layers=hp.sent_enc_layers)

    if CUDA:
        model.cuda()

    if hp.write_mode != 'NONE':
        logger.write_architecture(str(model))

    logger.write_current_run_details(str(model))

    optimizer = OptimWithDecay(model.parameters(),
                               method=hp.optim,
                               initial_lr=hp.learning_rate,
                               max_grad_norm=hp.grad_clipping,
                               lr_decay=hp.learning_rate_decay,
                               start_decay_at=hp.start_decay_at,
                               decay_every=hp.decay_every)


    loss_function = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_function, num_epochs=hp.epochs,
                      use_cuda=CUDA, log_interval=hp.log_interval)

    if hp.test:
        if hp.model_hash is None:
            raise RuntimeError(
                'You should have provided a pre-trained model hash with the '
                '--model_hash flag'
            )

        print(f'Testing model {model_path}')
        eval_dict = trainer.evaluate(corpus.test_batches)

        probs = np_softmax(eval_dict['output'])
        probs_filepath = os.path.join(ext_experiment_path,
                                      'test_probs.csv')
        np.savetxt(probs_filepath, probs,
                   delimiter=',', fmt='%.8f')
        print(f'Saved prediction probs in {probs_filepath}')

        labels_filepath = os.path.join(ext_experiment_path,
                                       'predictions.txt')
        labels = [label + '\n' for label in eval_dict['labels']]
        with open(labels_filepath, 'w', encoding='utf-8') as f:
            f.writelines(labels)
        print(f'Saved prediction file in {labels_filepath}')

        # We know the last segment of ext_experiment_path corresponds to the
        # model hash
        full_model_hash = os.path.basename(ext_experiment_path)

        # WARNING: We are accessing an internal method of the Logger class;
        # This could corrupt the results database if not used properly
        logger._update_in_db({'test_acc': eval_dict['accuracy'],
                              'test_f1': eval_dict['f1'],
                              'test_p': eval_dict['precision'],
                              'test_r': eval_dict['recall']},
                             experiment_hash=full_model_hash)

        exit()

    writer = None
    if hp.write_mode != 'NONE':
        writer = SummaryWriter(logger.run_savepath)
    try:
        best_accuracy = None
        for epoch in tqdm(range(hp.epochs), desc='Epoch'):
            total_loss = 0

            trainer.train_epoch(corpus.train_batches, epoch, writer)
            corpus.train_batches.shuffle_examples()
            eval_dict = trainer.evaluate(corpus.dev_batches, epoch, writer)

            if hp.training_schedule == 'decay':
                optim_updated, new_lr = trainer.optimizer.updt_lr_accuracy(epoch, eval_dict['accuracy'])
                lr_threshold = 1e-5
                if new_lr < lr_threshold:
                    tqdm.write(f'Learning rate smaller than {lr_threshold}, '
                               f'stopping.')
                    break
                if optim_updated:
                    tqdm.write(f'Learning rate decayed to {new_lr}')

            if hp.training_schedule == 'decay-nie':
                optim_updated, new_lr = trainer.optimizer.update_learning_rate_nie(epoch)
                if optim_updated:
                    tqdm.write(f'Learning rate decayed to {new_lr}')

            accuracy = eval_dict['accuracy']
            if not best_accuracy or accuracy > best_accuracy:
                best_accuracy = accuracy
                logger.update_results({'best_valid_acc': best_accuracy,
                                       'best_epoch': epoch})

                if hp.write_mode != 'NONE':
                    probs = np_softmax(eval_dict['output'])
                    probs_filepath = os.path.join(logger.run_savepath,
                                                  'best_eval_probs.csv')
                    np.savetxt(probs_filepath, probs,
                               delimiter=',', fmt='%.8f')

                    labels_filepath = os.path.join(logger.run_savepath,
                                                   'predictions_dev.txt')
                    labels = [label + '\n' for label in eval_dict['labels']]
                    with open(labels_filepath, 'w', encoding='utf-8') as f:
                        f.writelines(labels)

                if hp.save_model:
                    logger.torch_save_file('best_model_state_dict.pth',
                                           model.state_dict(),
                                           progress_bar=tqdm)
                    logger.torch_save_file('best_model.pth',
                                           model,
                                           progress_bar=tqdm)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
