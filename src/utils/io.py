# try:
#     import dill as pickle
# except ImportError:
#     try:
#         import cPickle as pickle
#     except ImportError:
#         import pickle

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import gc
import json
import csv

import dataset as dt

from .. import config


def load_pickle(path):
    with open(path, 'rb') as f:
        pckl = pickle.load(f)
    return pckl


def save_pickle(path, obj):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created {dirname}")
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Pickle {path} saved.')
    return None


def write_hyperparams(log_dir, params, mode='FILE'):
    if mode == 'FILE' or mode == 'BOTH':
        os.makedirs(log_dir)
        hyperparam_file = os.path.join(log_dir, 'hyperparams.json')
        with open(hyperparam_file, 'w') as f:
            f.write(json.dumps(params))
    if mode == 'DATABASE' or mode == 'BOTH':
        db = dt.connect(config.DATABASE_CONNECTION_STRING)
        runs_table = db['runs']
        runs_table.insert(params)
    if mode not in ('FILE', 'DATABASE', 'BOTH'):
        raise ValueError('{} mode not recognized. Try with FILE, DATABASE or '
                         'BOTH'.format(mode))


def update_in_db(datadict):
    db = dt.connect(config.DATABASE_CONNECTION_STRING)
    runs_table = db['runs']
    runs_table.update(datadict, keys=['hash'])


def write_metrics(log_dir, params):
    """This code asumes the log_dir directory already exists"""
    metrics_file = os.path.join(log_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        f.write(json.dumps(params))


def write_output(log_dir, output, best_or_last, mode):
    dev_output_file_path = os.path.join(log_dir,
                                        best_or_last +
                                        '_' + mode + '.output.json')

    with open(dev_output_file_path, 'w') as f:
        f.write(json.dumps(output))


def write_probs(filepath, sentences_ids, probs):
    """probs: a list of torch.cuda.FloatTensor"""
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        for sent_id, prob_tensor in zip(sentences_ids, probs):
            line = [sent_id] + ['{:.12f}'.format(elem) for elem
                                in prob_tensor.tolist()]
            writer.writerow(line)


def _repr2str(sent_id, sent_repr, p_or_h):
    formatted_prem_repr = ['{:.12f}'.format(elem) for elem
                           in sent_repr.tolist()]
    prem_list_string = ' '.join(formatted_prem_repr)
    line = '\t'.join([sent_id, p_or_h])
    line = '\t'.join([line, prem_list_string])
    return line


def write_sent_reprs(filepath, sentences_ids, prem_reprs, hypo_reprs):
    """[]_reprs are lists of torch.cuda.FloatTensor"""
    with open(filepath, 'w') as f:
        for idx, sent_id in enumerate(sentences_ids):
            prem_repr = prem_reprs[idx]
            hypo_repr = hypo_reprs[idx]
            line_p = _repr2str(sent_id, prem_repr, 'p')
            line_h = _repr2str(sent_id, hypo_repr, 'h')
            f.write(line_p + '\n')
            f.write(line_h + '\n')


def get_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_hyperparams_from_model(model):
    """This assumes the hyperparams.json file is located in the same directory
    as the model"""
    dirname = os.path.dirname(model)
    with open(os.path.join(dirname, 'hyperparams.json'), 'r') as f:
        hyperparams = json.loads(f.read())
    return hyperparams


def get_hash_from_model(model):
    """This assumes the hyperparams.json file is located in the same directory
    as the model"""
    hyperparams = get_hyperparams_from_model(model)
    model_hash = hyperparams['hash']
    return model_hash


def get_datetime_from_model(model):
    """This assumes the hyperparams.json file is located in the same directory
    as the model"""
    hyperparams = get_hyperparams_from_model(model)
    run_datetime = hyperparams['datetime']
    run_datetime = run_datetime.replace('-', '_')
    run_datetime = run_datetime.replace(' ', '_').replace(':', '_')
    return run_datetime


def write_output_details(savepath, corpus, pred_label_ids,
                         best_or_last, mode):
    """TODO: Rewrite this to make it less data dependent"""
    idx2label = {idx: label for label, idx in corpus.label_dict.items()}
    idx2word = corpus.lang.index2word
    result = []
    for i, (id_tuple, pred_label_id) in enumerate(zip(corpus.id_tuples,
                                                      pred_label_ids)):
        prem_ids = id_tuple[0]
        hypo_ids = id_tuple[1]
        gold_label_id = id_tuple[2]
        pair_id = id_tuple[3]
        premise_words = map(idx2word.__getitem__, prem_ids)
        hypothesis_words = map(idx2word.__getitem__, hypo_ids)
        gold_label = idx2label[gold_label_id]
        pred_label = idx2label[pred_label_id]

        # Assumes the corpus is NOT shuffled
        genre = corpus.raw_examples[i].genre
        result.append((pair_id, premise_words, hypothesis_words,
                       gold_label, pred_label, genre))

    output = {"fields": ['pair_id', 'premise', 'hypothesis',
                         'reference_label', 'predicted_label', 'genre'],
              "result": result}

    write_output(savepath, output, best_or_last, mode)


def load_or_create(path, function_or_object, *args, **kwargs):
    """
    Either load pickle from path or create desired object through the creation
    function and its args
    path: path of the potential pickle
    function_or_object: function used to create the object we want or the
                        object itself
    creation_args: args to be passed to the creation_fn only used if
                   function_or_object is a function
    force_reload: whether to force pickles being created again
    """
    force_reload = kwargs.pop('force_reload')
    if os.path.exists(path) and not force_reload:
        print('Loading', path)
        # Disable garbage collector; makes loading the pickle much faster
        # I don't know whether this has side effects
        gc.disable()
        loaded_or_created = load_pickle(path)
        gc.enable()
    else:

        if not os.path.exists(path):
            print(path, 'not found. Creating pickle.')
        else:
            print(path, 'found but force_reload flag was passed. Overwriting...')

        if callable(function_or_object):
            loaded_or_created = function_or_object(*args, **kwargs)
        else:
            loaded_or_created = function_or_object

        save_pickle(path, loaded_or_created)
    return loaded_or_created


def read_jsonl(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    proc_lines = []
    for line in lines:
        proc_lines.append(json.loads(line))
    return proc_lines
