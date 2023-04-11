import os

from matplotlib import pyplot as plt
from train import training, evaluate

from utils import get_pos_neg_ratio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from os.path import join, exists, basename
import argparse
import numpy as np
import paddle
import tqdm
import paddle.nn as nn
import datetime
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.datasets.inmemory_dataset import InMemoryDataset
import prettytable
# from gtrick import FLAG
from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from src.utils import get_dataset, create_splitter, get_downstream_task_names, \
    calc_rocauc_score, exempt_parameters

print(paddle.device.get_device())


def build_model(encoder_lr, head_lr, init_model, configs,args):
    ### build model

    compound_encoder = GeoGNNModel(configs[0])
    model = DownstreamModel(configs[1], compound_encoder)
    criterion = nn.BCELoss(reduction='mean')
    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)

    collate_fn = DownstreamCollateFn(
        atom_names=configs[0]['atom_names'],
        bond_names=configs[0]['bond_names'],
        bond_float_names=configs[0]['bond_float_names'],
        bond_angle_float_names=configs[0]['bond_angle_float_names'],
        task_type='class')

    encoder_opt = paddle.optimizer.AdamW(encoder_lr, beta1=0.9, beta2=0.999, weight_decay=0.01,
                                         parameters=encoder_params)
    head_opt = paddle.optimizer.AdamW(head_lr, beta1=0.9, beta2=0.999, weight_decay=0.01, parameters=head_params)
    print('Total param num: %s' % (sum(p.numel().numpy()[0] for p in model.parameters())))
    print('Encoder param num: %s' % (sum(p.numel().numpy()[0] for p in compound_encoder.parameters())))
    print('Head param num: %s' % (sum(p.numel().numpy()[0] for p in head_params)))
    for i, param in enumerate(model.named_parameters()):
        print(i, param[0], param[1].name)
    # 加载原始GEM参数
    #
    if not init_model is None and not init_model == "":
        compound_encoder.set_state_dict(paddle.load(init_model))
        print('Load state_dict from %s' % init_model)
    # 加载已训练好的参数
    if args.task == 'test':
        model.set_state_dict(paddle.load('pretrain_models-chemrl_gem/dilirank_model.pdparams'))

    return model, encoder_opt, head_opt, criterion, collate_fn


def main(args):
    compound_encoder_config = load_json_config(args.compound_encoder_config[1])
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate
    task_names = get_downstream_task_names(args.dataset_name, args.data_path)
    model_config['task_type'] = 'class'
    model_config['num_tasks'] = len(task_names)
    master = 0
    dataset_list = data_process(task_names, args)

    for train_dataset, test_dataset in dataset_list:
        list_test, list_train, list_auc = [], [], []
        ps_tool = Plot_save(args, train_dataset, test_dataset)

        model, encoder_opt, head_opt, criterion, collate_fn = build_model(args.encoder_lr, args.head_lr,
                                                                          args.init_params,
                                                                          [compound_encoder_config, model_config], args)
        for epoch_id in tqdm.trange(args.max_epoch):
            train_loss, train_auc, train_table, train_gra, train_label = training(args, model, train_dataset,
                                                                                  collate_fn,
                                                                                  criterion, encoder_opt, head_opt)
            test_loss, test_auc, test_table, test_gra, test_label, test_pred, test_mcc = evaluate(args, model,
                                                                                                  test_dataset,
                                                                                                  collate_fn)

            if test_mcc >= 0.8:
                # master = test_mcc / 1.
                print(test_mcc)
                ps_tool.save_split_data()
                ps_tool.save_model(model, test_mcc)
                ps_tool.save_data([test_table, (train_gra, test_gra), (train_label, test_label)])

            list_train.append(train_table._rows[0])
            list_test.append(test_table._rows[0])
            list_auc.append([train_auc, test_auc])

            train_table.add_row(test_table._rows[0])

            fieldname = 'Type'
            train_table._field_names.insert(0, fieldname)
            train_table._align[fieldname] = 'c'
            train_table._valign[fieldname] = 't'
            for i, a in enumerate(['train', 'test']):
                train_table._rows[i].insert(0, a)
            print(train_table)

        list_train = np.array(list_train)[:, 1:].astype('float64')
        list_test = np.array(list_test)

        best = prettytable.PrettyTable()
        best.field_names = ['Result', 'Loss', 'AUC', 'Accuracy', 'Precison', 'Recall', 'F1_score', 'Specificity', 'MCC']
        best.add_row(['best_train', list_train[:, 0].min()] + list(list_train[:, 1:].max(0)))
        best.add_row(['best_test', list_test[:, 0].min()] + list(list_test[:, 1:].max(0)))
        print(best)
        ps_tool.plot_auc(list_auc)


def data_process(task_names, args):
    if args.task == 'data':
        print('Preprocessing data...')
        dataset = get_dataset(args.dataset_name, args.data_path, task_names)
        dataset.transform(DownstreamTransformFn(), num_workers=args.num_workers)
        dataset.save_data(args.cached_data_path)
        return
    else:
        if args.cached_data_path is None or args.cached_data_path == "":
            print('Processing data...')
            dataset = get_dataset(args.dataset_name, args.data_path, task_names)
            dataset.transform(DownstreamTransformFn(), num_workers=args.num_workers)
        else:
            print('Read preprocessing data...')
            dataset = InMemoryDataset(npz_data_path=args.cached_data_path)
    train_dataset_ = []
    test_dataset = []
    dataset_list = []
    if os.path.isfile(f'data_pro/random/{args.dataset_name}_smiles_train.npy') is True:
        train_s = np.load(f'data_pro/random/{args.dataset_name}_smiles_train.npy')

        data = dataset.data_list
        for i in range(data.__len__()):
            if data[i]['smiles'] in train_s:
                train_dataset_.append(data[i])
            else:
                test_dataset.append(data[i])

        test_dataset = InMemoryDataset(data_list=test_dataset)
        train_dataset = InMemoryDataset(data_list=train_dataset_)
        dataset_list.append([train_dataset, test_dataset])
    else:
        splitter = create_splitter(args.split_type)
        train_dataset, valid_dataset, test_dataset = splitter.split(
            dataset, frac_train=0.8, frac_valid=0, frac_test=0.2)
        dataset_list.append([train_dataset, test_dataset])
    return dataset_list


class Plot_save:
    def __init__(self, args, train_dataset, test_dataset):
        self.args = args
        self.dataset = [train_dataset, test_dataset]

    def save_model(self, model, mcc):
        print('Saving!')
        paddle.save(model.compound_encoder.state_dict(),
                    f'finetune_models/{self.args.dataset_name}_encoder_{mcc}.pdparams')
        paddle.save(model.state_dict(), f'finetune_models/{self.args.dataset_name}_model_{mcc}.pdparams')

    def save_data(self, data):
        np.save(f'data_pro/random/{self.args.dataset_name}_graph_repr.npy', np.concatenate(data[1], 0))
        np.save(f'data_pro/random/{self.args.dataset_name}_graph_label.npy', np.concatenate(data[2], 0))
        with open('test.txt', 'a') as f:
            f.write(
                f'down_lr={self.args.head_lr}, GEM_lr={self.args.encoder_lr}, dropout={self.args.dropout_rate}, batch_size={self.args.batch_size}, split_type={self.args.split_type}' + '\n')
            f.write(datetime.datetime.now().strftime("%d_%H_%M_%S") + '\n')
            f.write(data[0].get_string() + '\n')

    def plot_auc(self, list_auc):
        np.save(f'finetune_models/finetune_auc', list_auc)
        # ‘'g‘'代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
        plt.plot(range(list_auc.__len__()), list_auc, label=['train_auc', 'test_auc'])
        plt.legend()
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.show()

    def save_split_data(self):
        smiles_train = []
        label_train = []
        for i in range(self.dataset[0].data_list.__len__()):
            smiles_train.append(self.dataset[0].data_list[i]['smiles'])
            label_train.append(self.dataset[0].data_list[i]['label'])

        smiles_test = []
        label_test = []
        for i in range(self.dataset[1].data_list.__len__()):
            smiles_test.append(self.dataset[1].data_list[i]['smiles'])
            label_test.append(self.dataset[1].data_list[i]['label'])

        print("Train/Test num: %s/%s" % (
            len(self.dataset[0]), len(self.dataset[1])))
        print('Train pos/neg ratio %s/%s' % get_pos_neg_ratio(self.dataset[0]))
        print('Test pos/neg ratio %s/%s' % get_pos_neg_ratio(self.dataset[1]))
        if os.path.isdir(f'{self.args.split_type}_split') is False:
            os.mkdir(f'{self.args.split_type}_split')
        dir_path = f'{self.args.split_type}_split'
        np.save(
            f'{dir_path}/{self.args.dataset_name}_finetune_smiles_train.npy',
            smiles_train)
        np.save(
            f'{dir_path}/{self.args.dataset_name}_finetune_smiles_test.npy',
            smiles_test)
        np.save(
            f'{dir_path}/{self.args.dataset_name}_finetune_label_train.npy',
            label_train)
        np.save(
            f'{dir_path}/{self.args.dataset_name}_finetune_label_test.npy',
            label_test)
