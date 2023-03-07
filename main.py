import argparse
import os

if __name__ == '__main__':
    path_root = 'E:\\PaddleHelix\\apps\\pretrained_compound\\ChemRL\\GEMOR\\'

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')
    parser.add_argument("--five_fold", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--dataset_name",
                        choices=['bbbp', 'rega', 'dilirank', 'diliset'], default='dilirank')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--cached_data_path", type=str, default=None)
    parser.add_argument("--split_type",
                        choices=['random', 'scaffold', 'random_scaffold', 'index'])

    parser.add_argument("--compound_encoder_config", type=str, default=[path_root + 'model_configs\\pretrain_gem.json',
                                                                        path_root + 'model_configs\\geognn_l8.json'])
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_params", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--encoder_lr", type=float, default=0.001)
    parser.add_argument("--head_lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--exp_id", type=int, help='used for identification only')
    parser.add_argument("--log", type=str)
    args = parser.parse_args()

    init_params = 'pretrain_models-chemrl_gem/class.pdparams'

    log_prefix = "log/pretrain-" + args.dataset_name

    cached_data_path = path_root + 'cached_data\\' + args.dataset_name
    data_path = path_root + 'chemrl_downstream_datasets\\' + args.dataset_name
    batch_size = 128
    model_config_list = ['down_mlp2.json', 'down_mlp3.json']

    lrs_list = [[5e-5, 5e-5], [1e-3, 1e-4], [1e-4, 1e-3]]
    drop_list = [0.2, 0.35, 0.4]
    log_dir = f'{log_prefix}/{args.dataset_name}'
    # os.mkdir('./'+log_dir)
    for model_config in model_config_list:

        if os.path.isfile(cached_data_path + '\\part-000000.npz') is False:
            # os.makedirs(cached_data_path)
            args.task = 'data'
            args.num_workers = 1
            args.dataset_name = args.dataset_name
            args.data_path = data_path
            args.cached_data_path = cached_data_path
            args.model_config = path_root + 'model_configs\\down_mlp2.json'
            main(args)
            # touch $cached_data_path.done
            break

        for lrs in lrs_list:
            lr = lrs[0]
            head_lr = lrs[1]
            for dropout_rate in drop_list:
                log_file = f'{log_dir}/lr{lr}_{head_lr}/drop{dropout_rate}.txt'
                # if os.path.exists(f'{log_dir}/lr{lr}_{head_lr}') is False:
                #     os.makedirs(f'{log_dir}/lr{lr}_{head_lr}')
                # else: os.remove(log_file)
                print(f'Outputs redirected to {log_file}')
                for time in range(1, 100):
                    args.task = 'train'
                    args.num_workers = 1
                    args.batch_size = batch_size
                    args.max_epoch = 200
                    args.dataset_name = args.dataset_name
                    args.data_path = data_path
                    args.cached_data_path = cached_data_path
                    args.split_type = 'random'
                    args.model_config = path_root + 'model_configs\\' + model_config
                    args.init_params = init_params
                    # args.init_model = None
                    args.model_dir = path_root + 'finetune_models\\' + args.dataset_name
                    args.encoder_lr = lr
                    args.head_lr = head_lr
                    args.dropout_rate = dropout_rate
                    args.log = log_file
                    if args.five_fold == True:
                        from five_hold_class import main

                        main(args)
                    else:
                        from finetune_class import main

                        main(args)

                    # ana_results.main(args)
# 						cat model_dir/* | grep FINAL| 只显示文件中每行FINAL之后的内容
# 						python ana_results.py > $log_dir/final_result
