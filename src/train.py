import os.path as osp

from enc_dec import mlp_architecture_ala_iclr_18
from pnt_ae import PointNetAutoEncoder, Configuration
from dataset import load_all_point_clouds_under_folder

blue = lambda x:'\033[94m' + x + '\033[0m'

'''
    主函数
'''
if __name__ == '__main__':
    bneck = 128
    ae_loss = 'emd'
    n_pc_points = 2048

    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck)    
    conf = Configuration (
                n_input = [n_pc_points, 3],
                loss = ae_loss,
                training_epochs = 100,
                batch_size = 50,
                learning_rate = 0.0005,
                train_dir = './train_dir-1252',
                training = True,
                # load existing model
                model_path = 'train_dir-1151/models.ckpt-1252',
                # coders
                encoder = encoder,
                decoder = decoder,
                encoder_args = enc_args,
                decoder_args = dec_args
            )
    experiment_id = '_'.join(['ae', str(n_pc_points), 'pts', str(bneck), 'bneck', ae_loss])
    conf.experiment_name = 'experiment_' + str(experiment_id)
    print(conf)

    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', 1)

    print(blue('load dataset'))
    dataset = load_all_point_clouds_under_folder('../shape_net_core_uniform_samples_2048')
    print(blue('start to train'))
    ae.train(train_data=dataset, log_file=fout)

    fout.close()

