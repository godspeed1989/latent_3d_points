import os.path as osp
import numpy as np
from ply_file_internal import PlyData, PlyElement

from enc_dec import mlp_architecture_ala_iclr_18
from pnt_ae import PointNetAutoEncoder, Configuration

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
                batch_size = 8,
                training = False,
                # load existing model
                model_path = 'train_dir-1252/models.ckpt-1261',
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

    def read_ply(filename):
        """ read XYZ point cloud from filename PLY file """
        plydata = PlyData.read(filename)
        pc = plydata['vertex'].data
        pc_array = np.array([[x, y, z] for x,y,z in pc])
        return pc_array

    def write_ply(points, filename, text=True):
        """ input: Nx3, write points to filename as PLY format. """
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=text).write(filename)

    filename = 'rot_printer_3d'
    pc_in = read_ply(filename + '.ply')

    loss, pc_out = ae._eval_one(pc_in)
    write_ply(pc_out, 'out_' + filename + '.ply')
    print(blue('loss %f' % loss))
