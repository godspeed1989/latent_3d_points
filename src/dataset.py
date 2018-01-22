import os
import os.path as osp
import re
import warnings
import numpy as np

from multiprocessing import Pool
from ply_file_internal import PlyData, PlyElement

blue = lambda x:'\033[94m' + x + '\033[0m'

def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name

'''
    多线程载入数据
'''
def load_point_clouds_from_filenames(file_names, n_threads, loader):
    # 载入一个获得文件大小信息
    pc = loader(file_names[0])[0]
    print(blue('loading %d [%d x %d] pc...' % (len(file_names), pc.shape[0], pc.shape[1])))
    pclouds = np.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=np.float)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    
    pool = Pool(n_threads)
    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data
    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))
    return pclouds, model_names, class_ids

def pc_loader(f_name):
    tokens = f_name.split('/')
    model_name = tokens[-1].split('.')[0]
    class_id = tokens[-2]
    # read XYZ point cloud from PLY file
    plydata = PlyData.read(f_name)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array, model_name, class_id

'''
    文件夹中载入数据集
'''
def load_all_point_clouds_under_folder(top_dir, n_threads=5, file_ending='.ply'):
    file_names = [f for f in files_in_subdirs(top_dir, file_ending)]
    pclouds, model_names, class_ids = load_point_clouds_from_filenames(file_names, n_threads, loader=pc_loader)
    # label仅仅是 文件名 和 目录名 的组合
    return PointCloudDataSet(pclouds, labels=class_ids + '_' + model_names)

class PointCloudDataSet():
    def __init__(self, point_clouds, labels=None, copy=True, init_shuffle=True):
        self.num_examples = point_clouds.shape[0]
        self.n_points = point_clouds.shape[1]

        if labels is not None:
            assert point_clouds.shape[0] == labels.shape[0], ('points.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            if copy:
                self.labels = labels.copy()
            else:
                self.labels = labels
        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)

        if copy:
            self.point_clouds = point_clouds.copy()
        else:
            self.point_clouds = point_clouds

        self._index_in_epoch = 0
        if init_shuffle:
            self.shuffle_data()

    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.point_clouds = self.point_clouds[perm]
        self.labels = self.labels[perm]
        return self

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        assert batch_size < self.num_examples
        start = self._index_in_epoch
        end = start + batch_size
        # Finished one epoch
        if end > self.num_examples:
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            end = batch_size
        self._index_in_epoch = end

        return self.point_clouds[start:end], self.labels[start:end]

if __name__ == '__main__':
    ds = load_all_point_clouds_under_folder('../../shape_net_core_uniform_samples_2048')
    print(ds.num_examples)
