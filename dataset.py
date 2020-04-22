import os
import numpy as np
from common import cf


class Dataset(object):
    def __init__(self, dataset_name):
        dataset_path = cf.data_path_root
        self.dataset_meta = { 'train': os.path.join(dataset_path, 'train_array.npy'),
                              'test': os.path.join(dataset_path, 'test_array.npy')}
        self.files = self.dataset_meta[dataset_name]
        
    def load(self):
        '''Load dataset metas from files'''
        dataset = np.load(self.files)
        self.instances = dataset.shape[0]
        try:                
            shuffle_indices = np.random.choice(np.arange(self.instances),size = self.instances,replace = False)
            shuffle_data = dataset[shuffle_indices,:]
        except IndexError:
            shuffle_indices = np.random.choice(np.arange(self.instances - 1),size = self.instances-1,replace = False)
            shuffle_data = dataset[shuffle_indices,:]
        self.samples = {'X': shuffle_data}
        return self
    
    def batch_iter(self):
        dataset = self.samples['X']
        num_batches_per_epoch = self.instances // cf.minibatch_size
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * cf.minibatch_size
            end_index = min((batch_num + 1) * cf.minibatch_size, self.instances)
            shuffle_data1 = dataset[start_index:end_index,1:-1]
            shuffle_len = dataset[start_index:end_index,-1:]
            shuffle_label = dataset[start_index:end_index,0]
            yield (shuffle_data1,shuffle_label,shuffle_len)
    @property
    def instances_per_epoch(self):
        return self.instances
    
    @property
    def minibatchs_per_epoch(self):
        return self.instances // cf.minibatch_size 
