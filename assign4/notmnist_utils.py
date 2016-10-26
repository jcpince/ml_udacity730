from six.moves import cPickle as pickle
from six.moves import range
import numpy as np

class notmnist:
  def __init__(self, filename, do_import = True):
    self._pickle_file = filename
    if do_import:
      self.import_data()
  
  def import_data(self):
    self.get_datasets()

  def get_datasets(self):
    f = open(self._pickle_file, 'rb')
    save = pickle.load(f)
    self.train_dataset = save['train_dataset']
    self.train_labels = save['train_labels']
    self.valid_dataset = save['valid_dataset']
    self.valid_labels = save['valid_labels']
    self.test_dataset = save['test_dataset']
    self.test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    self.image_size = self.train_dataset.shape[1]
    self.num_channels = 1
    self.num_labels = 10
    self.train_offset = 0
    self.train_size = self.train_dataset.shape[0]
    self.valid_offset = 0
    self.valid_size = self.valid_dataset.shape[0]
    self.test_offset = 0
    self.test_size = self.test_dataset.shape[0]
  
  def print_imports(self):
    print('Training set and labels:', self.train_dataset.shape,
        self.train_labels.shape)
    print('Validation set and labels:', self.valid_dataset.shape,
        self.valid_labels.shape)
    print('Test set and labels:', self.test_dataset.shape,
        self.test_labels.shape)
  
  def reshape_4d(self):
    self.train_dataset = self.train_dataset.reshape(
            (-1, self.image_size, self.image_size,
                self.num_channels)).astype(np.float32)
    self.test_dataset = self.test_dataset.reshape(
            (-1, self.image_size, self.image_size,
                self.num_channels)).astype(np.float32)
    self.valid_dataset = self.valid_dataset.reshape(
            (-1, self.image_size, self.image_size,
                self.num_channels)).astype(np.float32)
  
  def encode_onehot(self):
    self.train_labels = (np.arange(self.num_labels) ==
            self.train_labels[:,None]).astype(np.float32)
    self.test_labels = (np.arange(self.num_labels) ==
            self.test_labels[:,None]).astype(np.float32)
    self.valid_labels = (np.arange(self.num_labels) ==
            self.valid_labels[:,None]).astype(np.float32)
  
  def get_ds(self, ds_name):
    if ds_name == 'train':
      return self.train_dataset
    elif ds_name == 'valid':
      return self.valid_dataset
    elif ds_name == 'test':
      return self.test_dataset
  
  def get_labels(self, ds_name):
    if ds_name == 'train':
      return self.train_labels
    elif ds_name == 'valid':
      return self.valid_labels
    elif ds_name == 'test':
      return self.test_labels
  
  def next_batches(self, ds_name, batch_size):
    if ds_name == 'train':
      start = self.train_offset
      end = self.train_offset + batch_size
      self.train_offset = end % (self.train_size - batch_size)
    elif ds_name == 'valid':
      start = self.valid_offset
      end = self.valid_offset + batch_size
      self.valid_offset = end % (self.valid_size - batch_size)
    elif ds_name == 'test':
      start = self.test_offset
      end = self.test_offset + batch_size
      self.test_offset = end % (self.test_size - batch_size)
    batch_data = self.get_ds(ds_name)[start:end]
    batch_labels = self.get_labels(ds_name)[start:end]
    return batch_data, batch_labels
