import cPickle as pkl
import numpy as np

#########################################
### Data Statistics for ImageNet-1000 ###
#########################################
IMAGE_HEIGHT = 224   
IMAGE_WIDTH = 224
num_classes = 1000 
num_data_samples = num_classes*1024
num_valid_samples = num_classes*128

data_dir = "./ImageNet_data/"
save_dir = "./models/model"

dataset_split_name = "train"
fc_conv_padding='VALID'
fc_conv_filter_size = 7
mask = pkl.load(open('./mask_ratio_random.save','rb'))

# TRAIN
batch_size = 50
num_threads = 5
learning_rate = 0.0001

num_epochs = 1
num_batches = int(num_data_samples/batch_size)

# OPTIMIZER
rmsprop_decay = 0.9
momentum = 0.9
opt_epsilon = 1.0
learning_rate_decay_type = 'exponential'
num_epochs_per_decay = 2.0
learning_rate_decay_factor =  0.94

# Pruning 
list_filters = [ 64,   64,   128,   128,  256,  256,  256,  512,  512,  512,  512,  512, 512]
filter_sizes_after_pruning = [ int(np.sum(m)) for m in mask]
print("No of Filters before pruning:", list_filters)
print("No of Filters after pruning:",  filter_sizes_after_pruning)
print("Prune Factor in each layer:",  (1-np.divide(np.array(filter_sizes_after_pruning)*1.0,np.array(list_filters))).round(2))
# prune_factor = [ 0.08, 0.08, 0.16,  0.16, 0.32, 0.32, 0.32, 0.64, 0.64, 0.64, 0.64,  0,   0]

# UTIL FUNCTIONS:
def get_data_files(dataset_split_name):
    tfrecords_filename = []
    if dataset_split_name == 'train':
        length = 1024
        data = "-of-1024"
    if dataset_split_name == 'validation':
        length = 128
        data = "-of-0128"
    
    for k in range(length): # Train data tfrecords
        j = "-%04d"%k        # pad with 0's
        tfrecords_filename.append(data_dir+dataset_split_name+j+data)
    return tfrecords_filename

