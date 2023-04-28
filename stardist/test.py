import torch
import tensorflow as tf
from stardist import gputools_available

print('Number of GPUs: %d' % len(tf.config.list_physical_devices('GPU')))

print(tf.config.list_physical_devices(
device_type=None
))
print(torch.cuda.is_available())
print(gputools_available())