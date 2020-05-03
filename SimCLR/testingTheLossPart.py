import torch
import torch.nn as nn
cosineSimilarity = nn.CosineSimilarity(dim=1)

torch.manual_seed(0)

# list=

vec1 = torch.rand((5,8))
vec2 = torch.rand((5, 8))

# for i in range(10):
#
#     vec = torch.rand((3,3))
#     print(vec)

print(vec1[0])
print(vec2[0])

# print(cosineSimilarity(vec1[0], vec2[0]))

print(cosineSimilarity(vec1[:], vec2[:]))

import tensorflow as tf
print(tf.__version__)

a = tf.one_hot(tf.range(10), 20)

print(tf.range(10))

print(a[0])

for i in a:
    print(i)
