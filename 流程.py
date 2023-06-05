from fast.fast_layers import max_pool_forward_fast, conv_forward_fast
from layers import relu_forward


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):#卷积层relu,池化层的结合

  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache