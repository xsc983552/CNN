from layer_utils import *
import numpy as np
class ThreeLayerConvNet(object):    
    """    
    A three-layer convolutional network with the following architecture:       
       conv - relu - 2x2 max pool - affine - relu - affine - softmax
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,             
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        初始化网络

        输入:
        - input_dim: 数据以元组（C, H, W) 的方式给入
        - num_filters: 在卷积层中使用的卷积核数量
        - filter_size: 卷积核的宽高
        - hidden_dim: 在全连接的隐藏层中使用的单位数量
        - num_classes: 从final affine layer产生的scores的数量
        - weight_scale: 给定加权随机初始化标准差的标量
        - reg:  L2 正则化系数
        - dtype: 用于计算的数据类型
        """

        self.params = {}
        self.reg = reg
        self.dtype = dtype

        pad = 1 # 卷积时在图外围补0
        stride = 2 # 卷积核步长
        filter_size = pad*2 + 1 # 卷积核的大小 宽高都为3

        # W1和b1存储卷积层的权重和偏差
        W1 = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        b1 = np.zeros(num_filters)

        out_h = (input_dim[1] - filter_size + 2*pad) // stride + 1
        out_w = (input_dim[2] - filter_size + 2*pad) // stride + 1

        # W2和b2存储hidden affine层的权重和偏差
        W2 = weight_scale * np.random.randn(num_filters*out_h*out_w, hidden_dim)
        b2 = np.zeros(hidden_dim)

        # W3和b3用于输出affine的权重和偏差
        W3 = weight_scale * np.random.randn(hidden_dim, num_classes)
        b3 = np.zeros(num_classes)

        self.params["W1"], self.params["b1"] = W1, b1
        self.params["W2"], self.params["b2"] = W2, b2
        self.params["W3"], self.params["b3"] = W3, b3

        # 下面这个循环，k就是键，v就是键值，把v的数据类型改了
        for k, v in self.params.items():#3.7的Python 版本应该是没有iteritems这个函数了，改为items
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):



        W1, b1 = self.params['W1'], self.params['b1']#W1的shape(32, 3, 7, 7) b1(32,)
        W2, b2 = self.params['W2'], self.params['b2']#W2的shape(8192, 100) b2(100)
        W3, b3 = self.params['W3'], self.params['b3']#W3的shape(100, 10) b3(10)
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]# =7
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}#卷积层参数

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}#池化层参数
        scores = None

        '''
        实现三层卷积网络forward，计算X的类的scores
        '''
        conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param) # 进行卷积层，relu，再池化后的输出
        conv_relu_pool_out_flat = conv_relu_pool_out.reshape(conv_relu_pool_out.shape[0], -1)
        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_relu_pool_out_flat, W2, b2) # 用前面得到的数据再进行affine和relu
        affine_out, affine_cache = affine_forward(affine_relu_out, W3, b3) # 最后进行affine，得到的结果是scores
        scores = affine_out

        if y is None:    #y的默认值是None，就是给参数没有输入y的情况下
            return scores

        loss, grads = 0, {}


        reg = self.reg
        loss, dscore = softmax_loss(scores, y)
        loss += 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2) + 0.5 * reg * np.sum(W3 * W3) # 用reg添加了正则化

        '''
        实现三层卷积网的backward，将损耗和梯度存储在损耗和梯度变量中，
        使用softma计算数据丢失，并添加L2正则化
        '''
        daffine_out, dW3, db3 = affine_backward(dscore, affine_cache)
        daffine_relu_out, dW2, db2 = affine_relu_backward(daffine_out, affine_relu_cache)
        daffine_relu_out_build = daffine_relu_out.reshape(conv_relu_pool_out.shape)
        dconv_relu_pool_out, dW1, db1 = conv_relu_pool_backward(daffine_relu_out_build, conv_relu_pool_cache)

        grads["W1"], grads["b1"] = dW1 + reg * W1, db1
        grads["W2"], grads["b2"] = dW2 + reg * W2, db2
        grads["W3"], grads["b3"] = dW3 + reg * W3, db3

        return loss, grads
