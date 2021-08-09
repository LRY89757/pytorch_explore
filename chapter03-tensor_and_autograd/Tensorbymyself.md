```python
from __future__ import print_function

import torch as t

t.__version__
```




    '1.8.1+cpu'

## Tensor

Tensor，又名张量，读者可能对这个名词似曾相识，因它不仅在PyTorch中出现过，它也是Theano、TensorFlow、
Torch和MxNet中重要的数据结构。关于张量的本质不乏深度的剖析，但从工程角度来讲，可简单地认为它就是一个数组，且支持高效的科学计算。它可以是一个数（标量）、一维数组（向量）、二维数组（矩阵）和更高维的数组（高阶数据）。Tensor和Numpy的ndarrays类似，但PyTorch的tensor支持GPU加速。

本节将系统讲解tensor的使用，力求面面俱到，但不会涉及每个函数。对于更多函数及其用法，读者可通过在IPython/Notebook中使用函数名加`?`查看帮助文档，或查阅PyTorch官方文档[^1]。

[^1]: http://docs.pytorch.org

###  基础操作

学习过Numpy会感到非常熟悉，因tensor的接口有意设计成与Numpy类似，以方便用户使用。

从接口的角度来讲，对tensor的操作可分为两类：

1. `torch.function`，如`torch.save`等。
2. 另一类是`tensor.function`，如`tensor.view`等。

为方便使用，对tensor的大部分操作同时支持这两类接口，在本书中不做具体区分，如`torch.sum (torch.sum(a, b))`与`tensor.sum (a.sum(b))`功能等价。

而从存储的角度来讲，对tensor的操作又可分为两类：

1. 不会修改自身的数据，如 `a.add(b)`， 加法的结果会返回一个新的tensor。
2. 会修改自身的数据，如 `a.add_(b)`， 加法的结果仍存储在a中，a被修改了。

函数名以`_`结尾的都是inplace方式, 即会修改调用者自己的数据，在实际应用中需加以区分。

#### 创建Tensor

在PyTorch中新建tensor的方法有很多，具体如表3-1所示。

表3-1: 常见新建tensor的方法

|函数|功能|
|:---:|:---:|
|Tensor(\*sizes)|基础构造函数|
|tensor(data,)|类似np.array的构造函数|
|ones(\*sizes)|全1Tensor|
|zeros(\*sizes)|全0Tensor|
|eye(\*sizes)|对角线为1，其他为0|
|arange(s,e,step|从s到e，步长为step|
|linspace(s,e,steps)|从s到e，均匀切分成steps份|
|rand/randn(\*sizes)|均匀/标准分布|
|normal(mean,std)/uniform(from,to)|正态分布/均匀分布|
|randperm(m)|随机排列|

这些创建方法都可以在创建的时候指定数据类型dtype和存放device(cpu/gpu).


其中使用`Tensor`函数新建tensor是最复杂多变的方式，它既可以接收一个list，并根据list的数据新建tensor，也能根据指定的形状新建tensor，还能传入其他的tensor，下面举几个例子。


```python
# 指定tensor形状

a = t.Tensor(2, 3)  # 提前分配好空间

a   # 数值取决于内存空间的状态，
```




    tensor([[0., 0., 0.],
            [0., 0., 0.]])




```python
# 使用list数据创建tensor

b = t.Tensor([[1, 2, 3], [4, 5, 6]])

b
```




    tensor([[1., 2., 3.],
            [4., 5., 6.]])




```python
b.tolist()   # 和numpy的转化何其相似
```




    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]



`tensor.size()`返回`torch.Size`对象，它是tuple的子类，但其使用方式与tuple略有区别


```python
b_size = b.size()

b_size
```




    torch.Size([2, 3])




```python
b.numel()  #  b中元素总个数，2*3，等价于b.nelement()
```




    6




```python
# 创建一个与b形状一样的tensor

c = t.Tensor(b.size())

# 创建一个元素为2和3的tensor

d = t.Tensor((2, 3))   # 注意和t.Tensor(2, 3)完全不一样！

c, d
```




    (tensor([[0., 0., 0.],
             [0., 0., 0.]]),
     tensor([2., 3.]))




```python
c.shape   # equal to c.size()
```




    torch.Size([2, 3])



需要注意的是，`t.Tensor(*sizes)`创建tensor时，系统不会马上分配空间，只是会计算剩余的内存是否足够使用，使用到tensor时才会分配，而其它操作都是在创建完tensor之后马上进行空间分配。其它常用的创建tensor的方法举例如下。


```python
t.ones(2, 3)
```




    tensor([[1., 1., 1.],
            [1., 1., 1.]])




```python
print(t.zeros(2, 3))

print(t.arange(1, 20, 4))

print(t.linspace(1, 10, 3), '\n', t.linspace(1, 10, 9))
```

    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    tensor([ 1,  5,  9, 13, 17])
    tensor([ 1.0000,  5.5000, 10.0000]) 
     tensor([ 1.0000,  2.1250,  3.2500,  4.3750,  5.5000,  6.6250,  7.7500,  8.8750,
            10.0000])



```python
print(t.randn(2, 3, device=t.device('cpu')))
```

    tensor([[ 0.6485,  0.8807,  0.5077],
            [-1.1991, -0.9738, -0.8757]])



```python
t.randperm(5)  # 长度为5的随机排列
```




    tensor([1, 3, 2, 0, 4])




```python
print(t.eye(4, 2, dtype=t.int))

t.eye(2, 3, dtype=t.int)  # 对角线为1， 不要求行列数一致
```

    tensor([[1, 0],
            [0, 1],
            [0, 0],
            [0, 0]], dtype=torch.int32)





    tensor([[1, 0, 0],
            [0, 1, 0]], dtype=torch.int32)



`torch.tensor`是在0.4版本新增加的一个新版本的创建tensor方法，使用的方法，和参数几乎和`np.array`完全一致


```python
scalar = t.tensor(3.14159)

print('scalar: %s, shape of scalar: %s' %(scalar, scalar.shape)) # 零维标量
```

    scalar: tensor(3.1416), shape of scalar: torch.Size([])



```python
vector = t.tensor([1, 2])

print('vector: %s, shape of vector: %s' %(vector, vector.shape)) # 向量
```

    vector: tensor([1, 2]), shape of vector: torch.Size([2])



```python
tensor = t.Tensor(1, 2)  # 注意与t.tensor([1, 2])区别

tensor.shape
```




    torch.Size([1, 2])




```python
matrix = t.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])

matrix, matrix.shape
```




    (tensor([[0.1000, 1.2000],
             [2.2000, 3.1000],
             [4.9000, 5.2000]]),
     torch.Size([3, 2]))




```python
empty_tensor = t.tensor([])

empty_tensor.shape
```




    torch.Size([0])



通过`tensor.view`方法可以调整tensor的形状，但必须保证调整前后元素总数一致。`view`不会修改自身的数据，返回的新tensor与源tensor共享内存，也即更改其中的一个，另外一个也会跟着改变。在实际应用中可能经常需要添加或减少某一维度，这时候`squeeze`和`unsqueeze`两个函数就派上用场了。


```python
a = t.arange(0, 6)

a.view(2, 3)
```




    tensor([[0, 1, 2],
            [3, 4, 5]])




```python
b = a.view(-1, 3)

b.shape
```




    torch.Size([2, 3])




```python
b.unsqueeze(1)  # 注意形状，在第1维（下标从0开始）上增加“1”

# 等价于b[, None]

b[:, None].shape
```




    torch.Size([2, 1, 3])




```python
b.unsqueeze(-2)  # -2表示倒数第二个维度
```




    tensor([[[0, 1, 2]],
    
            [[3, 4, 5]]])




```python
c = b.view(1, 1, 1, 2, 3)

c.squeeze(0) # 压缩第0维的“1”
```




    tensor([[[[0, 1, 2],
              [3, 4, 5]]]])




```python
c.squeeze()  # 把所有维度为“1”的压缩
```




    tensor([[0, 1, 2],
            [3, 4, 5]])




```python
a[1] = 100

b  # a修改， b作为view之后的，也会跟着修改
```




    tensor([[  0, 100,   2],
            [  3,   4,   5]])




```python
help(t.squeeze)
```

    Help on built-in function squeeze:
    
    squeeze(...)
        squeeze(input, dim=None, *, out=None) -> Tensor
        
        Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.
        
        For example, if `input` is of shape:
        :math:`(A \times 1 \times B \times C \times 1 \times D)` then the `out` tensor
        will be of shape: :math:`(A \times B \times C \times D)`.
        
        When :attr:`dim` is given, a squeeze operation is done only in the given
        dimension. If `input` is of shape: :math:`(A \times 1 \times B)`,
        ``squeeze(input, 0)`` leaves the tensor unchanged, but ``squeeze(input, 1)``
        will squeeze the tensor to the shape :math:`(A \times B)`.
        
        .. note:: The returned tensor shares the storage with the input tensor,
                  so changing the contents of one will change the contents of the other.
        
        .. warning:: If the tensor has a batch dimension of size 1, then `squeeze(input)`
                  will also remove the batch dimension, which can lead to unexpected
                  errors.
        
        Args:
            input (Tensor): the input tensor.
            dim (int, optional): if given, the input will be squeezed only in
                   this dimension
        
        Keyword args:
            out (Tensor, optional): the output tensor.
        
        Example::
        
            >>> x = torch.zeros(2, 1, 2, 1, 2)
            >>> x.size()
            torch.Size([2, 1, 2, 1, 2])
            >>> y = torch.squeeze(x)
            >>> y.size()
            torch.Size([2, 2, 2])
            >>> y = torch.squeeze(x, 0)
            >>> y.size()
            torch.Size([2, 1, 2, 1, 2])
            >>> y = torch.squeeze(x, 1)
            >>> y.size()
            torch.Size([2, 2, 1, 2])


​    


```python

```
