# pytorch
## 数据类型
![img.png](img.png)

    a = torch.randn(2,3)
    a.type === 'torch.FloatTensor'
    type(a) === torch.Tensor
    isinstance(a, torch.FloatTensor) === true
标量

    a = torch.tensor(1.)
    b = torch.tensor(1.3)
向量

    torch.tensor([1.1])
    torch.FloatTensor(1)
    torch.FloatTensor(2)
    data = np.ones(2)
    torch.from_numpy(data)
    torch.randn(2,3)
    torch.rand(1,2,3)  ## rnn
    torch.rand(2,3,28,28)  ## cnn
    a.numel()   ##tensor含有的元素个数
    a.dim() ##tensor的维度
## 创建tensor
1. from_numpy


    a=np.array([2,3,3])
    torch.from_numpy(a)
    torch.tensor([2,3.2])
    torch.FloatTensor([])
    torch.FloatTensor(2,3)
    Torch.empty(2,3)
    torch.rand(3,3)
    torch.rand_like(tensor_a)
    torch.randint(1,10,[3,3])
    torch.randn(3,3)
    torch.full([2,3],7)
    torch.full([],7)
    torch.arange(0,10)
    torch.arange(0,10,2)
    torch.linespace(0,10,steps=4)  # steps:数量
    torch.logspace(0,-1,setps = 10) # 返回10^x
    torch.ones(3,3)
    torch.zeros(3,3)
    torch.eye(3,4)
    torch.ones_like(tensor_a)
    torch.randperm(10) # tensor([1,3,4,2,5,9,0,9,8,7]) 用于shuffle  两个tensor进行索引shuffle匹配

## 索引切片


    a = torch.rand(4,3,28,,28) # dim 0 first
    a[:2] # 2*3*28*28
    a[:2,:1,:,:] # 2*1*28*28
    a[:2,1:,:,:] # 2*2*28*28
    a[:2,-1:,:,:] # 2*1*28*28
    
    ## select by steps
    a[:,:,0:28:2,0:28:2] # 4,3,14,14
    a[:,:,::2,::2]  # 4*3*14*14
    a.index_select(0,tensor([0,2])) # 2*3*28*28  第二个参数必须是tensor
    a.index_select(1,tensor([1,2])) # 4*2*28*28
    a.index_select(2,torch.arange(8)) # 4*3*8*28
    a[...] # 4*3*28*28
    a[0,...] # 3*28*28
   
    ## select by mask 通过掩码
    x = torch.randn(3,4)
    mask = x.ge(0.5)
    torch.masked_select(x,mask)
    ## select by flatten index
    src = torch.tensor([[4,3,5],[6,7,8]])
    torch.take(src,torch.tensor([0,2,5]))

## tensor维度变换


    ## view reshape 功能相同
    a = torch.rand(4,1,28,28)
    a.view(4,28*28)
    a.reshape(4,28*28)
    ## Squeeze unsqueeze
    a.unsqueeze(index/pos)  index in [-a.dim-1,a.dim()+1]
    a.unsqueeze(0) # 插入维度 正的 index在前面插入 负的在后面插入
    a.squeeze(index) # 删减维度  不加参数 把能挤压的都挤压 dim=1
    
    ## Expand repeat  expand不会复制数据(推荐) repeat会复制数据
    a = torch.rand(4,32,14,14)
    b = torch.rand(1,32,1,1)
    b.expand(4,32,14,14) # 某个维度不变的话可以写 -1
    b.repeat(4,32,1,1)  # 4*1024*1*1
    b.repeat(4,1,1,,1)  # 4*32*1*1 对应的维度上的复制的次数
    
    ## t() 转置 只使用二维矩阵
    b.t()

    ## transpose(a,b) 交换维度 一般和contiguous()方法一起使用  跟踪维度
    a2 = a.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)  # a = a2
    a1 = a.transpose(1,3).contiguous().view(4,3*32*32).view(4,3,32,32) # 会造成数据污染
    torch.all(torch.eq(a,a1))

    ## permute 交换维度 里面放的是交换维度的位置
    b.permute(0,3,3,1)

    ## broadcast自动扩展 不需要拷贝数据 unsqueeze+expand  从最小维度开始

    ## 拼接与拆分
    ## 拼接
    ## cat(concat)
    a = torch.rand(4,32,8)
    b = torch.rand(5,32,8)
    torch.cat([a,b],dim = 0) # [9,32,8]
    ## stack()
    a1 = torch.rand(4,3,16,32)
    a2 = torch.rand(4,3,16,32)
    torch.cat([a1,a2],dim = 2) # [4,3,32,32]
    torch.stack([a1,a2],dim =2) # [4,3,2,16,32] 类似分组 创建一个新的维度
    ## 拆分
    ## split() 根据长度拆分
    aa = torch.split([1,3],dim=0)  返回拆分后的list
    ## chunk()
    aa = torch.chunk(2,dim=0) # 按照数量进行拆分

## 数学运算

    ## 加减乘除 + - * /
    a+b
    ## 矩阵乘
    torch.mm(a,b) # 只适合2d
    torch.matmul(a,b) # 适合所有 也可以 @ 
    a@b
    a = torch.rand(4,3,28,64)
    b = torch.rand(4,3,64,32)
    torch.mutmul(a,b) # [4,3,28,32]
    c = torch.rand(4,1,64,32) 
    torch.matmul(a,c) # [4,3,28,32] 首先使用brodcast 在相乘
    # power
    a = torch.full([2,2],3)
    a.pow(2)
    a ** 2 
    # exp log
    a = torch.exp(torch.ones(2,2))
    torch.log(a)
    # floor ceil trunc(整数部分) frac(小数部分)
    # round() 四舍五入
    
    ## clamp 裁剪 特别常用 梯度裁剪
    grad = torch..rand(2,3)*15
    grad.max()
    grad.median()
    grad.clamp() # 只传一个数 是最小值 两个只（min,max） (min)
## 统计属性

    # norm(范数) 
    a = torch.full([8],1)
    b = a.view(2,4)
    c = a.view(2,2,2)
    a.norm(1) # 1范数
    b.norm(1)
    c.norm(1)
    a.norm(2) # 2范数
    b.norm(1,dim=1)
    b.norm(2,dim = 2)
    # mean
    # sum
    # prod  # 阶乘
    # max,min,argmin,argmax
    a = torch.arange(8).view(2,4).float
    a.min()
    a.max()
    a.mean()
    a.prod()
    a.argmax() # 返回最大值对应的序号 tensor
    a.argmin() # 返回最小值对应的序号 tensor
    a.argmax(dim = 1)
    # kthvalue,topk
    a.topk(3,dim=1) # 最大的k个
    a.topk(3,dim=1,Largest=False) # 最小的k个
    a.kthvalue(8,dim=1)  # 第k小的
    # dim,keepdim
    a.max(dim=1,keepdim = True) #和原来的a维度保持一致

    # 比较
    # >,>=,<,<=.!=,== 对每个元素进行比较
    torch.eq(a,b) # 返回一个tensor
    torch.equal(a,b) # 返回True 或者 False
## 高阶操作

### where


    torch.where(condition,x,y) -> tensor # 根据条件选择x或者y
    # condition
    cond = torch.rand(2,2)
    a= torch.ones(2,2)
    b = torch.zeros(2,2)
    c = torch.where(cond>0.5,a,b)

### Gather

    torch.gather(input,dim,index,out=None) -> tensor # 常用于查表操作 lable和对应的index
    prob = torch.randn(4,10)
    idx = prob.topk(dim=1,k=3)
    idx = idx[1]
    label = torch.arange(10)+100
    torch.gather(label.expand(4,10),dim=1,index = idx.long()) # gather和where 可以使用GPU进行加速

## 梯度计算

    #方法1"torch.autograd.grad(loss,[w1,w2,w3])
    #方法2：loss.backward
    w1.grad
    w2.grad

## 感知机

    # 单输出感知机
    x = torch.randn(1,10)
    w = torch.randn(1,10,requires_grad= True)
    o = torch.sigmoid(x@w.t())
    loss = F.mse_loss(torch.ones(1,1),o)  # 这里使用了brocast
    loss.backward
    w.grad
    # 多输出感知机
    x = torch.randn(1,10)
    w = torch.randn(2,10,requires_grad= True)
    o = torch.sigmoid(x@w.t())
    loss = F.mse_loss(torch.ones(1,1),o)
    loss.backward()
    w.grad

## 链式法则

    链式法则：求导
    


    
    



    
    
    