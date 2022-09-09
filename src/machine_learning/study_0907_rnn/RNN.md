# RNN
* 感受野：特征局部相关性

## 卷积操作
    #卷积操作 nn.Conv2d()
    layer = nn.Conv2d(1,3,kernel_size=3,stride = 1,padding= 0) # 第一个参数和输入的图片个数相同，第二个参数是是卷积核的个数（kernel的个数）
    第三个参数是卷积核的大小，第四个参数移动的步长，第五个参数是加入padding的大小,stride有降维的功能
    x = torch.rand(1,1,28,28)
    out = layer.forward(x) # [1,3,26,26]
    out = layer(x) #和上面功能类似 建议采用这个
    # F.conv2d()
    w = torch.rand(16,3,5,5) # 16个kernel,3和输入图片channel相同，5*5代表卷积核大小
    b = torch..rand(16)   # 16个kernel对应的bias
    out = F.conv2d(x,w,b,stride=1,padding = 1)

## 池化层
### pooling(下采样)
把feature map变小的操作：降维
* max pooling
* avg pooling


    x = out   # [1,16,14,14]
    layer = nn.MaxPool2d(2,stride= =2) # 第一个是window大小
    out = layer(x) # [1,16,7,7]
    out = F.avg_pool2d(x,2,stride=2)
### upsample(上采样)


    # 向上采样的方法 interpolate
    x = out  # [1,16,7,7]
    out = F.interpolate(xx,scale_factor = 2,mode = 'nearest') #[1,16,14,14] 第二个参数放大的倍数

把feature map变大的操作
### ReLU

    # 方式1
    layer = nn.ReLU(inplace = True)
    out = layer(x)
    # 方式2
    out = F.relu(x)

## Batch Norm
    
    # 1d
    x = torch.rand(100,16,784)
    layer = nn.BatchNorm1d(16) # 在图片的channel上进行求解均值和方差
    layer.running_mean  #均值
    layer.running_val  # 方差
    # 2d
