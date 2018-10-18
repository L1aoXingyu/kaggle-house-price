# kaggle 房价预测比赛

## 项目介绍

在这个项目中，我们给定一个一层神经网络的baseline以及一些帮助函数，大家可以自由建立任意的神经网络来对[kaggle房价预测](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)的比赛进行预测，通过该项目，我们能够对神经网络以及调参有一个更加全面的了解。

## 项目下载

打开终端，运行
```bash
git clone https://github.com/L1aoXingyu/kaggle-house-price.git
```
能够自动下载项目，或者网页直接下载

<div align=center>
<img src='https://ws1.sinaimg.cn/large/006tNbRwly1fw46x2kesfj30ti0aldgb.jpg' width='600'>
</div>

通过上面的过程，我们准备好了项目，在开始之前，需要根据 [StartKit](https://github.com/sharedeeply/DeepLearning-StartKit) 配置好了深度学习环境，所以请按照操作完成深度学习环境的配置，当你完成好环境配置之后，你可以直接进入 `predict-house-prices.ipynb` 完成项目。

## 数据下载
我们已经将数据集放在了项目中，大家根据上面下载好项目之后，便能在`all`中看到所有的数据集。

## 评估与提交

通过`predict-house-prices.ipynb`，你会建立一个模型进行房价的预测，同时在测试集上能够看到模型的效果，最后可以得到一个最优的模型，并在 testset 上面运行结果，在 kaggle 的[提交页面](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)上面按照下面的步骤提交。

1. 点击提交结果

<div align=center>
<img src='https://ws1.sinaimg.cn/large/006tNbRwly1fw46qul9fgj30vk0o9jtf.jpg' width='500'>
</div>

2. 提交本地生成的文件

<div align=center>
<img src='https://ws1.sinaimg.cn/large/006tNbRwly1fw46r2epp9j30tr0c50t6.jpg' width='500'>
</div>

3. 提交结果

<div align=center>
<img src='https://ws2.sinaimg.cn/large/006tNbRwly1fw46rclzw5j30x20j8mxq.jpg' width='500'>
</div>

4. 查看结果

<div align=center>
<img src='https://ws1.sinaimg.cn/large/006tNbRwly1fw46ryv28rj30sk0ln0tw.jpg' width='500'>
</div>

可以考虑在 Github 上为该项目创建一个仓库，记录训练的过程、所使用的库以及数据等的 README 文档，构建一个完善的 Github 简历。
