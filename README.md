# 太阳黑子分类
本项目实践于[阿里云天池比赛项目](https://tianchi.aliyun.com/competition/entrance/531803/forum)，通过 Paddle 深度学习框架搭建而成。
### 1.数据
数据由[比赛平台提供](https://tianchi.aliyun.com/competition/entrance/531803/information)。
数据集有三种类型：alpha、beta、betax；数据格式为 FITS 天文文件格式，使用 astropy.io.fits 读取 FITS 文件。
本项目，为方便数据可视化，将 FITS 天文图片数据格式转为 PNG 图片格式，由于图片是时序图片，所以在转换过程中，每隔 6 次做一次数据增强操作（原图、旋转90°、旋转180°、旋转270°、上下翻转、左右翻转）。<br>
**注**：若需要转换后的数据集，可联系[邮箱](zxqyiyang@google.com)。
### 2.配置
项目已发布在[百度AI Studio 平台](https://aistudio.baidu.com/aistudio/projectdetail/591709)，可注册登录后，进行实践。模型使用经典图像分类模型 ResNET，学习框架采用 Paddle。
### 参考资料
[1].空间环境人工智能预警创新工坊整理提供的太阳活动区观测图像磁分类数据集[FANG Y, et al., 2019]<br>
[2].[百度深度学习框架paddlpaddle](https://www.paddlepaddle.org.cn/)、百度AI Studio 平台<br>
