# pth-to-kmodel(k210)
提供一个将pth的模型转到kmodel的方法链/n

本次智能机器人比赛被转kmodel烦死了，网上查了一堆资料，不是没用的就是缺少文件的（ncc），最后从一个有资料的网站下了个ncc0.1，然后根据ChatGPT的命令完成了这个从pth转kmodel的过程。

在此点名maixhub这个网站，我曾经用maixhub训练过kmodel，但是maixhub上不能换主干网络，导致模型太大一直爆k210的运行内存，然后我加了maixhub的官方网站QQ群，发现里面都是对komdel太大的求问，没有人提供更换主干网络的方法或者提供一个更好的生成kmodel方法。

感谢csdn上的一篇文章提供了从pth-onnx-tflite-komdel的思路，我这个仓库就是以此为基础来上传的
# PyTorch `.pth` 转 K210 `.kmodel` 通用流程

## 1. 总体流程

```text
best.pth
↓
export_onnx.py
↓
yolo.onnx
↓
onnx2tf
↓
saved_model
↓
tflite
↓
ncc
↓
kmodel
↓
K210 / MaixPy
```

## 2. K210 模型设计要求

K210 / nncase 0.1 对模型结构比较严格。

推荐使用
```
Conv2d stride=1
MaxPool2d 下采样
BatchNorm2d
ReLU
普通 Conv
固定输入尺寸
```
尽量避免
```
stride=2 Conv
DepthwiseConv
GroupConv
ZeroPad
非对称 Padding
动态 shape
复杂后处理算子
NMS 放进模型
```
## 3. ONNX 转 TensorFlow / TFLite
使用 onnx2tf。
```
onnx2tf -i yolo.onnx -o saved_model -nuo
```
-nuo 表示不让 onnx2tf 内部再次调用 onnxsim
成功标志：
```
saved_model output complete!
Float32 tflite output complete!
Float16 tflite output complete!
```
## 4. 查找 TFLite 文件
转换完成后执行：
```
dir /s /b *.tflite
```
通常会看到：
```
saved_model\xxx_float32.tflite
saved_model\xxx_float16.tflite
```
优先使用：
```
float32.tflite
```
不要优先使用 float16。
## 5. 准备 ncc 校准图片目录
ncc --dataset 用于 int8 量化校准，只需要图片，不需要标签。
不同类别图片数量尽量均衡。
## 6. TFLite 转 kmodel
使用 nncase 0.1 的 ncc.exe。
"D:address\ncc.exe" yolo.tflite yolo.kmodel -i tflite -o k210model --dataset D:\address\climg
