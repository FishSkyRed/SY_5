# 使用TensorFlow Lite Model Maker生成图像分类器模型 

## 安装并导入所需的库

运行以下命令:


```python
!pip install tflite-model-maker
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Requirement already satisfied: tflite-model-maker in d:\anaconda3\lib\site-packages (0.3.4)
    Requirement already satisfied: six>=1.12.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (1.16.0)
    Requirement already satisfied: tensorflow-datasets>=2.1.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (4.5.2)
    Requirement already satisfied: sentencepiece>=0.1.91 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.1.96)
    Requirement already satisfied: fire>=0.3.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.4.0)
    Requirement already satisfied: tensorflow-hub<0.13,>=0.7.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.12.0)
    Requirement already satisfied: tensorflow>=2.6.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (2.9.1)
    Requirement already satisfied: absl-py>=0.10.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (1.0.0)
    Requirement already satisfied: neural-structured-learning>=1.3.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (1.3.1)
    Requirement already satisfied: lxml>=4.6.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (4.6.3)
    Requirement already satisfied: flatbuffers==1.12 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (1.12)
    Requirement already satisfied: matplotlib<3.5.0,>=3.0.3 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (3.4.3)
    Requirement already satisfied: tflite-support>=0.3.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.4.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (1.25.11)
    Requirement already satisfied: tf-models-official==2.3.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (2.3.0)
    Requirement already satisfied: tensorflow-model-optimization>=0.5 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.7.2)
    Requirement already satisfied: tensorflow-addons>=0.11.2 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.17.0)
    Requirement already satisfied: pillow>=7.0.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (8.4.0)
    Requirement already satisfied: PyYAML>=5.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (6.0)
    Requirement already satisfied: numpy>=1.17.3 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (1.20.3)
    Requirement already satisfied: tensorflowjs>=2.4.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (3.18.0)
    Requirement already satisfied: numba==0.53 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.53.0)
    Requirement already satisfied: Cython>=0.29.13 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.29.24)
    Requirement already satisfied: librosa==0.8.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.8.1)
    Requirement already satisfied: joblib>=0.14 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (1.1.0)
    Requirement already satisfied: resampy>=0.2.2 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (0.2.2)
    Requirement already satisfied: scipy>=1.0.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (1.7.1)
    Requirement already satisfied: pooch>=1.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (1.6.0)
    Requirement already satisfied: audioread>=2.0.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (2.1.9)
    Requirement already satisfied: packaging>=20.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (20.9)
    Requirement already satisfied: soundfile>=0.10.2 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (0.10.3.post1)
    Requirement already satisfied: decorator>=3.0.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (5.1.0)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (0.24.2)
    Requirement already satisfied: llvmlite<0.37,>=0.36.0rc1 in d:\anaconda3\lib\site-packages (from numba==0.53->tflite-model-maker) (0.36.0)
    Requirement already satisfied: setuptools in d:\anaconda3\lib\site-packages (from numba==0.53->tflite-model-maker) (58.0.4)
    Requirement already satisfied: gin-config in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (0.5.0)
    Requirement already satisfied: kaggle>=1.3.9 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (1.5.12)
    Requirement already satisfied: google-cloud-bigquery>=0.31.0 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (3.1.0)
    Requirement already satisfied: psutil>=5.4.3 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (5.8.0)
    Requirement already satisfied: pandas>=0.22.0 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (1.3.4)
    Requirement already satisfied: py-cpuinfo>=3.3.0 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (8.0.0)
    Requirement already satisfied: google-api-python-client>=1.6.7 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (2.49.0)
    Requirement already satisfied: dataclasses in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (0.6)
    Requirement already satisfied: tf-slim>=1.1.0 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (1.1.0)
    Requirement already satisfied: opencv-python-headless in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (4.5.5.64)
    Requirement already satisfied: termcolor in d:\anaconda3\lib\site-packages (from fire>=0.3.1->tflite-model-maker) (1.1.0)
    Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5 in d:\anaconda3\lib\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.8.1)
    Requirement already satisfied: uritemplate<5,>=3.0.1 in d:\anaconda3\lib\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (4.1.1)
    Requirement already satisfied: httplib2<1dev,>=0.15.0 in d:\anaconda3\lib\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.20.4)
    Requirement already satisfied: google-auth<3.0.0dev,>=1.16.0 in d:\anaconda3\lib\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.6.6)
    Requirement already satisfied: google-auth-httplib2>=0.1.0 in d:\anaconda3\lib\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.1.0)
    Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in d:\anaconda3\lib\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (1.56.2)
    Requirement already satisfied: protobuf<4.0.0dev,>=3.15.0 in d:\anaconda3\lib\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (3.19.4)
    Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in d:\anaconda3\lib\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.26.0)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in d:\anaconda3\lib\site-packages (from google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (5.1.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in d:\anaconda3\lib\site-packages (from google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in d:\anaconda3\lib\site-packages (from google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (4.8)
    Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.4.1 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (2.3.0)
    Requirement already satisfied: proto-plus>=1.15.0 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (1.20.5)
    Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (2.8.2)
    Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (2.3.3)
    Requirement already satisfied: google-cloud-bigquery-storage<3.0.0dev,>=2.0.0 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (2.13.1)
    Requirement already satisfied: pyarrow<9.0dev,>=3.0.0 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (8.0.0)
    Requirement already satisfied: grpcio<2.0dev,>=1.38.1 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (1.46.3)
    Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in d:\anaconda3\lib\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (1.46.3)
    Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in d:\anaconda3\lib\site-packages (from google-resumable-media<3.0dev,>=0.6.0->google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (1.3.0)
    Requirement already satisfied: pyparsing!=3.0.0,!=3.0.1,!=3.0.2,!=3.0.3,<4,>=2.4.2 in d:\anaconda3\lib\site-packages (from httplib2<1dev,>=0.15.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (3.0.4)
    Requirement already satisfied: tqdm in d:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (4.62.3)
    Requirement already satisfied: python-slugify in d:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (5.0.2)
    Requirement already satisfied: certifi in d:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (2021.10.8)
    Requirement already satisfied: cycler>=0.10 in d:\anaconda3\lib\site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in d:\anaconda3\lib\site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker) (1.3.1)
    Requirement already satisfied: attrs in d:\anaconda3\lib\site-packages (from neural-structured-learning>=1.3.1->tflite-model-maker) (21.2.0)
    Requirement already satisfied: pytz>=2017.3 in d:\anaconda3\lib\site-packages (from pandas>=0.22.0->tf-models-official==2.3.0->tflite-model-maker) (2021.3)
    Requirement already satisfied: appdirs>=1.3.0 in d:\anaconda3\lib\site-packages (from pooch>=1.0->librosa==0.8.1->tflite-model-maker) (1.4.4)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in d:\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.4.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda3\lib\site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in d:\anaconda3\lib\site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in d:\anaconda3\lib\site-packages (from scikit-learn!=0.19.0,>=0.14.0->librosa==0.8.1->tflite-model-maker) (2.2.0)
    Requirement already satisfied: cffi>=1.0 in d:\anaconda3\lib\site-packages (from soundfile>=0.10.2->librosa==0.8.1->tflite-model-maker) (1.14.6)
    Requirement already satisfied: pycparser in d:\anaconda3\lib\site-packages (from cffi>=1.0->soundfile>=0.10.2->librosa==0.8.1->tflite-model-maker) (2.20)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (0.4.0)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (0.26.0)
    Requirement already satisfied: h5py>=2.9.0 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (3.2.1)
    Requirement already satisfied: tensorboard<2.10,>=2.9 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (2.9.0)
    Requirement already satisfied: tensorflow-estimator<2.10.0,>=2.9.0rc0 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (2.9.0)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (1.1.2)
    Requirement already satisfied: libclang>=13.0.0 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (14.0.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (3.3.0)
    Requirement already satisfied: wrapt>=1.11.0 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (1.12.1)
    Requirement already satisfied: keras<2.10.0,>=2.9.0rc0 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (2.9.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (3.10.0.2)
    Requirement already satisfied: google-pasta>=0.1.1 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (0.2.0)
    Requirement already satisfied: astunparse>=1.6.0 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (1.6.3)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in d:\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow>=2.6.0->tflite-model-maker) (0.37.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (1.8.1)
    Requirement already satisfied: werkzeug>=1.0.1 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (2.0.2)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (0.6.1)
    Requirement already satisfied: markdown>=2.6.8 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (3.3.7)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (0.4.6)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in d:\anaconda3\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (1.3.1)
    Requirement already satisfied: importlib-metadata>=4.4 in d:\anaconda3\lib\site-packages (from markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (4.8.1)
    Requirement already satisfied: zipp>=0.5 in d:\anaconda3\lib\site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (3.6.0)
    Requirement already satisfied: oauthlib>=3.0.0 in d:\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (3.2.0)
    Requirement already satisfied: typeguard>=2.7 in d:\anaconda3\lib\site-packages (from tensorflow-addons>=0.11.2->tflite-model-maker) (2.13.3)
    Requirement already satisfied: tensorflow-metadata in d:\anaconda3\lib\site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker) (1.8.0)
    Requirement already satisfied: promise in d:\anaconda3\lib\site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker) (2.3)
    Requirement already satisfied: dill in d:\anaconda3\lib\site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker) (0.3.5.1)
    Requirement already satisfied: dm-tree~=0.1.1 in d:\anaconda3\lib\site-packages (from tensorflow-model-optimization>=0.5->tflite-model-maker) (0.1.7)
    Requirement already satisfied: sounddevice>=0.4.4 in d:\anaconda3\lib\site-packages (from tflite-support>=0.3.1->tflite-model-maker) (0.4.4)
    Requirement already satisfied: pybind11>=2.6.0 in d:\anaconda3\lib\site-packages (from tflite-support>=0.3.1->tflite-model-maker) (2.9.2)
    Requirement already satisfied: text-unidecode>=1.3 in d:\anaconda3\lib\site-packages (from python-slugify->kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (1.3)
    Requirement already satisfied: colorama in d:\anaconda3\lib\site-packages (from tqdm->kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (0.4.4)
    

可能出现一个错误：Cannot uninstall 'llvmlite'，需要手动在Anaconda Navigator中Environments组件管理和卸载相关的Package,如下：

![image.png](img/1.png)

接下来导入相关库：


```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```

## 模型训练

### 获取数据

从storage.googleapis.com中下载本实验所需要的数据集。（image_path可以定制，默认是在用户目录的.keras\datasets中）


```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```

### 运行示例

1.加载数据集，将数据集分为训练数据和测试数据。


```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

```

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    

2.训练TensorFlow模型


```python
inception_v3_spec = image_classifier.ModelSpec(uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')
inception_v3_spec.input_image_shape = [240, 240]
model = image_classifier.create(train_data, model_spec=inception_v3_spec)
```

    INFO:tensorflow:Retraining the models...
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
     rasLayerV1V2)                                                   
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5
    

    D:\Anaconda3\lib\site-packages\keras\optimizers\optimizer_v2\gradient_descent.py:108: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    

    103/103 [==============================] - 19s 99ms/step - loss: 0.8843 - accuracy: 0.7649
    Epoch 2/5
    103/103 [==============================] - 11s 107ms/step - loss: 0.6573 - accuracy: 0.8965
    Epoch 3/5
    103/103 [==============================] - 10s 97ms/step - loss: 0.6244 - accuracy: 0.9157
    Epoch 4/5
    103/103 [==============================] - 10s 98ms/step - loss: 0.6082 - accuracy: 0.9208
    Epoch 5/5
    103/103 [==============================] - 11s 102ms/step - loss: 0.5923 - accuracy: 0.9378
    

3.评估模型


```python
loss, accuracy = model.evaluate(test_data)
```

    12/12 [==============================] - 4s 147ms/step - loss: 0.5972 - accuracy: 0.9346
    

4.导出TensorFlow Lite模型


```python
model.export(export_dir='.')
```

    INFO:tensorflow:Assets written to: C:\Users\86130\AppData\Local\Temp\tmpx90bnaev\assets
    

    INFO:tensorflow:Assets written to: C:\Users\86130\AppData\Local\Temp\tmpx90bnaev\assets
    D:\Anaconda3\lib\site-packages\tensorflow\lite\python\convert.py:766: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Saving labels in C:\Users\86130\AppData\Local\Temp\tmpuw20p8ur\labels.txt
    

    INFO:tensorflow:Saving labels in C:\Users\86130\AppData\Local\Temp\tmpuw20p8ur\labels.txt
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite
    

导出的Tensorflow Lite模型包含了元数据(metadata),其能够提供标准的模型描述。导出的模型存放在Jupyter Notebook当前的工作目录中。

### 放进项目测试

如图替换掉原来的训练模型：


![image.png](img/2.png)

在手机(物理机)上运行app:

![image.png](img/3.jpg)
