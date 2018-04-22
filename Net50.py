from keras.utils import to_categorical
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model,load_model
from keras.initializers import glorot_uniform
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedShuffleSplit
from keras.layers import Dropout
import os
import random
import numpy as np
import h5py

#获取训练数据
def read_and_process_image(data_dir, width=64, height=64, channels=3, preprocess=False):
    train_images = [data_dir + i for i in os.listdir(data_dir) ]#if i != '.DS_Store'#将所有的要训练的问图片的名称读取到train_images的列表中
    #print(train_images)
    random.shuffle(train_images)#将train_image中的内容随机排序
    #定义读取图片
    def read_image(file_path, preprocess):
        img = image.load_img(file_path, target_size=(height, width))
        #print("img",img)
        x = image.img_to_array(img)#使用keras中的img_to_array()函数，可以将一张图片转换成一个矩阵
        #print('img_to_array返回的结果：',x)
        x = np.expand_dims(x, axis=0)#将x增加维数
        if preprocess:
            #x = preprocess_input(x)#进行图像预处理
            pass
        return x
    #将所有的图片转换成一个2进制的形式进行保存
    def prep_data(images, preprocess):
        count = len(images)
        data = np.ndarray((count, width, height, channels), dtype=np.float32)
        for i, image_file in enumerate(images):
            #print("i的值：",i)
            image = read_image(image_file, preprocess)
            #print("image的值：",image)
            data[i] = image
        print("data",data.shape)

        return data

    def read_labels(file_path):
        # Using 1 to represent dog and 0 for cat
        labels = []
        label_encoder = LabelEncoder()
        for i in file_path:
            label = i.split('/')[1].split('.')[0].split('_')[0]
            labels.append(label)
        labels = label_encoder.fit_transform(labels)

        return labels, label_encoder

    X = prep_data(train_images, preprocess)#调用前面写的函数 将所有的图片转换成向量的形式进行保存并且返回
    labels, label_encoder = read_labels(train_images)#将训练好的模型跟所有的名字进行调用并且保存，labels包括所有的结果集，label_encoder是训练好的模型

    assert X.shape[0] == len(labels)

    print("Train shape: {}".format(X.shape))

    return X, labels, label_encoder

WIDTH = 64
HEIGHT = 64
CHANNELS = 3
#函数开始运行
X, y, label_encoder = read_and_process_image('image/', width = WIDTH, height = HEIGHT, channels = CHANNELS)
print(X.shape)
print(y.shape)
print(y)
#Y=to_categorical(y,3)
#Wprint(Y.shape)
#print(Y)
def identity_block(X,f,filters,stage,block):
    #参数含义：
    # X:输入的张量(m,h,w,c)
    #f:整数型，为主路径中间处卷积的窗口的大小
    #filters：python的整数列表，定义主路径CONV层中的过滤器数量
    #stage：整数型，用于命名层，依赖于在网络中的位置
    #block：字符串/字符型，用于命名层，依赖于在网络中的位置

    #Returns:
    #X:标志块的输出，形状的张量(h,w,c)

    #定义名称的基础
    conv_name_base='res'+str(stage)+block+'branch'
    bn_name_base='bn'+str(stage)+block+'branch'
    #检索过滤器
    F1,F2,F3=filters
    #保存输入变量
    X_shortcut=X
    #主路径的第一个分量
    X=Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
    X=Activation('relu')(X)

    #主路径的第二个分量
    X=Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding='same',name=conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X=Activation('relu')(X)

    #主路径的第三个分量
    X=Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="valid",name=conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2c')(X)

    #最后一步,将快捷键添加到主路径
    X=layers.add([X,X_shortcut])
    X=Activation('relu')(X)

    return X

#当输入和输出维度不匹配时，您可以使用这种类型的块。与标识块的不同之处在于，在快捷路径中有一个“卷积”层：
def convolutional_block(X,f,filters,stage,block,s=2):
   #X,f,filters,stage,block等跟上一个方法中的含义相同，s：整数类型，指定步幅
   # 定义名称的基础
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
   # 检索过滤器
    F1, F2, F3 = filters  # Save the input value
    X_shortcut = X
   # 第一步的卷积
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a', padding='valid',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

   #第二部的卷积
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    #第三步的卷积
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    #最后一步，对输捷径进行卷积
    X_shortcut=Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),name=conv_name_base+'1',padding='valid',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X=layers.add([X,X_shortcut])
    return X

#定义一个50层的残差网络
def ResNet50(input_shape=(64,64,3),classes=3):
    #实现50层的残差网络的结构
    #conv2D->batchnorm->relu->maxpool->convblock->idblock*2->convblock->idblock*3->
    # convblock->idblock*5->convblock->idblock*2->avgpool->toplayer
    #参数含义：
    #input_shape：数据集图形的形状
    #classes：整数的类型，分类的数量

    #Returns:
    #model--a Model() instance in Keras

    #将输入定义成有形状的张量
    X_input = Input(input_shape)
    #进行0填充
    X=ZeroPadding2D((3,3))(X_input)
    #第一步
    #使用64个7*7的过滤器，步长为(2,2),名字为‘conv1’，进行卷积
    X=Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name='nb_conv1')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)#池化层
    #第二步
    X=convolutional_block(X,f=3,filters=[64,64,256],stage=2,block='a',s=1)#三层
    X=identity_block(X,3,filters=[64,64,256],stage=2,block='b')#三层
    X=identity_block(X,3,filters=[64,64,256],stage=2,block='c')#三层
    X = Dropout(0.3)(X)

    #第三步
    X=convolutional_block(X,3,filters=[128,128,512],stage=3,block='a',s=2)
    X=identity_block(X,3,filters=[128,128,512],stage=3,block='b')
    X=identity_block(X,3,filters=[128,128,512],stage=3,block='c')
    X=identity_block(X,3,filters=[128,128,512],stage=3,block='d')
    #X = Dropout(0.3)(X)

    #第四步
    X=convolutional_block(X,3,filters=[256,256,1024],stage=4,block='a',s=2)
    X=identity_block(X,3,[256,256,1024],stage=4,block='b')
    X=identity_block(X,3,[256,256,1024],stage=4,block='c')
    X=identity_block(X,3,[256,256,1024],stage=4,block='d')
    X=identity_block(X,3,[256,256,1024],stage=4,block='e')
    X=identity_block(X,3,[256,256,1024],stage=4,block='f')
    X = Dropout(0.3)(X)

    #第五步
    X=convolutional_block(X,3,filters=[512,512,2048],stage=5,block='a',s=2)
    X=identity_block(X,3,[256,256,2048],stage=5,block='b')
    X=identity_block(X,3,[256,256,2048],stage=5,block='c')

    #avg pool
    X=AveragePooling2D(pool_size=(2,2),name='avg_pool')(X)
    #flatten,输出层
    X=Flatten()(X)#将整个矩阵进行压扁

    X=Dense(classes,activation='softmax',name='fc'+str(classes),kernel_initializer=glorot_uniform(seed=0))(X)

    #创建模型
    model = Model(inputs=X_input,outputs=X,name='ResNet50')
    return model
#划分训练集，划分出来的用来进行交叉检验
sss=StratifiedShuffleSplit(test_size=0.1,random_state=0)
for train_index,test_index in sss.split(X,y):
    train_X,train_Y = X[train_index],y[train_index]
    test_X,test_Y=X[test_index],y[test_index]
train_Y=to_categorical(train_Y)
test_Y=to_categorical(test_Y)
#调用残差网络
model = ResNet50(input_shape=(64,64,3),classes=3)
model.compile(loss='categorical_crossentropy',optimizer = Adam(0.0001),metrics=['accuracy'])#categorical_crossentropy
model.fit(train_X,train_Y,validation_data=(test_X,test_Y),epochs=15,batch_size=32)
model.save("Net50.h5")







