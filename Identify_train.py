import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image #从Keras中导入image模块 进行图片处理
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  StratifiedShuffleSplit
import h5py
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
            x = preprocess_input(x)#进行图像预处理
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
WIDTH = 48
HEIGHT = 48
CHANNELS = 3
#函数开始运行
X, y, label_encoder = read_and_process_image('input/', width = WIDTH, height = HEIGHT, channels = CHANNELS)
label_encoder.classes_
sss = StratifiedShuffleSplit(test_size=0.1, random_state=0)#StratifiedShuffleSplit数据集划分函数，n_splits将训练数据分成train/test对的组数
# sss.get_n_splits(X, y)
# print("sss.splits(X, y)",sss.split(X, y))
#看不太懂,为什么要用sss.split()
for train_index, test_index in sss.split(X, y):
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]
train_y = to_categorical(train_y) #将一个类向量转换为二进制类矩阵
test_y = to_categorical(test_y)
def vgg16_model(input_shape=(WIDTH, HEIGHT, CHANNELS)):
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    #print("vgg16.layers",vgg16.layers)

    for layer in vgg16.layers:
        layer.trainable = False

    last = vgg16.output

    # Please modify this part to fill your own fully connected layers.
    x = Flatten()(last)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='softmax')(x)

    model = Model(inputs=vgg16.input, outputs=x)

    return model
model_vgg16 = vgg16_model()
model_vgg16.summary()
model_vgg16.compile(loss='categorical_crossentropy', optimizer = Adam(0.001), metrics = ['accuracy'])#categorical_crossentropy
history = model_vgg16.fit(train_X, train_y, validation_data=(test_X, test_y), epochs = 20, batch_size = 8)
# Final evaluation of the model
scores = model_vgg16.evaluate(test_X, test_y, verbose = True)
model_vgg16.save('model4_test.h5')
print("VGG-16 Pretrained Model Error: %.2f%%" % (100 - scores[1] * 100))
y_test_pred = model_vgg16.predict(test_X)
y_label = np.argmax(y_test_pred, axis=1)
import matplotlib.cbook as cbook
for i in range(test_X.shape[0]):
    plt.figure(frameon=True)
    img = test_X[i, :, :, ::-1]
    img = img/255.
    plt.imshow(img)
    plt.title('result: {}'.format(label_encoder.inverse_transform([y_label[i]])[0]))
    plt.show()