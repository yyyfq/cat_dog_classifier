import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 载入猫狗数据集
(train_ds, val_ds), metadata = tf.keras.datasets.cifar10.load_data()
(train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.cifar10.load_data()

# 筛选出猫（3）和狗（5）
def filter_cat_dog(images, labels):
    cat_dog_mask = np.logical_or(labels == 3, labels == 5).squeeze()
    return images[cat_dog_mask], labels[cat_dog_mask]

train_images, train_labels = filter_cat_dog(train_images, train_labels)
val_images, val_labels = filter_cat_dog(val_images, val_labels)

# 数据预处理
IMG_SIZE = 160

def format_image(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # 归一化
    label = tf.cast(label == 5, tf.float32)  # 狗设为 1，猫为 0
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

train_ds = train_ds.map(format_image).batch(32).shuffle(1000)
val_ds = val_ds.map(format_image).batch(32)

# 加载预训练模型 MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# 添加分类头部
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译和训练模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=5)

model.save("saved_model/cat_dog_classifier")

# 测试图像显示
for images, labels in val_ds.take(1):
    pred = model.predict(images)
    plt.imshow(images[0])
    plt.title(f'Predicted: {"Dog" if pred[0][0] > 0.5 else "Cat"}')
    plt.axis('off')
    plt.show()
