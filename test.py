import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 设置图片路径
image_path = "/home/yfq/Desktop/test/cag.jpeg"

# 加载保存的模型
model = tf.keras.models.load_model("saved_model/cat_dog_classifier")

# 加载并预处理图片
IMG_SIZE = 160

def load_and_preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 加载图像
image = load_and_preprocess_image(image_path)

# 预测
prediction = model.predict(image)[0][0]

# 显示结果
confidence = prediction if prediction > 0.5 else 1 - prediction
label = "Dog" if prediction > 0.5 else "Cat"
print(f"Prediction: {label} (Confidence: {confidence:.2%})")

# 显示图片
plt.imshow(np.squeeze(image))
plt.title(f"Predicted: {label}")
plt.axis("off")
plt.show()
