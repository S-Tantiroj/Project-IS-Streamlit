# import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# กำหนดพาธข้อมูล
train_dir = '../Neural_Network/train' 
valid_dir = '../Neural_Network/valid'
test_dir = '../Neural_Network/test'

# ตั้งค่า Data Augmentation และโหลดข้อมูล
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=False  # ไพ่ไม่ควรถูกกลับด้าน
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_test_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = valid_test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# สร้างโมเดล CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(53, activation='softmax')  # 53 คลาส
])

# คอมไพล์โมเดล
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# เทรนโมเดล
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=30,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)]
)

# แสดงกราฟความแม่นยำและ Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# ประเมินผลโมเดล
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy:.4f}')

# บันทึกโมเดลที่เทรนเสร็จแล้ว
model.save("card_classifier.keras")
