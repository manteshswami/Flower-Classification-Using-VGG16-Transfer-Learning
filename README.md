# 🌻🌹Flower Classification Using VGG16-Transfer Learning
*This project implements a deep learning–based flower classification system using VGG16 Transfer Learning.
The model classifies flower images into five categories:*
  - Daisy,
  - Dandelion,
  - Roses,
  - Sunflowers,
  - Tulips

*This is a complete end-to-end Computer Vision project covering:*
- ***dataset download → preprocessing → CNN feature extraction → training → evaluation → real image prediction.***

### 📌 Project Overview

Instead of training a CNN from scratch, this project uses VGG16 pretrained on ImageNet as a feature extractor.
A custom neural network is added on top to learn flower-specific patterns.

*This approach provides:*
- Faster training
- Higher accuracy
- Better generalization

### 🧠 Model Architecture
<pre><code> 
  Input Image (160×160×3) 
          ↓ 
  Pretrained VGG16 (Frozen) 
          ↓
  Global Average Pooling 
          ↓ 
    Dense (256, ReLU) 
          ↓ 
     Dropout (0.5) 
          ↓ 
        Dense 
     (5,Softmax) 
</code></pre>
### 📊 Results
*Metric	Value*
- Training Accuracy	~75.5%
- Validation Accuracy	~78%
- Training Loss	~0.85
- Validation Loss	~0.71

*The learning curves show stable training with no overfitting.*

### 📥 Clone the Repository
<pre><code> git clone https://github.com/manteshswami/Flower-Classification-Using-VGG16-Transfer-Learning.git cd Flower-Classification-Using-VGG16-Transfer-Learning </code></pre>
### 📦 Install Required Libraries
<pre><code> pip install tensorflow numpy matplotlib pillow scikit-learn </code></pre>
### 📌 Project Code
#### *Dataset Download*
<pre><code> 
  import os
  import tarfile 
  import urllib.request
  
  url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz" 
  dataset_dir="flower_photos" 
  tgz_path="flower_photos.tgz" 
  
  if not os.path.exists(tgz_path): 
    print("Downloading flower dataset...") 
    urllib.request.urlretrieve(url, tgz_path) 
    print("Download Completed") 
  
  if not os.path.exists(dataset_dir):
    print("Extracting flower dataset...")
    with tarfile.open(tgz_path, 'r:gz') as tar:
    tar.extractall()
    print("Extraction Completed")
  
  print("Available classes:", os.listdir(dataset_dir)) 
</code></pre>
#### *Dataset Split*
<pre><code> 
  from sklearn.model_selection import train_test_split 
  import shutil
  import os
  
  original_dir="flower_photos"
  bash_dir="dataset"
  os.makedirs(bash_dir, exist_ok=True) 
  
  for class_name in os.listdir(original_dir):
    class_path=os.path.join(original_dir, class_name) 
  if not os.path.isdir(class_path):
    continue
  
  images = os.listdir(class_path) 
  train, temp = train_test_split(images, test_size=0.2, random_state=42) 
  val, test = train_test_split(temp, test_size=0.5, random_state=42) 
  
  for split, split_data in zip(['train', 'val', 'test'], [train, val, test]):
    split_path = os.path.join(bash_dir, split, class_name) 
    os.makedirs(split_path, exist_ok=True) 
    for image in split_data:
      shutil.copy(os.path.join(class_path, image), os.path.join(split_path, image)) 
</code></pre>
#### *Data Generators*
<pre><code> 
  from tensorflow.keras.preprocessing.image import ImageDataGenerator 
  from tensorflow.keras.applications.vgg16 import preprocess_input 
  
  train_dir = "dataset/train" 
  val_dir = "dataset/val" 
  test_dir = "dataset/test" 
  img_size=(160,160) 
  batch_size = 32 
  
  train_datagen = ImageDataGenerator( preprocessing_function=preprocess_input, horizontal_flip=True, zoom_range=0.2) 
  val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 
  test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 
  
  train_generator = train_datagen.flow_from_directory( train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical') 
  val_generator = val_datagen.flow_from_directory( val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical') 
  test_generator = test_datagen.flow_from_directory( test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False) 
</code></pre>
#### *Model Building*
<pre><code>
  from tensorflow.keras.applications import VGG16 
  from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout 
  from tensorflow.keras.models import Model 
  from tensorflow.keras.optimizers import Adam 
  
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(160,160,3)) 
  
  for layer in base_model.layers:
    layer.trainable = False 
    x = base_model.output 
    x = GlobalAveragePooling2D()(x) 
    x = Dense(256, activation='relu')(x) 
    x = Dropout(0.5)(x) 
  
  predictions = Dense(5, activation='softmax')(x) 
  model = Model(inputs=base_model.input, outputs=predictions) 
  model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
</code></pre>
#### *Training*
<pre><code> 
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
  callback = [ EarlyStopping(patience=3, restore_best_weights=True), ModelCheckpoint("vgg16_best_model.h5", save_best_only=True) ]
  history = model.fit( train_generator, epochs=10, validation_data=val_generator, callbacks=callback) 
</code></pre>
#### *Prediction*
<pre><code> 
  from tensorflow.keras.preprocessing.image import load_img, img_to_array 
  import numpy as np 
  from tensorflow.keras.applications.vgg16 import preprocess_input 
  
  def predict_image(image_path):
    img = load_img(image_path, target_size=(160,160)) 
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) 
    prediction = model.predict(img_array) 
    predicted_class = np.argmax(prediction) 
    class_names = list(train_generator.class_indices.keys())
    print("Predicted Class:", class_names[predicted_class]) 
</code></pre>
### 📈 Training & Validation Performance
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/957e6cd8-a50f-4a40-938d-2736b2034782" />

### 🚀 How to Improve Accuracy

*The current model achieves ~78% validation accuracy using frozen VGG16 features. Accuracy can be further improved by:*
- Fine-tuning the last VGG16 convolution block with a lower learning rate
- **Training for more epochs after fine-tuning**
- Using stronger data augmentation (rotation, brightness, zoom, etc.)
- Trying more advanced models like **ResNet50 or EfficientNet**

*These techniques can increase performance to 85–90%+*

### 🏆 Conclusion
This project demonstrates a real-world deep learning application using VGG16 Transfer Learning for multi-class image classification.
It includes dataset preparation, CNN feature extraction, training, evaluation, and real-time prediction.
