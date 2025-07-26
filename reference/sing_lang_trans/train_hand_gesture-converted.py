# %%
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# %%
%pwd
%cd ..

# %%
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
             'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
time_num, time_num1, time_num2 = '1669720403','1669723415', '1669724266'
for i in time_num:
    data = np.concatenate([
        np.load(f'dataset/seq_ㄱ_{time_num}.npy'),
        np.load(f'dataset/seq_ㄴ_{time_num}.npy'),
        np.load(f'dataset/seq_ㄷ_{time_num}.npy'),
        np.load(f'dataset/seq_ㄹ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅁ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅂ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅅ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅇ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅈ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅊ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅋ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅌ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅍ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅎ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅏ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅑ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅓ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅕ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅗ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅛ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅜ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅠ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅡ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅣ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅐ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅒ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅔ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅖ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅢ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅚ_{time_num}.npy'),
        np.load(f'dataset/seq_ㅟ_{time_num}.npy'),
        np.load(f'dataset/seq_ㄱ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㄴ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㄷ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㄹ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅁ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅂ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅅ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅇ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅈ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅊ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅋ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅌ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅍ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅎ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅏ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅑ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅓ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅕ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅗ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅛ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅜ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅠ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅡ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅣ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅐ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅒ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅔ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅖ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅢ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅚ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㅟ_{time_num1}.npy'),
        np.load(f'dataset/seq_ㄱ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㄴ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㄷ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㄹ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅁ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅂ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅅ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅇ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅈ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅊ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅋ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅌ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅍ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅎ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅏ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅑ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅓ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅕ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅗ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅛ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅜ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅠ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅡ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅣ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅐ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅒ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅔ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅖ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅢ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅚ_{time_num2}.npy'),
        np.load(f'dataset/seq_ㅟ_{time_num2}.npy')
    ], axis=0)

data.shape

# %%
print(np.load(f'sample_train\data\output\NIA_SL_WORD0001_REAL01_F_frame_range.npy').shape)
print(np.load(f'dataset/seq_ㄴ_1669720403.npy').shape)
print(np.load(f'dataset/seq_ㄷ_1669720403.npy').shape)
print(np.load(f'dataset/seq_ㄱ_1669723415.npy').shape)
print(np.load(f'dataset/seq_ㄴ_1669723415.npy').shape)
print(np.load(f'dataset/seq_ㄷ_1669723415.npy').shape)
print(np.load(f'dataset/seq_ㄱ_1669724266.npy').shape)
print(np.load(f'dataset/seq_ㄴ_1669724266.npy').shape)
print(np.load(f'dataset/seq_ㄷ_1669724266.npy').shape)

# %%
x_data = data[:, :, :-1]
labels = data[:, 0, -1]

print(x_data.shape)
print(labels.shape)

# %%
labels

# %%
len(labels)

# %%
np.unique(labels)

# %%
from tensorflow.keras.utils import to_categorical

y_data = to_categorical(labels, num_classes=len(actions))
y_data.shape

# %%
y_data

# %%
from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2020)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# %%
x_train.shape[1:3]

# %%
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# %%
# custom f1 score
def metric_F1score(y_true,y_pred): 
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras as keras

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3], kernel_regularizer=keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(len(actions), activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',metric_F1score])
model.summary()

# %%
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

early_stopping = EarlyStopping(monitor = 'val_metric_F1score', min_delta = 0, patience = 20, mode = 'auto')


history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/multi_hand_gesture_classifier.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_metric_F1score', factor=0.5, patience=50, verbose=1, mode='auto'),
        early_stopping
    ]
)

# %%
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()

# %%
fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')

acc_ax.plot(history.history['metric_F1score'], 'b', label='train f1')
acc_ax.plot(history.history['val_metric_F1score'], 'g', label='val f1')
acc_ax.set_ylabel('f1-score')
acc_ax.legend(loc='upper left')

plt.show()

# %%
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.models import load_model

model = load_model('models/multi_hand_gesture_classifier.h5', custom_objects = {'metric_F1score':metric_F1score})

y_pred = model.predict(x_val)
multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))

# %%
multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))

# %%
'''
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.models import load_model

model = load_model('models/multi_hand_gesture_classifier.h5')

y_pred = model.predict(x_val)

multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
'''

# %%
import tensorflow as tf
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('models/multi_hand_gesture_classifier.tflite', 'wb') as f:
    f.write(tflite_model)


