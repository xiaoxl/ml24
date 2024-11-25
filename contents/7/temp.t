

```{python}
#| eval: false
#| warning: false
# import keras_core as keras
from keras import models, layers, Input
model = models.Sequential()

model.add(Input(shape=(X_train.shape[1],)))
model.add(layers.Dense(10, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=500, batch_size=30, validation_data=(X_test, y_test), verbose=0)

loss_train = hist.history['loss']
loss_val = hist.history['val_loss']

acc_train = hist.history['accuracy']
acc_val = hist.history['val_accuracy']
```


It seems that our model has overfitting issues. Therefore we need to modifify the architects of our model. The first idea is to add `L2` regularization as we talked about it in LogsiticRegression case. Here we use `0.01` as the regularization strenth.

Let us add the layer to the model and retrain it.


```{python}
#| output: false
#| warning: false
# import keras_core as keras
from keras import regularizers
model = models.Sequential()

model.add(layers.Dense(10, activation='sigmoid', input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=500, batch_size=30, validation_data=(X_test, y_test), verbose=0)

loss_train = hist.history['loss']
loss_val = hist.history['val_loss']

acc_train = hist.history['accuracy']
acc_val = hist.history['val_accuracy']
```

```{python}
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
ax[0].plot(loss_train, label='train_loss')
ax[0].plot(loss_val, label='val_loss')
ax[0].legend()

ax[1].plot(acc_train, label='train_acc')
ax[1].plot(acc_val, label='val_acc')
ax[1].legend()
```


Another way to deal with overfitting is to add a `Dropout` layer. The idea is that when training the model, part of the data will be randomly discarded. Then after fitting, the model tends to reduce the variance, and then reduce the overfitting. 

The code of a `Dropout` layer is listed below. Note that the number represents the percentage of the training data that will be dropped.



```{python}
#| output: false
# import keras_core as keras
from keras import regularizers
model = models.Sequential()

model.add(layers.Dense(10, activation='sigmoid', input_dim=X_train.shape[1]))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=500, batch_size=30, validation_data=(X_test, y_test), verbose=0)

loss_train = hist.history['loss']
loss_val = hist.history['val_loss']

acc_train = hist.history['accuracy']
acc_val = hist.history['val_accuracy']
```

```{python}
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
ax[0].plot(loss_train, label='train_loss')
ax[0].plot(loss_val, label='val_loss')
ax[0].legend()

ax[1].plot(acc_train, label='train_acc')
ax[1].plot(acc_val, label='val_acc')
ax[1].legend()
```


After playing with different hyperparameters, the overfitting issues seem to be better (but not entirely fixed). However, the overall performance is getting worse. This means that the model is moving towards underfitting side. Then we may add more layers to make the model more complicated in order to capture more information.


```{python}
#| output: false
# import keras_core as keras
from keras import regularizers
model = models.Sequential()

model.add(layers.Dense(10, activation='sigmoid', input_dim=X_train.shape[1]))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='sigmoid', input_dim=X_train.shape[1]))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=500, batch_size=30, validation_data=(X_test, y_test), verbose=0)

loss_train = hist.history['loss']
loss_val = hist.history['val_loss']

acc_train = hist.history['accuracy']
acc_val = hist.history['val_accuracy']
```

```{python}
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
ax[0].plot(loss_train, label='train_loss')
ax[0].plot(loss_val, label='val_loss')
ax[0].legend()

ax[1].plot(acc_train, label='train_acc')
ax[1].plot(acc_val, label='val_acc')
ax[1].legend()
```
