# example of using the upsampling layer
import numpy as np
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import UpSampling2D

# define input data
X = np.array([[1, 2],
              [3, 4]])

# show input data for context
print(X)

# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))  # no. of samples, rows, columns, and channels

# print('\nX reshaped: \n', X)

# define model
model = Sequential()
model.add(UpSampling2D(input_shape=(2, 2, 1),
                       size=(2, 2),  # size 2x rows, 2x columns
                       interpolation='bilinear'))  # 'nearest' or 'bilinear'
# summarize the model
model.summary()

# make a prediction with the model
yhat = model.predict(X)

# print('\nyhat before shaping bach: \n', yhat)

# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))

# summarize output
print('yhat = \n', yhat)

# plot graph
plot_model(model, to_file='upsampling.png')
