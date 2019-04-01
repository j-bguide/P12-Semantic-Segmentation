from keras.callbacks import Callback
import numpy as np


class IoUCallback(Callback):
  def __init__(self, validation_data):
    super(IoUCallback, self).__init__()
    self.validation_data = validation_data

  def on_epoch_end(self, epoch, logs=None):
    tp = 0
    fp = 0
    fn = 0
    for k in range(len(self.validation_data)):
      X, y = self.validation_data[k]
      y = y > 0.5
      y_pred = self.model.predict(X) > 0.5
      tp += np.logical_and(y_pred, y).sum()
      fp += np.logical_and(y_pred, ~y).sum()
      fn += np.logical_and(~y_pred, y).sum()

    logs['val_iou'] = tp / (tp + fp + fn)
    print('val_iou: %.4f' % (logs['val_iou']))
