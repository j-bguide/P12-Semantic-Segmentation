import pytest
import numpy as np
from src.model import get_model


def test_model_valid_input_shape():
  num_classes = 1
  for input_shape in [(160, 576, 3), (320, 1152, 3)]:
    model = get_model(num_classes=num_classes, weights=None, input_shape=input_shape)
    y_pred = model.predict(np.expand_dims(np.random.randn(*input_shape), 0))
    assert(y_pred.shape[1:3] == input_shape[:2])
    assert(y_pred.shape[3] == num_classes)


def test_model_invalid_input_shape():
  num_classes = 1
  for input_shape in [(123, 567, 3)]:
    with pytest.raises(ValueError):
      get_model(num_classes=num_classes, weights=None, input_shape=input_shape)


if __name__ == '__main__':
  pytest.main([__file__])
