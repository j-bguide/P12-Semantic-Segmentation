from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import UpSampling2D, Conv2D, Add, Concatenate, Activation, Dropout
from keras.models import Model


def create_pyramid_features(x4, x8, x16, x32, feature_size=256):
  x32 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', kernel_initializer="he_normal")(x32)
  x32_upsampled = UpSampling2D()(x32)

  x16 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', kernel_initializer="he_normal")(x16)
  x16 = Add()([x32_upsampled, x16])
  x16_upsampled = UpSampling2D()(x16)

  x8 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', kernel_initializer="he_normal")(x8)
  x8 = Add()([x16_upsampled, x8])
  x8_upsampled = UpSampling2D()(x8)

  x4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', kernel_initializer="he_normal")(x4)
  x4 = Add()([x8_upsampled, x4])

  return x4, x8, x16, x32


def network_head(x, feature_size=128):
  x = Conv2D(feature_size, kernel_size=3, strides=1, activation='relu',
             padding='same', kernel_initializer="he_normal")(x)
  x = Conv2D(feature_size, kernel_size=3, strides=1, activation='relu',
             padding='same', kernel_initializer="he_normal")(x)
  return x


def get_model(num_classes=1, weights='imagenet', input_shape=(160, 576, 3), alpha=1.0, dropout=0.2):
  base = MobileNetV2(input_shape=input_shape, alpha=alpha, weights=weights, include_top=False)

  x32 = base.get_layer('out_relu').output
  x16 = base.get_layer('block_13_expand_relu').output
  x8 = base.get_layer('block_6_expand_relu').output
  x4 = base.get_layer('block_3_expand_relu').output
  x4, x8, x16, x32 = create_pyramid_features(x4, x8, x16, x32, feature_size=192)

  x4 = network_head(x4, feature_size=96)
  x8 = network_head(x8, feature_size=96)
  x16 = network_head(x16, feature_size=96)
  x32 = network_head(x32, feature_size=96)

  x = Concatenate()([
      UpSampling2D(size=(8, 8))(x32),
      UpSampling2D(size=(4, 4))(x16),
      UpSampling2D(size=(2, 2))(x8),
      x4
  ])

  x = Dropout(dropout)(x)
  x = Conv2D(num_classes, kernel_size=3, strides=1,
             padding='same', kernel_initializer="he_normal")(x)
  x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
  x = Activation('sigmoid')(x)
  return Model(base.input, x)
