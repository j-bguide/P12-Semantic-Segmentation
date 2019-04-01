from keras.utils import Sequence
import cv2
import numpy as np
from glob import glob

BACKGROUND_COLOR = [255, 0, 0]


def load_data(train_size=0.9, seed=123):
  image_fns = sorted(glob('data_road/training/image_2/*.png'))
  gt_fns = sorted(glob('data_road/training/gt_image_2/*road*.png'))

  train_len = int(train_size * len(image_fns))
  np.random.seed(seed=seed)
  perm = np.random.permutation(len(image_fns))

  train_image_fns = [image_fns[p] for p in perm[:train_len]]
  train_gt_fns = [gt_fns[p] for p in perm[:train_len]]

  test_image_fns = [image_fns[p] for p in perm[train_len:]]
  test_gt_fns = [gt_fns[p] for p in perm[train_len:]]
  return train_image_fns, train_gt_fns, test_image_fns, test_gt_fns


def normalize_tf(image):
  return np.float32(image) / 127.5 - 1.0


def augmentation_fn(image, gt):
  if np.random.random() < 0.5:
    image = image[:, ::-1]
    gt = gt[:, ::-1]

  if np.random.random() < 0.5:
    image = change_illumination(image)

  if np.random.random() < 0.5:
    image = blur(image, k=np.random.choice([3, 5, 7]))

  return image, gt


class RoadSequence(Sequence):
  def __init__(self, image_fns, batch_size=32, gt_fns=None,
               shuffle=False, augment=False, resolution=(160, 576),
               crop=(100, 9999), normalize_fn=normalize_tf):
    self.image_fns = image_fns
    self.batch_size = batch_size
    self.gt_fns = gt_fns
    self.shuffle = shuffle
    self.augment = augment
    self.resolution = resolution
    self.normalize_fn = normalize_fn
    self.crop = crop
    self.shuffle_data()

  def __len__(self):
    return int(np.ceil(len(self.image_fns) / float(self.batch_size)))

  def shuffle_data(self):
    if self.shuffle:
      perm = np.random.permutation(range(len(self.image_fns)))
      self.image_fns = [self.image_fns[p] for p in perm]
      self.gt_fns = [self.gt_fns[p] for p in perm]

  def on_epoch_end(self):
    self.shuffle_data()

  def __getitem__(self, batch_index):
    X, y = [], []
    for k in range(self.batch_size):
      index = batch_index * self.batch_size + k
      if index >= len(self.image_fns):
        break
      image_fn = self.image_fns[index]

      # read as RGB
      image = cv2.imread(image_fn, 1)[..., ::-1]
      image = image[self.crop[0]: self.crop[1]]
      image = cv2.resize(image, self.resolution[::-1])

      if self.gt_fns is not None:
        gt_fn = self.gt_fns[index]
        # read as RGB
        gt = cv2.imread(gt_fn, 1)[..., ::-1]
        foreground = (~np.all(gt == BACKGROUND_COLOR, -1)).astype(np.float32)
        gt = foreground
        gt = gt[self.crop[0]: self.crop[1]]
        gt = cv2.resize(gt, self.resolution[::-1])
        gt = np.expand_dims(gt, -1)

      if self.augment:
        image, gt = augmentation_fn(image, gt)

      image = self.normalize_fn(image)
      X.append(image)

      if self.gt_fns is not None:
        y.append(gt)

    X = np.float32(X)
    y = np.float32(y)

    if self.gt_fns is not None:
      return X, y

    return X


def blur(image, k=7):
  image = cv2.GaussianBlur(image, (k, k), 0)
  return image


def change_illumination(image, sat_limit=(-12, 12), val_limit=(-30, 30)):
  image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  h, s, v = cv2.split(image)
  sat_shift = np.random.uniform(sat_limit[0], sat_limit[1])
  s = cv2.add(s, sat_shift)
  value_shift = np.random.uniform(val_limit[0], val_limit[1])
  v = cv2.add(v, value_shift)
  image = np.dstack((h, s, v))
  image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
  return image
