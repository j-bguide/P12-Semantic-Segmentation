import argparse
import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm
from keras.models import load_model
from road_sequence import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model.hdf5', help='model filename')
parser.add_argument('--resolution', type=str, default='160,576', help='height,width')
parser.add_argument('--train-size', type=float, default=0.9)
parser.add_argument('--fn-glob', type=str, default='')
parser.add_argument('--crop', type=str, default='50,9999')

args, _ = parser.parse_known_args()
args.resolution = tuple(int(x) for x in args.resolution.split(','))
args.crop = tuple(int(x) for x in args.crop.split(','))


out_dir = 'val_predictions'
os.makedirs(out_dir, exist_ok=True)
_, _, test_image_fns, test_gt_fns = load_data(train_size=args.train_size, seed=123)

if args.fn_glob != '':
  test_image_fns = sorted(glob(args.fn_glob))

model = load_model(args.model)

for k in tqdm(range(len(test_image_fns))):
  img = cv2.imread(test_image_fns[k])[..., ::-1]
  h, w = img.shape[:2]
  crop = img[args.crop[0]: args.crop[1]]
  h_crop, w_crop = crop.shape[:2]
  pimg = cv2.resize(crop, args.resolution[::-1])
  pimg = np.float32(pimg) / 127.5 - 1.0
  pimg = np.expand_dims(pimg, 0)
  y_pred = model.predict(pimg)[0].squeeze(-1)
  y_pred = cv2.resize(y_pred, (w_crop, h_crop))
  y_pred = np.pad(y_pred, ((args.crop[0], h - (h_crop + args.crop[0])), (0, 0)), mode='constant')
  img[..., 1][y_pred > 0.5] = 255
  out_fn = os.path.join(out_dir, '%06d.png' % (k))
  cv2.imwrite(out_fn, img[..., ::-1])
