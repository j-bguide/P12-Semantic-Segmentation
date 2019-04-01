from model import get_model
from road_sequence import RoadSequence
import argparse
import os
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from callbacks import IoUCallback
from road_sequence import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--alpha', type=float, default=1.0, help='width multiplier in the MobileNetV2 paper')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--train-size', type=float, default=0.9)
parser.add_argument('--resolution', type=str, default='160,576', help='height,width')
parser.add_argument('--crop', type=str, default='50,9999', help='height crop range start,end')

args, _ = parser.parse_known_args()
args.resolution = tuple(int(x) for x in args.resolution.split(','))
args.crop = tuple(int(x) for x in args.crop.split(','))

log_dir = 'log_dir'
os.makedirs(log_dir, exist_ok=True)
train_image_fns, train_gt_fns, test_image_fns, test_gt_fns = load_data(train_size=args.train_size, seed=123)
train_sequence = RoadSequence(train_image_fns, batch_size=args.batch_size, gt_fns=train_gt_fns,
                              shuffle=True, augment=True, resolution=args.resolution, crop=args.crop)

test_sequence = RoadSequence(test_image_fns, batch_size=args.batch_size, gt_fns=test_gt_fns,
                             shuffle=False, augment=False, resolution=args.resolution, crop=args.crop)

model = get_model(alpha=args.alpha, dropout=args.dropout)

model.compile(loss=binary_crossentropy, optimizer=Adam(lr=5e-4), metrics=[])

callbacks = [
    IoUCallback(validation_data=test_sequence),
    ModelCheckpoint('ep-{epoch:02d}-val_iou-{val_iou:.3f}-val_loss-{val_loss:.3f}.hdf5',
                    save_best_only=True, monitor='val_iou', mode='max'),
    TensorBoard(log_dir=log_dir, write_graph=True, write_images=False),
    ReduceLROnPlateau(monitor='val_iou', mode='max', patience=5, verbose=1)
]
model.fit_generator(train_sequence, verbose=1, epochs=args.epochs,
                    validation_data=test_sequence, callbacks=callbacks)
