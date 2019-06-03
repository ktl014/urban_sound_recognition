""" """
# Standard dist imports
import argparse
import os
import time
# Third party imports

# Project level imports
from data.dataloader import get_dataloader

# Module level constants
root_dir = '/Users/ktl014/Google Drive/ECE Classes/ECE 228 Machine Learning w: Physical Applications/urban_sound'

parser = argparse.ArgumentParser('Prepare preprocessed dataset')
# parser.add_argument('--window_size', type=int,
#                  help='Window size')
parser.add_argument('--fold', type=int, default=1,
                    help='Fold to preprocess (default:1)')
parser.add_argument('--save', action='store_false', default=True,
                    help='Flag to save (STORE_FALSE)(default: True)')
parser.add_argument('--quick_dev', action='store_true', default=False,
                    help='Flag for quick development (STORE_FALSE)(default: True)')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Flag to debug (STORE_FALSE)(default: True)')

arg = parser.parse_args()
since = time.time()
if not arg.debug:
    if not os.path.exists(os.path.join('data/folds')):
        os.makedirs(os.path.join('data/folds'))
    print('Preparing dataset for fold {}'.format(arg.fold))
    loader = get_dataloader(fold=[arg.fold], db_prepped=False,
                            batch_size=1, shuffle=True, save=arg.save, quick_dev=arg.quick_dev)
else:
    print('Testing dataloader for fold {}'.format(arg.fold))
    loader = get_dataloader(fold=[arg.fold], db_prepped=True,
                            batch_size=1, shuffle=True, save=arg.save, quick_dev=arg.quick_dev)
    window, label = next(iter(loader))
    for ii in range(len(window)):
            img = window.numpy()[:, ii, :, :, :]
            lbl = label.numpy()
            print(ii, window.shape, window.min(), window.max(), window.dtype)
            print(ii, img.shape, img.min(), img.max(), img.dtype)
            print(ii, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)  
print('Completed! Time: {}'.format(time.time() - since))
