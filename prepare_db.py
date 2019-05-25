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
parser.add_argument('--root_dir', type=str, default=root_dir,
                 help='Abs path of project repo')
# parser.add_argument('--window_size', type=int,
#                  help='Window size')
parser.add_argument('--fold', type=int, default=1,
                    help='Fold to preprocess (default:1)')
parser.add_argument('--save', action='store_false', default=True,
                    help='Flag to save (STORE_FALSE)(default: True)')

arg = parser.parse_args()
if not os.path.exists(os.path.join(arg.root_dir, 'data/folds')):
    os.makedirs(os.path.join(arg.root_dir, 'data/folds'))
print('Preparing dataset for fold {}'.format(arg.fold))
since = time.time()
loader = get_dataloader(fold=[arg.fold], root_dir=arg.root_dir, db_prepped=False,
                        batch_size=1, shuffle=True, save=arg.save)
print('Completed! Time: {}'.format(time.time() - since))
