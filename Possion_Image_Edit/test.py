from possion import *
import cv2
import numpy as np
import os
from os import path
import argparse
import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-base-dir', default='./data/bear', type=str)
    parser.add_argument('-src-path', default='source.jpg', type=str)
    parser.add_argument('-mask-path', default='mask.jpg', type=str)
    parser.add_argument('-dst-path', default='target.jpg', type=str)
    parser.add_argument('-result-path', default='./results')
    parser.add_argument('-pos', nargs="+", default=(200, 50), type=int)
    
    parser.add_argument('-type', default="bear", type=str)

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = get_args()

    src = cv2.imread(path.join(args.base_dir, args.src_path)).astype(np.float64)
    dst = cv2.imread(path.join(args.base_dir, args.dst_path)).astype(np.float64)
    mask_path = path.join(args.base_dir, args.mask_path)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
    else:
        mask = None

    # pdb.set_trace()
    
    seamless_clone = PossionEdit(src, mask, dst, args.pos)
    final_img = seamless_clone()
    cv2.imwrite(filename=path.join(args.result_path, f'{args.type}.jpg'), img=final_img)
