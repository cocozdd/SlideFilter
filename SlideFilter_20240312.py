##版本2  以ostu过滤后的二值化图像来保留组织区?坐标会保存为json文件
import argparse
import glob
import os.path
from tqdm import tqdm
import wsi.util as util
import wsi.filter as filter
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 7519517424
import numpy as np
from Slide.dispatch import openSlide
import json

parser = argparse.ArgumentParser(description="Cut Slide every ext")
parser.add_argument('--slideRoot', type=str, default=r'E:\һݡ\jiangsu_zigongjing', help='directory to save slide')
parser.add_argument('--ext', type=str, default='.kfb', help='.kfb .sdpc .svs')
parser.add_argument('--saveRoot', type=str, default="/run/user/1000/gvfs/sftp:host=10.15.20.76/homes/xingzehang/syy/data/patch/jiangsu_zigongjing", help='directory to save result')
parser.add_argument('--maskSave', type=bool, default=True, help='save thumb?')
args = parser.parse_args()
args.patchSaveRoot = os.path.join(args.saveRoot, 'patch')
args.listSaveRoot = os.path.join(args.saveRoot, 'patchList')
os.makedirs(args.patchSaveRoot, exist_ok=True)
os.makedirs(args.listSaveRoot, exist_ok=True)
if args.maskSave:
    args.thumbSaveRoot = os.path.join(args.saveRoot, 'thumb')
    os.makedirs(args.thumbSaveRoot, exist_ok=True)

def getThumb(slidePath):
    slideData = openSlide(slidePath)
    width = slideData.width
    height = slideData.height
    img_thumb = slideData.read((0, 0), (width, height), 32)
    return img_thumb

def ostuFilter(img_thumb):
    grayscale = filter.filter_rgb_to_grayscale(img_thumb)
    complement = filter.filter_complement(grayscale)
    ostu = filter.filter_otsu_threshold(complement)
    # util.display_img(ostu,"ostu")
    return ostu

def pathchFilter(ostu):
    block_size = 16
    patchRegion = []  ##patch的最高倍野上的坐标
    ostu_regiton = np.zeros_like(ostu)
    for row_start in range(0, ostu.shape[0], block_size):
        for col_start in range(0, ostu.shape[1], block_size):
            row_end = min(row_start + block_size, ostu.shape[0])
            col_end = min(col_start + block_size, ostu.shape[1])
            region_sum = ostu[row_start:row_end, col_start:col_end].sum()
            threshold = (row_end - row_start) * (col_end - col_start) * 255 * 0.2
            if region_sum > threshold:
                patchRegion.append([col_start * 32, row_start * 32])
                ostu_regiton[row_start:row_end, col_start:col_end] = 255
            else:
                ostu_regiton[row_start:row_end, col_start:col_end] = 0
    return ostu_regiton, patchRegion

def SlideCut(slidePath, patchRegionList,patchSaveRoot):
    slideData = openSlide(slidePath)
    for pix_w,pix_h in tqdm(patchRegionList):
        img = slideData.read((pix_w, pix_h), (512, 512), 2)
        if img is not None:
            img = Image.fromarray(img)
            img.save(f"{patchSaveRoot}/{str(pix_w)}_{pix_h}.jpg")
        else:
            print((pix_w, pix_h))

def main():
    slideAll = glob.glob(f"{args.slideRoot}/*{args.ext}")+glob.glob(f"{args.slideRoot}/*/*{args.ext}")
    for process_i, slide_item in enumerate(slideAll):
        imageName = os.path.splitext(os.path.basename(slide_item))[0]
        patchSaveRoot = os.path.join(args.patchSaveRoot, imageName)
        # if os.path.exists(patchSaveRoot):
        #     continue
        print(f"{process_i}/{len(slideAll)} processing slide {imageName}{args.ext}")
        ##读取预览?        img_thumb = getThumb(slide_item)
        ##生成ostu过滤图像
        ostu = ostuFilter(img_thumb)
        ##生成组织patch 坐标?组织块mask
        ostu_region, patchRegionList = pathchFilter(ostu)
        with open(f"{args.listSaveRoot}/{imageName}.json", 'w') as f:
            json.dump(patchRegionList, f)
        if args.maskSave == True:
            filterd_imag = np.repeat(ostu_region[:, :, np.newaxis], 3, axis=2)
            Image.fromarray(img_thumb).save(f"{args.thumbSaveRoot}/{imageName}_original.jpg")
            img_thumb[filterd_imag != 255] = 0
            img_thumb = Image.fromarray(img_thumb)
            img_thumb.save(f"{args.thumbSaveRoot}/{imageName}_thumb.jpg")
        os.makedirs(patchSaveRoot)
        SlideCut(slide_item, patchRegionList, patchSaveRoot)

if __name__ == '__main__':
    main()
