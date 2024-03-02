from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
# utils_image是加distortion的
import utils_image
import argparse
import json
import os
import torch
import pycocotools.mask as maskUtils

# np.random.seed(666)

distortion = {
    1: 'noise',
    2: 'blur',
    3: 'jpeg',
}


def show_anns(anns, image):
    if len(anns) == 0:
        return
    print(len(anns))
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    dpi = plt.rcParams['figure.dpi']
    height, width = image.shape[:2]
    plt.figure(figsize=(width/dpi, height/dpi))
    plt.imshow(image)
    plt.axis('off')
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.6]])
        img[m] = color_mask
    ax.imshow(img)
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img = np.asarray(img)
    buf.close()
    plt.close()
    return img


def region_regress(anns, region_thresh, img_area, area_initial=0.01):
    while len(anns['annotations']) > region_thresh:
        anns['annotations'] = [m for m in anns['annotations'] if m['area'] > img_area * (area_initial**2)]
        # 0.05太大了
        area_initial += 0.005
    print(f'The area thresh is {area_initial**2:2f}.')
    return anns


def add_region_distortion(anns, image: np.ndarray):
    d_image = image.copy().astype(np.float32) / 255.
    # bbox_list=[]
    # distortion_list=[]
    if anns['annotations'] is not None:
        for i in range(len(anns['annotations'])):
            ann = anns['annotations'][i]
            # ann['segmentation']['counts']=torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            d_region = d_image[y: y+h, x: x+w]
            
            d_region_ori = d_region.copy()
            d_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()[y: y+h, x: x+w]
            # d_mask = ann['segmentation']['counts'][y: y+h, x: x+w]
            
            order = [1 if np.random.rand()< 0.8 else 0 for i in range(4)]
            d_region, distortion_type = utils_image.task(d_region, order)
            ann.update({'distortion':distortion_type})

            d_region_ori[d_mask == True] = d_region[d_mask == True]
            d_image[y: y+h, x: x+w] = d_region_ori
            
            # distortion_list.append(ann['distortion'])

    return np.clip((d_image * 255.).round(), 0, 255).astype(np.uint8)
        
        
def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_image', type=str, default='../DF2K/train/000002.png', required=False)
    parser.add_argument('--input_image', type=str, default='./autodl-tmp/DIV2K_train_HR/', required=False)
    parser.add_argument('--output_image', type=str, default='./autodl-tmp/DIV2K_train_HR_output/', required=False)
    parser.add_argument('--sam_weight', type=str, default='./EIQA/sam_vit_h_4b8939.pth', required=False)

    args = parser.parse_args()


    sam = sam_model_registry["default"](checkpoint=args.sam_weight)
    sam.to('cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)

    # dict_write={}
    image_all = os.listdir(args.input_image)
    for i in image_all:
        print('************{} is processing***********'.format(i))
        input_image = os.path.join(args.input_image, i)
        # print(input_image)

        image = Image.open(input_image).convert('RGB')
        image = np.array(image)

        h, w, _ = image.shape

        # mask_generator.min_mask_region_area = h * w * 0.09  ## 0.5**2

        masks = mask_generator.generate(image)

        masks = region_regress(masks, 15, img_area=h*w)

        masks=delete_overlap_anns(masks)

        # 看看效果
        # image = show_anns(masks, image)
        
        image, bbox_list, distortion_list = add_region_distortion(masks, image)
        # dict_write[i]={'bbox': bbox_list, 'distortion': distortion_list}


        output_path = args.output_image+i
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        new_data = {i: {'bbox': bbox_list, 'distortion': distortion_list}}

        with open("data1.json", "r", encoding="utf-8") as f:
            file = f.read()
            if len(file) > 0:
                old_data = json.loads(file)
            else:
                old_data = {}
            old_data.update(new_data)
        with open("data1.json", "w", encoding="utf-8") as f:
            json.dump(old_data, f)

        print('************{} is done***********'.format(i))

    # with open('data1.json','a') as f:   
    #     # 写'bbox'和'distortion'
    #     # f.write(json.dumps(dict((('name', name), list(ann.items())[2], list(ann.items())[-1]))))
    #     f.write(json.dumps(dict_write))


def delete_overlap_anns(anns, p=0.8):
    # 如果丢弃重复的，那么不同部分之间存在边缘微小重复，会导致丢弃过多
    # 如果丢弃存在包含关系的，那么存在不同部分之间非完全包含，仅大部分包含的情况
    if len(anns) == 0:
        return
    # 从大到小
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    i=0
    while i < len(sorted_anns):
        ann2int = sorted_anns[i]['segmentation'].astype(int)
        j=len(sorted_anns)-1
        while j>i:
            x = ann2int+sorted_anns[j]['segmentation'].astype(int)
            overlap=np.sum(x==2)
            thresh=np.sum(sorted_anns[j]['segmentation'].astype(int))*p
            # print('overlap',overlap,'thresh',thresh)
            if overlap>thresh:
            # 丢弃具有包含关系的
            # if -1 not in x:
                del sorted_anns[j]
            j-=1
        i+=1
    return sorted_anns

# 细节的东西一般都没有distortion
if __name__ == '__main__':
    main()


