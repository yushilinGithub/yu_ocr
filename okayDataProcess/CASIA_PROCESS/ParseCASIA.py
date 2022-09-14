# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: data_process.py
# @Author: SWHL
# @Contact: liekkaskono@163.com
from asyncio.format_helpers import extract_stack
import struct
from pathlib import Path
import os
from PIL import Image
import zipfile
import os
import struct
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm
def write_txt(save_path: str, content: list, mode='w'):
    """
    将list内容写入txt中
    @param
    content: list格式内容
    save_path: 绝对路径str
    @return:None
    """
    with open(save_path, mode, encoding='utf-8') as f:
        for value in content:
            f.write(value + '\n')

def extract_zip(zip_files):
    
    
    for zip_file in zip_files:
        print("extract_zip",zip_file)
        with zipfile.ZipFile(zip_file,"r") as zip_ref:
            des_path = zip_file.parents[0] / zip_file.stem
            if not des_path.exists():
                zip_ref.extractall(des_path)
#unzip file
def parse_gnt():
    zip_path = Path('/home/public/yushilin/handwrite/CASIA_ori/gnt/')
    save_dir = '/home/public/yushilin/handwrite/CASIA_ori/HWDB1'  # 目录下均为gnt文件
    zip_files =[zip_file for zip_file in zip_path.iterdir() if str(zip_file).endswith(".zip")]
    extract_zip(zip_files)

    label_list = []
    for zip_file in zip_files:
        for gnt_path in (zip_path / zip_file.stem).iterdir():
            print(gnt_path)
            count = 0
            with open(str(gnt_path), 'rb') as f:
                while f.read(1) != "":
                    f.seek(-1, 1)
                    count += 1
                    try:
                        # 只所以添加try，是因为有时f.read会报错 struct.error: unpack requires a buffer of 4 bytes
                        # 原因尚未找到
                        length_bytes = struct.unpack('<I', f.read(4))[0]

                        tag_code = f.read(2)

                        width = struct.unpack('<H', f.read(2))[0]

                        height = struct.unpack('<H', f.read(2))[0]

                        im = Image.new('RGB', (width, height))
                        img_array = im.load()
                        for x in range(height):
                            for y in range(width):
                                pixel = struct.unpack('<B', f.read(1))[0]
                                img_array[y, x] = (pixel, pixel, pixel)

                        filename = str(count) + '.png'
                        tag_code = tag_code.decode('gbk').strip('\x00')
                        save_path = f'{save_dir}/images/{gnt_path.stem}'
                        if not Path(save_path).exists():
                            Path(save_path).mkdir(parents=True, exist_ok=True)
                        im.save(f'{save_path}/{filename}')

                        label_list.append(f'{gnt_path.stem}/{filename}\t{tag_code}')
                    except:
                        break

    write_txt(f'{save_dir}/gt.txt', label_list)



def read_from_dgrl(dgrl):
    if not os.path.exists(dgrl):
        print('DGRL not exis!')
        return

    dir_name, base_name = os.path.split(dgrl)
    label_dir = dir_name+'_label'
    image_dir = dir_name+'_images'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    print("label_dir,image_dir",label_dir,image_dir)
    with open(dgrl, 'rb') as f:
        # 读取表头尺寸
        header_size = np.fromfile(f, dtype='uint8', count=4)
        header_size = sum([j << (i*8) for i, j in enumerate(header_size)])
        # print(header_size)

        # 读取表头剩下内容，提取 code_length
        header = np.fromfile(f, dtype='uint8', count=header_size-4)
        code_length = sum([j << (i*8) for i, j in enumerate(header[-4:-2])])
        # print(code_length)

        # 读取图像尺寸信息，提取图像中行数量
        image_record = np.fromfile(f, dtype='uint8', count=12)
        height = sum([j << (i*8) for i, j in enumerate(image_record[:4])])
        width = sum([j << (i*8) for i, j in enumerate(image_record[4:8])])
        line_num = sum([j << (i*8) for i, j in enumerate(image_record[8:])])
        print('图像尺寸:')
        print(height, width, line_num)

        # 读取每一行的信息
        for k in range(line_num):
            print(k+1)

            # 读取该行的字符数量
            char_num = np.fromfile(f, dtype='uint8', count=4)
            char_num = sum([j << (i*8) for i, j in enumerate(char_num)])
            print('字符数量:', char_num)

            # 读取该行的标注信息
            label = np.fromfile(f, dtype='uint8', count=code_length*char_num)
            label = [label[i] << (8*(i % code_length))
                     for i in range(code_length*char_num)]
            label = [sum(label[i*code_length:(i+1)*code_length])
                     for i in range(char_num)]
            label = [struct.pack('I', i).decode(
                'gbk', 'ignore')[0] for i in label]
            print('合并前：', label)
            label = ''.join(label)
            # 去掉不可见字符 \x00，这一步不加的话后面保存的内容会出现看不见的问题
            label = ''.join(label.split(b'\x00'.decode()))
            print('合并后：', label)

            # 读取该行的位置和尺寸
            pos_size = np.fromfile(f, dtype='uint8', count=16)
            y = sum([j << (i*8) for i, j in enumerate(pos_size[:4])])
            x = sum([j << (i*8) for i, j in enumerate(pos_size[4:8])])
            h = sum([j << (i*8) for i, j in enumerate(pos_size[8:12])])
            w = sum([j << (i*8) for i, j in enumerate(pos_size[12:])])
            # print(x, y, w, h)

            # 读取该行的图片
            bitmap = np.fromfile(f, dtype='uint8', count=h*w)
            bitmap = np.array(bitmap).reshape(h, w)

            # 保存信息
            label_file = os.path.join(
                label_dir, base_name.replace('.dgrl', '_'+str(k)+'.txt'))
            with open(label_file, 'w') as f1:
                f1.write(label)
            bitmap_file = os.path.join(
                image_dir, base_name.replace('.dgrl', '_'+str(k)+'.jpg'))
            cv.imwrite(bitmap_file, bitmap)


if __name__ == '__main__':
    dgrl_dirs = [zip_file for zip_file in Path('/home/public/yushilin/handwrite/CASIA_ori/dgrl').iterdir() if zip_file.is_dir()]
    #extract_zip(dgrl_dirs)
    for dgrl_dir in dgrl_dirs:
        dgrl_paths = dgrl_dir.iterdir()
        for dgrl_path in tqdm(dgrl_paths):
            read_from_dgrl(dgrl_path)

