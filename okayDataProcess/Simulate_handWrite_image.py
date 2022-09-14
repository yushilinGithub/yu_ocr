import argparse
from ast import Or, parse
from curses.ascii import isspace
from tkinter.messagebox import NO
from tkinter.tix import Tree
import cv2
import numpy as np
import os
from collections import OrderedDict
import random
from multiprocessing import Pool
from tqdm import tqdm
table = {ord(f):ord(t) for f,t in zip(
     u'，！？：；＝．﹣－＋＜＞【】（）％＃＠＆１２３４５６７８９０',
     u',!?:;=.--+<>[]()%#@&1234567890')}
upper_punctuations = ["'",'"',"“","”","‘","’"]
lowwer_punctuations = [",",".","。","，","?","!",":",";","、","…"]
english_charactors = list("qwertyuiopasdfghjklzxcvbnm")
center_english_charactor = list("weruioaszxcvnm")
upper_english_charactor = list("tdfhklb")
lowwer_english_charactor = list("qypgj")
def is_there_chinese(line):
    for ch in line:
        if u"\u4e00"<=ch<=u"\u9fff":
            return True
    return False

def get_pos(top_r,min_h,im_h,im_w,width_variation,charactor):
    x1 = top_r+width_variation
    x2 = x1+im_w
    
    if charactor in upper_punctuations:
        mean_y = (min_h-im_h)*0.25#center
    elif charactor in lowwer_punctuations:
        mean_y = (min_h-im_h)*0.8
    elif charactor in upper_english_charactor:
        mean_y = max(((min_h-im_h)//2)-im_h*0.2,0.1)
    elif charactor in lowwer_english_charactor:
        mean_y = min(((min_h-im_h)//2)+im_h*0.2,(min_h-im_h))
    else:
        mean_y = (min_h-im_h)//2 #center
        
    y1 = min(np.abs(int(np.random.normal(mean_y,mean_y//4))),min_h-im_h)
    
    y2 = y1+im_h
    return x1,x2,y1,y2
def generate_subprocess(arguments):
    args,sub_image_paths,line,line_id = arguments
    image_dir = os.path.join(args.src_image_path,"images")

    output_image_name = "%07d.jpg"%line_id
    out_image_path = os.path.join(args.output_dir,"images",output_image_name)


    line = line.translate(table)
    images = []
    h_w = []
    width_variations = []
    for cha_id,character in enumerate(line):
        image_name = sub_image_paths[cha_id]
        if image_name:    
            image_path = os.path.join(image_dir,image_name)
            image = cv2.imread(image_path)
            if not isinstance(image,np.ndarray):
                print(image_path)
                images.append(None)
            else:
                images.append(image)
                h_w.append([image.shape[0],image.shape[1]])
        else:
            images.append(None)
        wv_base = round(h_w[-1][1]*0.25) if h_w else 15
        width_variation = random.randint(0,wv_base) if character not in lowwer_punctuations else random.randint(args.margin-5,args.margin+5)
        if not character:
            width_variation=random.randint(wv_base,wv_base+8) 
        width_variations.append(width_variation)
        
    character_loc = np.max(h_w,axis=0)[0]+10
    min_h = character_loc+args.margin*2
    
    min_w = np.sum(h_w,axis=0)[1]+np.sum(width_variations)+args.margin*2

    base_image = np.zeros((min_h,min_w,3))+255
    top_r = args.margin
    for image_id,image in enumerate(images):

        if isinstance(image,np.ndarray):
            im_h,im_w = image.shape[:2]
            wv = width_variations[image_id]
            x1,x2,y1,y2 = get_pos(top_r,character_loc,im_h,im_w,wv,line[image_id])
            base_image[y1:y2,x1:x2] = image
        else:
            x2 = top_r+width_variations[image_id]
        top_r = x2
    cv2.imwrite(out_image_path,base_image)

def generate_image(args,gt_dict,lines):
    image_path = os.path.join(args.src_image_path,"images")
    i = 0
    output_label = []
    for line in lines:
        image_name = "%07d.jpg"%i
        out_image_path = os.path.join(args.output_dir,"images",image_name)
        print(out_image_path)

        new_line = ""
        images = []
        h_w = []
        width_variations = []
        line = line.translate(table)
        if len(line)==0 or not is_there_chinese(line): #没有字符或者没有中文字符的
            continue 
        person = None
        for character in line:
            if character in gt_dict.keys():
                if args.human_source == "sgs":
                    if person==None:
                        person = np.random.choice(list(gt_dict[character].keys()))

                    if person in gt_dict[character].keys():
                        paths = gt_dict[character][person]
                        if paths:
                            image_p = np.random.choice(paths)
                            image = cv2.imread(os.path.join(image_path,image_p))
                        else:
                            path = np.random.choice(list(gt_dict[character].values()))
                            image =  cv2.imread(os.path.join(image_path,path))
                    else:
                        paths = [v[0] for v in gt_dict[character].values()]
                        path = np.random.choice(paths)
                        image =  cv2.imread(os.path.join(image_path,path))
                elif args.human_source == "random":
                    image =  cv2.imread(os.path.join(image_path,np.random.choice(list(gt_dict[character].values()))))
                    
                if isinstance(image,np.ndarray):
                    images.append(image)
                    h_w.append([image.shape[0],image.shape[1]])
                    
                    width_variations.append(random.randint(0,round(image.shape[1]*0.25)))
                    new_line += character
            elif character in [" ","_"]:
                images.append(None)
                white_space = h_w[-1][1] if h_w else args.margin
                width_variations.append(random.randint(white_space-5, white_space+5))
                new_line += " "
            else:
                images.append("unknow")
                width_variations.append(random.randint(args.margin-5,args.margin+5))
                new_line += " "
        if len(h_w)==0:
            continue
        character_loc = np.max(h_w,axis=0)[0]+10
        min_h = character_loc+args.margin*2
        
        min_w = np.sum(h_w,axis=0)[1]+np.sum(width_variations)+args.margin*2

        base_image = np.zeros((min_h,min_w,3))+255
        top_r = args.margin
        for image_id,image in enumerate(images):

            if isinstance(image,np.ndarray):
                im_h,im_w = image.shape[:2]
                wv = width_variations[image_id]
                x1,x2,y1,y2 = get_pos(top_r,character_loc,im_h,im_w,wv,new_line[image_id])
                base_image[y1:y2,x1:x2] = image
            else:
                x2 = top_r+width_variations[image_id]
            top_r = x2
        image_name = "%07d.jpg"%i
        out_image_path = os.path.join(args.output_dir,"images",image_name)
        output_label.append(os.path.join("images",image_name)+"\t"+new_line)
        cv2.imwrite(out_image_path,base_image)
        i = i+1
        if i>args.num_images:
            break
    with open(os.path.join(args.output_dir,"label.txt"),"w") as f:
        f.write("\n".join(output_label))
def get_line(gt_dict,lines):
    image_path = os.path.join(args.src_image_path,"images")
    SUB_IMAGE_PATH = []
    line_ids = []
    new_lines = []
    for line in lines:
        new_line = ""
        sub_image_paths = []
        line = line.translate(table)
        if len(line)==0 or not is_there_chinese(line): #没有字符或者没有中文字符的
            continue 
        person = None
        for character in line:
            if character in gt_dict.keys():
                if person==None:
                    person = np.random.choice(list(gt_dict[character].keys())) #randomly choice a writer
                    paths = gt_dict[character][person]
                    if paths:
                        image_p = np.random.choice(paths)
                        sub_image_paths.append(image_p)
                    else:
                        path = np.random.choice(list(gt_dict[character].values()))
                        sub_image_paths.append(path)
                        
                elif person in gt_dict[character].keys():
                    paths = gt_dict[character][person]
                    if paths:
                        image_p = np.random.choice(paths)
                        sub_image_paths.append(image_p)
                    else:
                        paths = [v[0] for v in gt_dict[character].values()]
                        path = paths[np.random.randint(0,len(paths))]
                        sub_image_paths.append(path)
                else:
                    paths = [v[0] for v in gt_dict[character].values()]
                    path = paths[np.random.randint(0,len(paths))]
                    sub_image_paths.append(path)
                new_line += character
            else:
                new_line += " "
                sub_image_paths.append(None)
        if not new_line.isspace():
            
            SUB_IMAGE_PATH.append(sub_image_paths)
            line_ids.append(len(new_lines))
            new_lines.append(new_line)
        
    return SUB_IMAGE_PATH,new_lines,line_ids
def get_gt(args):
    with open(os.path.join(args.src_image_path,"gt.txt"),"r") as f:
        gt = f.read().split("\n")
    gt_dict = OrderedDict()
    for g in gt:
        if g:
            image_path,image_content = g.split("\t")
            if not os.path.exists(os.path.join(args.src_image_path,"images",image_path)):
                print(image_path)
                continue
            
            if image_content not in gt_dict.keys():
                gt_dict[image_content] = {os.path.dirname(image_path):[image_path]}
            else:
                path_dirname = os.path.dirname(image_path)
                if path_dirname not in gt_dict[image_content].keys():
                    gt_dict[image_content][path_dirname] = [image_path]
                else:
                    gt_dict[image_content][path_dirname].append(image_path)
    return gt_dict
def main(args):
    if not os.path.exists(args.text_path):
        raise "%s not exist"%args.text_path
    if not os.path.isdir(args.src_image_path):
        raise "%s must be a directory"%args.src_image_path
    with open(args.text_path,"r") as f:
        lines = f.read().split("\n")[:300000]
    gt_dict = get_gt(args)
    sub_image_paths,line,line_id = get_line(gt_dict,lines)

    p = Pool(args.num_process)
    for _ in tqdm(
        p.imap_unordered(generate_subprocess,
                         zip([args]*args.num_images,
                             sub_image_paths[:args.num_images],
                             line[:args.num_images],
                             line_id[:args.num_images])),
        total=args.num_images,
    ):
        pass
    p.terminate()
    
    output_label = []
    for id,line in zip(line_id[:args.num_images],line[:args.num_images]):
        image_name = "%07d.jpg"%id
        output_label.append(os.path.join("images",image_name)+"\t"+line)

    with open(os.path.join(args.output_dir,"label.txt"),"w") as f:
        f.write("\n".join(output_label))
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path",type=str,default="/home/public/yushilin/handwrite/okay_question/okay_queston_0_800000.txt")
    parser.add_argument("--src_image_path",type=str,default="/home/public/yushilin/handwrite/CASIA_ori/HWDB1/")
    parser.add_argument("--output_dir",type=str,default="/home/public/yushilin/handwrite/simulated")
    parser.add_argument("--margin",type=int,default=10)
    parser.add_argument("--human_source",type = str,default="sgs",help="--sgs: single source, --random: random source")
    parser.add_argument("--space_length",type=int,default=15)
    parser.add_argument("--num_images",type=int,default=200000)
    parser.add_argument("--num_process",type=int,default=5)
    
    args = parser.parse_args()
    main(args)