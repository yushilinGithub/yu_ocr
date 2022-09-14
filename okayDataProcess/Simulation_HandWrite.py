import json
from tokenize import group
from unittest import result
import pandas as pd
import os
import argparse 
from pathlib import Path
from itertools import chain
import glob
import re
min_text_length = 15
max_text_length = 30

def get_text(text_list,json_data):
    if isinstance(json_data,dict):
        if "type" in json_data.keys():
            if json_data["type"] == "text": 
                #return json_data["value"]
                text_list.append(json_data["value"])
        else:
            for key,value in json_data.items():
                get_text(text_list,value)
    elif isinstance(json_data,list):
        for instance in json_data:
            get_text(text_list,instance)
    elif isinstance(json_data,str):
        text_list.append(json_data)
    else:
        pass
def load_curpus(args,path):
    corpus_dataFrame = pd.read_csv(path,on_bad_lines='skip')
    question_type = pd.read_csv(os.path.join(args.data_path,"entity_question_type.csv")).set_index("type_id")["name"].to_dict()
    text = []
    for index,line in corpus_dataFrame.iterrows():
        json_data = eval(line["json_data"])
        question_type_id = line["question_type_id"]
        question_text = []
        get_text(question_text,json_data)
        text.extend(question_text)
    return text
def cut_line(cutted,line):
    if len(line)>min_text_length:
        left = re.split("[。.，,?？!！;；]",line[min_text_length:],1)
        if len(left)==2:
            left,right = left
            if len(left)>max_text_length-min_text_length:
                left,right = line[:max_text_length],line[max_text_length:]
                cutted.append(left)
            else:
                cutted.append(line[:min_text_length]+left)
        else:
            left,right = line[:max_text_length],line[max_text_length:]
            cutted.append(left)

        if len(right)>min_text_length:
            cut_line(cutted,right)
        else:
            cutted.append(right)
    else:
        cutted.append(line)        
def cut_text(output):
    result = []
    for line in output:
        cutted = []
        cut_line(cutted,line)
        result.extend(cutted)
    return result
def main(args):
    texts = []
    for path in glob.glob(args.data_path+"/"+"*0000.csv"):
        print(path)
        texts.extend(load_curpus(args,path))
  
    texts = set(texts)
    output = list(texts)
    output = cut_text(output)
    output = set(output)
    output = list(output)
    if not output[0]:
        output = output[1:]
    output_path = Path(args.output_path)/"okay_queston_0_800000.txt"
    output_path.write_text("\n".join(output))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str,default="/home/public/yushilin/handwrite/okay_question/")
    parser.add_argument("--output_path",type=str,default="/home/public/yushilin/handwrite/okay_question/")
    args = parser.parse_args()
    main(args)