import argparse
import os
import pandas as pd
from dautils import blue_score
def main(args):
    assert os.path.exists(args.src_path),"{} not exists".format(args.src_path)
    file = open(args.src_path,"r")
    data = []
    l_j = {}
    for i,f in enumerate(file):
        
        line = f.split('\t')
        if line[0].startswith('T'):
            l_j['T'] = line[1].strip()
            l_j['line'] = line[0].split("-")[1]
        elif line[0].startswith('H'):
            l_j['H'] = line[2].strip()
        elif line[0].startswith('D'):
            l_j['D'] = line[2].strip()
        
        if len(l_j)==4:
            s=blue_score.compute_bleu([[l_j['T']]],[l_j['D']])
            l_j['s'] = s[0]
            data.append(l_j)
            l_j={}
    df = pd.DataFrame(data)
    df = df.sort_values(by = 's')
    df.to_csv(args.output)
    file.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path",type=str,default="/home/public/yushilin/formular/results/mopai_chinese_support/generate-valid.csv")
    parser.add_argument("--output",type=str,default="/home/public/yushilin/formular/results/mopai_chinese_support/data_observation.lst")
    args = parser.parse_args()
    main(args)