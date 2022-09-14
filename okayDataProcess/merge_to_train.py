import argparse
from pathlib import Path
#parser label data, [image_path]\t[label]
def get_TAL(args):
    Dir = Path(args.TAL)
    train_path = Dir/"train_64"
    test_path = Dir/"test_64"
    

    image_paths = train_path.glob("*.jpg")
    for image_path in image_paths:
        text_path = image_path.with_suffix(".txt")
        if text_path.exists():
            text = text_path.read_text().strip()
            
def get_CASIZ(args):
    pass
def get_Simulated(args):
    pass


def main(args):
    TAL = get_TAL(args)
    CASIA = get_CASIZ(args)
    Simulated = get_Simulated(args)    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--TAL",type=str,default="/home/public/yushilin/handwrite/TAL/")
    parser.add_argument("--CASIA",type=str,default="/home/public/yushilin/handwrite/CASIA_ori/CASIA")
    parser.add_argument("--Simulated",type=str,default="/home/public/yushilin/handwrite/simulated")
    args = parser.parse_args()
    main(args)