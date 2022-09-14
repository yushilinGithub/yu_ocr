from pathlib import Path 
import shutil
from tkinter import image_names
def main():
    output_path = Path("/home/public/yushilin/handwrite/CASIA_ori/CASIA")
    if not output_path.exists():
        output_path.mkdir()
        (output_path/"images").mkdir()
    image_id = 0
    data_path = Path("/home/public/yushilin/handwrite/CASIA_ori/HWDB2")
    train_labels = []
    test_labels = []
    for dir in data_path.iterdir():
        if str(dir).endswith("images"):
            txt_dir = Path(str(dir).replace("images","label"))
            for image_path in dir.iterdir():
                txt = (txt_dir/(image_path.stem+".txt")).read_text()
                image_name = "%07d.jpg"%image_id
                output_image = output_path/"images"/image_name
                shutil.copy(image_path,output_image)
                if str(dir).endswith("Train_images"):
                    train_labels.append(image_name+"\t"+txt)
                elif str(dir).endswith("Test_images"):
                    test_labels.append(image_name+"\t"+txt)
                image_id+=1
    (output_path/"gt_train.txt").write_text("\n".join(train_labels))
    (output_path/"gt_test.txt").write_text("\n".join(test_labels))
if __name__=="__main__":
    main()