# OKayOCR

## Introduction
original repositary is located at [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr), we make a little modification, added image relative positional encoding and several encoder methods, we added swin transformer and T2T-VIT as encoder, the best model we get is using swin transformer as encoder, it doesn't matter much that change decoder to unilm or several others famous pretrained language model. we changed the image input size and data augmentation from original one, this make the project big progress.

 you can use pretrained moded from trocr, thanks for the original one.
 [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282), Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei, Preprint 2021.

The TrOCR is currently implemented with the fairseq library. We hope to convert the models to the Huggingface format later.



## Installation
~~~bash
conda create -n trocr python=3.7
conda activate trocr
git clone https://github.com/microsoft/unilm.git
cd unilm
cd trocr
pip install pybind11
pip install -r requirements.txt
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" 'git+https://github.com/NVIDIA/apex.git'
~~~

## Fine-tuning and evaluation
~~~bash
bash run_formular.sh
~~~
### Evaluation
~~~bash
bash eval.sh
~~~


## Inference 
```
python pic_inference.py
```

## Citation
If you want to cite TrOCR in your research, please cite the following paper:
``` latex
@misc{li2021trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2021},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree. Portions of the source code are based on the [fairseq](https://github.com/pytorch/fairseq) project. [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information
For help or issues using TrOCR, please submit a GitHub issue.

For other communications related to TrOCR, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).

