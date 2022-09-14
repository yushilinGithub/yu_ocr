#export MODEL=/xdfapp/yushilin/formular/result/ft_formular_swin/checkpoint_best.pt
#export MODEL=/home/public/yushilin/formular/harvard/ft_formular_harDict/checkpoint116.pt
#export RESULT_PATH=/xdfapp/yushilin/formular/result/test_swin_10
#export DATA=/xdfapp/yushilin/formular/data/
#export BSZ=46

#export MODEL=/home/public/yushilin/formular/harvard/ft_formular_swin_minilm/checkpoint_best.pt
#export MODEL=/home/public/yushilin/formular/result/okay/ft_formular_swin_taojuan/checkpoint_best.pt
export MODEL=/home/public/yushilin/ocr/model/handwrite/CASIA_64_336_CSwin/checkpoint_best.pt
export RESULT_PATH=/home/public/yushilin/ocr/result/handwrite/CASIA_64_336_CSwin
#export DATA=/home/public/yushilin/formular/harvard/data/
export DATA=/home/public/yushilin/handwrite/CASIA_ori/CASIA
export BSZ=32

#swin_tiny_patch4_window7
CUDA_VISIBLE_DEVICES=0 $(which fairseq-generate)  \
        --data-type HandWrite  --user-dir ./ --task handwrite_recognition --input-size 64-336 \
        --beam 5 --scoring bleu --gen-subset valid --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess FM \
        --decoder-pretrained-url /home/public/yushilin/ocr/model/pretrained/chinese_roberta_L-6_H-512 \
        --skip-invalid-size-inputs-valid-test  \
        --data ${DATA}
