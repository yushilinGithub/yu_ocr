#export MODEL_NAME=ft_formular_harDict_192_384_irpe_newsize
export MODEL_NAME=CASIA_64_336_swinv2
export SAVE_PATH=/home/public/yushilin/ocr/model/handwrite/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}
#export DATA=/home/public/yushilin/formular/harvard/data/
DATA=/home/public/yushilin/handwrite/CASIA_ori/CASIA
#DATA=/home/public/handwritten/data/
mkdir ${LOG_DIR}
export BSZ=72
export valid_BSZ=72
#--input-size 192-768
# CUDA_VISIBLE_DEVICES=1 python $(which fairseq-train) --data-type formular --input-size 192-768 --user-dir ./ --task text_recognition  --arch deit_formular  --seed 1111 --optimizer adam --lr 5e-05 --lr-scheduler inverse_sqrt \
#     --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm  \
#     --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} \
#     --tensorboard-logdir ${LOG_DIR} --max-epoch 300 --patience 20 --ddp-backend legacy_ddp \
#     --num-workers 8 --preprocess FM --update-freq 1 \
#     --decoder-pretrained unilm\
#     --finetune-from-model /home/yushilin/workspace/ocr/unilm/trocr/pretrain/formular_small_harDict.pt --fp16 \
#     --data ${DATA} 


#deit_small_distilled_formular the differences to original one is image data augament, and change input size 384*384 to 192*768 
#deit_formular_irpe  #with irpe registered by Yushilin
#192-768
#224-784
#--decoder-pretrained minilm  
#--decoder-pretrained-url /home/public/yushilin/formular/pretrained/MiniLM-L6-H384-distilled-from-BERT-Large/pytorch_model.bin \
#taojuan --data-type formular 
#mopai --data-type formularSyn 
CUDA_VISIBLE_DEVICES=1 python $(which fairseq-train) --data-type HandWrite --input-size 64-384 --user-dir ./ --task handwrite_recognition  \
            --arch swinv2_tiny_patch4_window8  --seed 1111 --optimizer adam --lr 5e-05 --lr-scheduler inverse_sqrt \
            --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm  --log-interval 10 \
            --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} --tensorboard-logdir ${LOG_DIR} \
            --max-epoch 300 --patience 20 --ddp-backend legacy_ddp --num-workers 4 --preprocess FM --update-freq 1 \
            --skip-invalid-size-inputs-valid-test \
            --adapt-dictionary \
            --decoder-pretrained-url /home/public/yushilin/ocr/model/pretrained/chinese_roberta_L-6_H-512 \
            --encoder-pretrained-url /home/public/yushilin/ocr/model/pretrained/encoder/swinv2_tiny_patch4_window8_256.pth \
            --data ${DATA} 
