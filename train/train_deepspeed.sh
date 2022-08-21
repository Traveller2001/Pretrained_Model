export CUDA_VISIBLE_DEVICES="0,1,2,3"
deepspeed train_deepspeedbs12.py --fp16 1 --deepspeed_config config.json 
