cd ..
python distill_cos.py ^
--cuda ^
--n_layers 2 ^
--batch_size 16 ^
--epoch 1 ^
--to_ckpt "./.ckpt/coco_train_ep1_ly2_distill_cos.pt" ^
--save_path "./.mobilenet/project_coco_train_ep1_ly2.pt" ^
--log_path "./.log/coco_train_ep1_ly2_distill_cos.log" ^
--ann_path "./.coco/annotations/captions_train2017.json" ^
--image_path "./.coco/train2017/train2017"