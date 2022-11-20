cd ..
python distill_cos.py ^
--cuda ^
--n_layers 2 ^
--batch_size 16 ^
--epoch 10 ^
--from_ckpt "./.ckpt/coco_val_ep10_ly2_distill_cos.pt" ^
--to_ckpt "./.ckpt/coco_val_ep20_ly2_distill_cos.pt" ^
--save_path "./.mobilenet/project_coco_val_ep20_ly2.pt" ^
--log_path "./.log/coco_val_ep20_ly2_distill_cos.log"