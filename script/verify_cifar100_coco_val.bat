cd ..
python verify.py ^
--cuda ^
--load_path "./.mobilenet/project_coco_val_ep20_ly2.pt" ^
--n_layer 2 ^
--log_path "./.log/coco_val_ep20_ly2_cifar100.log"