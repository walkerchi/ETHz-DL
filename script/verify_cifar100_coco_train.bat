cd ..
python verify.py ^
--cuda ^
--load_path "./.mobilenet/project_coco_train_ep1_ly2.pt" ^
--n_layer 2 ^
--log_path "./.log/coco_train_ep1_ly2_cifar100.log"