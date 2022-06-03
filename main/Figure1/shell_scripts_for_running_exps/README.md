You can run iid/noniid experiments either by running "run_iid_all_seeds.sh"/ "run_noniid_all_seeds.sh", which takes about 50~75 hours on a Tesla P100 gpu.

Or, by manually running mutliple scripts simultaneously (if you have multiple gpu resources) each with a specific seed (strongly recommended to set seeds to be 1,2,3,4,5 for later-on plotting purpose), need to modify the corresponding entry in "run_iid_specific_seed_example.sh"/"run_noniid_specific_seed_example.sh". This takes about 10~15 hours on Tesla P100 gpus.


* Always put the shell script(s) under the same folder as 'federated_main.py' and MAKE SURE the directory "save/objects/mgda/" is created under the same folder BEFORE you run the experiments.