You can reproduce Figure 3 experiments by running the corresponding shell scripts.

Folder 'batch_size10' contains scripts with the standard local batch size setting and will produce results in Figure 3 (Left).
Folder 'batch_size400' contains scripts with full local batch size setting and will produce results in Figure 3 (Right).

Each script takes about 8~12 hours on Tesla P100 gpu.

Output files will be saved under "save/objects/improve/" directory.


* Always put the shell script(s) under the same folder as 'federated_main.py' and MAKE SURE the directory "save/objects/improve/" is created under the same folder BEFORE you run the experiments.
