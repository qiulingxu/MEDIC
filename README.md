# MEDIC Trigger Removal

MEDIC is a retraining based backdoor removal method. It uses layer-wise cloning with importance. It exhibits strong advantage over strong backdoors, e.g. the adversarial trained ones with data augmentation.

We compare our method over 5 baselines and 9 types of backdoor attack. The datasets includes CIFAR-10 and large-scale KIWICITY.

Here is an example of results, where all methods are aligned to roughly similar accuracy.

![Comparison of ASR](./ASR.png)

# Supports
    Our scripts support training and removal on the baselines we mentioned except TrojAI models including filter and polygon, which requires some time of configuration.

# Backdoor Models

> TODO: Upload and release the model used in the paper.

# Commands

## Step 1

Training a backdoor model. 

Use cleanlabel attack as example:

CUDA_VISIBLE_DEVICES=$DEVICE$ python train_badnet.py --inject_portion=0.15 --checkpoint_root="./weight/cleanlabel" --clean_label  --epochs=100 --target_type="cleanLabel"

## Step 2 

Remove Trigger using Medic, clean label attack as an example.

Python main.py --hook  --converge --beta3=0 --beta2=0 --beta1=0 --isample='l2' --epochs={} --lr=0.01  --ratio=0.05 --keepstat --norml2 --hookweight=10  --hook-plane="conv+bn" --imp_temp=5 --s_model="./weight/cleanlabel/WRN-16-1-S-model_best.pth.tar" --t_model=./weight/cleanlabel/WRN-16-1-S-model_best.pth.tar"

# Dependency

Our code is based the structure from https://github.com/bboylyg/NAD.

We include the backdoor attack training and evaluating.

We test the code on Ubuntu 18.04, Pytorch 1.7.1.

We require the specific dependent of package TrojAI for testing on Trojai model, https://pages.nist.gov/trojai/.

If you don't have that need, please feel free to comment out the trojai import.

# Paper and citation

Paper can be found [here](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_MEDIC_Remove_Model_Backdoors_via_Importance_Driven_Cloning_CVPR_2023_paper.html).

You are welcome to cite the paper if you find our work useful.
```
@InProceedings{Xu_2023_CVPR,
    author    = {Xu, Qiuling and Tao, Guanhong and Honorio, Jean and Liu, Yingqi and An, Shengwei and Shen, Guangyu and Cheng, Siyuan and Zhang, Xiangyu},
    title     = {MEDIC: Remove Model Backdoors via Importance Driven Cloning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20485-20494}
}
```