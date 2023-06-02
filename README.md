# MEDIC Trigger Removal

Our code is based the structure from https://github.com/bboylyg/NAD.

We include the backdoor attack training and evaluating.

We test the code on Ubuntu 18.04, Pytorch 1.7.1.

We require the specific dependent of package TrojAI for testing on Trojai model, https://pages.nist.gov/trojai/.

If you don't have that need, please feel free to comment out the trojai import.

# Step 1

Training a backdoor model. 

Use cleanlabel attack as example:

CUDA_VISIBLE_DEVICES=$DEVICE$ python train_badnet.py --inject_portion=0.15 --checkpoint_root="./weight/cleanlabel" --clean_label  --epochs=100 --target_type="cleanLabel"

# Step 2 

Remove Trigger using Medic, clean label attack as an example.

Python main.py --hook  --converge --beta3=0 --beta2=0 --beta1=0 --isample='l2' --epochs={} --lr=0.01  --ratio=0.05 --keepstat --norml2 --hookweight=10  --hook-plane="conv+bn" --imp_temp=5 --s_model="./weight/cleanlabel/WRN-16-1-S-model_best.pth.tar" --t_model=./weight/cleanlabel/WRN-16-1-S-model_best.pth.tar"

