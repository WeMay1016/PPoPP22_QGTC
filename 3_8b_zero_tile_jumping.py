#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

hidden = 		[16] # [16, 32, 64, 128, 256]
num_layers = 	[1]
partitions = 	[1500] # 1500, 3000, 4500, 6000, 7500, 9000]

dataset = [
		( 'artist'                 	 , 100	  , 12),
		( 'soc-BlogCatalog'	     	 , 128	  , 39),    
]

for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			for p in partitions:
				command = "python cluster_gcn.py \
    						--gpu 0 \
							--dataset {} \
       						--dim {} \
                            --n-hidden {} \
                            --n-classes {} \
							--psize {}\
							--use_QGTC \
							--zerotile_jump \
           					--n-epochs 1 \
       						".\
							format(data, d, hid, c, p)		
				os.system(command)
				print()

os.system("python cluster_gcn.py --gpu 0 --dataset ppi --use_QGTC --zerotile_jump --zerotile_jump --n-epochs 1")
print()
os.system("python cluster_gcn.py --gpu 0 --dataset ogbn-arxiv --use_QGTC --zerotile_jump --n-epochs 1")
print()
# os.system("python cluster_gcn.py --gpu 0 --dataset ogbn-products --use_QGTC")