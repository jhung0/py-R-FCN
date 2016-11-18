import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import numpy as np
import subprocess
import pickle
from shutil import copyfile
import os
'''
Trains multiple times (at different learning rates)
'''
base_dir = '/home/ubuntu/py-R-FCN'
ohem = True
if ohem:
	ohem = '_ohem'
else:
	ohem = ''	
prototxt = "models/try1/ResNet-50/rfcn_end2end/solver"+ohem+".prototxt"
weights = "data/rfcn_models/resnet50_rfcn_final.caffemodel"
cfg_file = "experiments/cfgs/rfcn_end2end"+ohem+".yml"
output_dir = os.path.join(base_dir, 'output/rfcn_end2end'+ohem+'/train/')
iterations = 100000
learning_range = [.01, .001, .0001]#.008
for learning in learning_range:		
	print 'learning ', learning
        new_prototxt = "models/try1/ResNet-50/rfcn_end2end/solver"+ohem+"_lr"+str(learning)+".prototxt"
        copyfile(prototxt, new_prototxt)
        with open(new_prototxt, "r") as f:
                lines = f.readlines()
        with open(new_prototxt, "w") as f:
                for line in lines:
                        if 'base_lr' not in line:
                                f.write(line)
                        else:
                                f.write('base_lr: '+str(learning)+"\n")
        prototxt = new_prototxt
	subprocess.call(['python', '/home/ubuntu/py-R-FCN/tools/train_net.py', '--gpu', '0', '--solver', prototxt, '--weights',  weights, '--imdb', 'try1_train', '--cfg', cfg_file, '--iters', str(iterations)])
	'''
	try:
		subprocess.call(['python', '/home/ubuntu/py-faster-rcnn/tools/train_net.py', '--gpu', '0', '--solver', prototxt, '--weights',  weights, '--imdb', 'try1_train', '--cfg', cfg_file, '--iters', str(iterations)])
	except:
		print 'learning rate ' + learning + 'failed'
		continue
	'''
	#save caffemodel as another name so it's not overwritten
	for caffemodel in os.listdir(output_dir):
		if '0000.caffemodel' in caffemodel:
			copyfile(os.path.join(output_dir, caffemodel),os.path.join(output_dir, caffemodel.split('.caffemodel')[0]+'_lr'+str(learning)+'.caffemodel')) 

