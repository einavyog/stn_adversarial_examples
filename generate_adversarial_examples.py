from __future__ import print_function
import sys

import torchvision
import matplotlib
import matplotlib.pyplot as plt
from STN import Net
from data_loader import load_data
from utils import convert_image_np
import torch
import argparse
import logging
from subprocess import check_output
import os
import csv
import imageio

sys.version

print("PyTorch version: ")
print(torch.__version__)
print("CUDA Version: ")
print(torch.version.cuda)
print("cuDNN version is: ")
print(torch.backends.cudnn.version())

parser = argparse.ArgumentParser(description='ALIGNet for adverserial')
parser.add_argument('--seed', default=5829, type=int, help='Manually set RNG seed')
parser.add_argument('--use_cuda', default=True, type=bool, help='Use GPU if available')
parser.add_argument('--model', default='/home/einavyogev/Documents/deep_learning_project/results/'
                                       '2019_02_10__14_58_beta_0.1_epochs_40_from_0_to_2/model.pth',
                    type=str, help='Path to model')
parser.add_argument('--batchSize', default=64, type=int, help='mini-batch size (1 = pure stochastic)')
parser.add_argument('--adversarial_label', default=2, type=int, help='Adversarial target class (0-9)')
parser.add_argument('--source_label', default=0, type=int, help='Adversarial target class (0-9)')
args = parser.parse_args()

# Define what device we are using
print("CUDA Available: ", (args.use_cuda and torch.cuda.is_available()))
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

args.is_run_local = False
ips = check_output(['hostname', '--all-ip-addresses'])
if ips == b'132.66.50.93 \n':
    args.is_run_local = True
    print('running local')
    plt.ion()  # interactive mode
else:
    matplotlib.use('Agg')

results_folder = os.path.dirname(args.model)
args.outputDir = results_folder

if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)
    print('created', args.outputDir)

logger = logging.getLogger('STN')
hdlr = logging.FileHandler(os.path.join(args.outputDir, 'generate_adversarial_examples.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

logging.basicConfig(filename=os.path.join(args.outputDir, 'myapp.log'),level=logging.DEBUG)

for arg in vars(args):
    # print(arg, getattr(args, arg))
    logger.info(arg + ': ' + str(getattr(args, arg)))

torch.manual_seed(args.seed)

__, test_loader = load_data(args.batchSize, args.source_label)

model = Net().to(device)

model.load_state_dict(torch.load(args.model), strict=True)
print(model)

adversarial_mnist_csv = []
with torch.no_grad():

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)

        transformed_input_tensor = model.stn(data)[0].cpu()

        transformed_input_tensor = torch.squeeze(transformed_input_tensor)

        for im_index in range(transformed_input_tensor.shape[0]):
            # im_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor[im_index]))
            im_grid = transformed_input_tensor[im_index]
            file_name = os.path.join(args.outputDir, 'adversIm_' + str(batch_idx) + '_' + str(im_index) + '_.png')
            adversarial_mnist_csv.append(file_name)
            imageio.imwrite(file_name, im_grid)

with open(os.path.join(args.outputDir, 'adversarial_mnist_csv.csv''adversarial_mnist_csv.csv'), "w") as output:
    fieldnames = ['Filename', 'OrigLabel', 'AdvLabel']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    for val in adversarial_mnist_csv:
        writer.writerows([{'Filename': val, 'OrigLabel': args.source_label, 'AdvLabel': args.adversarial_label}])

print('done')

