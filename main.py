from __future__ import print_function
import sys
import torch.optim as optim
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from STN import Net
from data_loader import load_data
from train_test import train, test
from utils import convert_image_np
import torch
import argparse
import logging
from subprocess import check_output
import os
import datetime

sys.version

print("PyTorch version: ")
print(torch.__version__)
print("CUDA Version: ")
print(torch.version.cuda)
print("cuDNN version is: ")
print(torch.backends.cudnn.version())

parser = argparse.ArgumentParser(description='ALIGNet for adverserial')
# # ------------ General
parser.add_argument('--data_set', type=str, default='CIFAR10',
                    help='Supported options are: MNIST or CIFAR10')
parser.add_argument('--outputDir', default='/home/einav/deepLearningProject/results/', type=str,
                    help='where to save the output images')
parser.add_argument('--seed', default=5829, type=int,
                    help='Manually set RNG seed')
parser.add_argument('--use_cuda', default=True, type=bool,
                    help='Use GPU if available')
# parser.add_argument('--GPU', default=1, type=int, help='Default preferred GPU')
# # ------------- Training
parser.add_argument('--nEpochs', default=2, type=int,
                    help='Number of total epochs to run')
parser.add_argument('--batchSize', default=64, type=int,
                    help='mini-batch size (1 = pure stochastic)')
# # ---------- Model
parser.add_argument('--adversarial_label', default=2, type=int,
                    help='Adversarial target class (0-9)')
parser.add_argument('--source_label', default=0, type=int,
                    help='Adversarial target class (0-9)')
# # ---------- Optimization
parser.add_argument('--LR', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--beta', default=0.1, type=float,
                    help='weight of the L2 loss (source resemble ')
args = parser.parse_args()

# Define what device we are using
print("CUDA Available: ", (args.use_cuda and torch.cuda.is_available()))
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

time_str = datetime.datetime.now().strftime("_%Y_%m_%d__%H_%M")

results_folder = 'from_' + str(args.source_label) + \
                 '_to_' + str(args.adversarial_label) + \
                 time_str + '_beta_' + str(args.beta) + \
                 '_epochs_' + str(args.nEpochs) +  \
                 '_lr_' + str(args.LR)

ips = check_output(['hostname', '--all-ip-addresses'])
if ips == b'132.66.50.93 \n':
    args.is_run_local = True
    print('running local')
    args.outputDir = os.path.join('/home/einavyogev/Documents/deep_learning_project/results/', results_folder)
    plt.ion()  # interactive mode
else:
    args.is_run_local = False
    args.is_run_local = False
    args.outputDir = os.path.join(args.outputDir, results_folder)
    matplotlib.use('Agg')

if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)
    print('created', args.outputDir)

logger = logging.getLogger('STN')
hdlr = logging.FileHandler(os.path.join(args.outputDir, 'adversarial_stn_log.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

# logging.basicConfig(filename=os.path.join(args.outputDir, 'myapp.log'),level=logging.DEBUG)

for arg in vars(args):
    # print(arg, getattr(args, arg))
    logger.info(arg + ': ' + str(getattr(args, arg)))


torch.manual_seed(args.seed)

train_loader, test_loader = load_data(args.batchSize, args.source_label, args.data_set)

model = Net(args.data_set).to(device)

model.load_pretrained_classifier()

optimizer = optim.SGD(model.parameters(), lr=args.LR)
num_epochs = args.nEpochs
train_losses = []
train_mse_losses = []

adversarial_tensor = torch.tensor(args.adversarial_label, requires_grad=False).to(device)

for epoch in range(1, num_epochs + 1):
    train_loss, train_mse_loss = train(model, device, train_loader, optimizer, epoch,  adversarial_tensor, args.beta)
    print('Train total loss:', train_loss, 'Train MSE loss:', train_mse_loss)
    logger.info('[epoch ' + str(epoch) + '] Train loss:' + str(train_loss) + ' Train MSE loss:' + str(train_mse_loss))

    train_mse_losses.append(train_mse_loss)
    train_losses.append(train_loss)
    if epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.5

attack_accuracy = test(model, device, test_loader, adversarial_tensor, args.beta, logger)
epochs = range(1, len(train_losses) + 1)

torch.save(model.state_dict(), os.path.join(args.outputDir, 'model.pth'))

# Visualize the STN transformation on some input batch
# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.
with torch.no_grad():
    # Get a batch of training data
    data = next(iter(test_loader))[0].to(device)

    input_tensor = data.cpu()
    transformed_input_tensor = model.stn(data)[0].cpu()

    in_grid = convert_image_np(
        torchvision.utils.make_grid(input_tensor))

    out_grid = convert_image_np(
        torchvision.utils.make_grid(transformed_input_tensor))

    f, axarr = plt.subplots(1, 2)
    f.canvas.set_window_title('Beta: ' + str(args.beta) + 'Epochs: ' + str(args.nEpochs) + 'LR: ' + str(args.LR))

    axarr[0].imshow(in_grid)
    axarr[0].set_title('Original Images')
    axarr[0].set_yticklabels([])
    axarr[0].set_xticklabels([])

    axarr[1].imshow(out_grid)
    axarr[1].set_title('Transformed Images, Target:' + str(args.adversarial_label))
    axarr[1].set_yticklabels([])
    axarr[1].set_xticklabels([])

    f.suptitle('Targeted Attack Success ' + str(attack_accuracy))

    f.savefig(os.path.join(args.outputDir, 'original_vs_warped_images.png'))

print('num of epochs =' + str(num_epochs))


fig = plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training loss')
plt.legend()
plt.title('Learning curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
fig.savefig(os.path.join(args.outputDir, 'training_loss.png'))


fig = plt.figure(figsize=(10, 6))
plt.plot(train_mse_losses, label='Training MSE loss')
plt.legend()
plt.title('Learning curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
fig.savefig(os.path.join(args.outputDir, 'training_mse_loss.png'))

if args.is_run_local:

    plt.ioff()
    plt.show()
print('done')

