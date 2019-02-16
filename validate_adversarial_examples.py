from __future__ import print_function
import sys
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from STN import Net
from utils import convert_image_np
import torch
import argparse
import logging
from subprocess import check_output
import os
from data_loader import load_adv_data


# # A simple test procedure to measure STN the performances on MNIST.
def validate(model, device, test_loader, adversarial_target, beta, logger):
    with torch.no_grad():
        model.eval()
        correct = 0

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output_class = model.classifier(data)
            pred = output_class.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()

        print('\nTest set:Targeted Attack Success: {}/{} ({:.0f}%)'
              .format(correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

        logger.info('Test set: Targeted Attack Success: {}/{} ({:.0f}%)'
                    .format(correct, len(test_loader.dataset),
                            100. * correct / len(test_loader.dataset)))


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
parser.add_argument('--beta', default=0.1, type=float, help='weight of the L2 loss (source resemble ')

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
csv_path = os.path.join(results_folder, 'adversarial_mnist_csv.csv')
args.outputDir = results_folder

if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)
    print('created', args.outputDir)

logger = logging.getLogger('STN')
hdlr = logging.FileHandler(os.path.join(args.outputDir, 'validate_adversarial_examples.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

for arg in vars(args):
    # print(arg, getattr(args, arg))
    logger.info(arg + ': ' + str(getattr(args, arg)))

torch.manual_seed(args.seed)

# load AdvMNIST dataset
plt.ion()
validation_set_loader = load_adv_data(args.outputDir, batch_size=args.batchSize)

model = Net().to(device)
model.load_state_dict(torch.load(args.model), strict=True)
print(model)

adversarial_tensor = torch.tensor(args.adversarial_label, requires_grad=False).to(device)
validate(model, device, validation_set_loader, adversarial_tensor, args.beta, logger)

torch.save(model.state_dict(), os.path.join(args.outputDir, 'model.pth'))

# Visualize the STN transformation on some input batch
# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.
with torch.no_grad():
    # Get a batch of training data
    data = next(iter(validation_set_loader))[0].to(device)

    input_tensor = data.cpu()

    in_grid = convert_image_np(
        torchvision.utils.make_grid(input_tensor))

    f, axarr = plt.subplots(1, 1)
    f.canvas.set_window_title('Beta: ' + str(args.beta))

    axarr.imshow(in_grid)
    axarr.set_title('Dataset Images')

    f.savefig(os.path.join(args.outputDir, 'validation_original_vs_warped_images.png'))

if args.is_run_local:

    plt.ioff()
    plt.show()
print('done')


