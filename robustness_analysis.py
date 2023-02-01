import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from models import ResNet18
from compute_pgd_map import computePGDMap
from fourier_heatmap import compute_fourier_heat_map


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True,
                    type=str,help = 'path to checkpoint file')
parser.add_argument('--norm_fourierHM', required=True,
                    type=float, help = 'norm of the perturbation for the fourier heatmap')
args = parser.parse_args()


seed=0
torch.manual_seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

modelPath = args.checkpoint_path

net = ResNet18()
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load(modelPath)["net"])

transform = transforms.Compose([
    transforms.ToTensor(),
])


batch_size = 1000
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=16)


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
net.eval()
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

perturbation, adv_accuracy = computePGDMap(net,testloader)
print(f'Adverasrial Accuracy of the network on the 10000 test images: {100 * adv_accuracy} %')
np.save(args.checkpoint_path+'_perturbationMap.npy', perturbation)

fourier_heatMap = compute_fourier_heat_map(net, test_loader=testloader, norm=args.norm_fourierHM)
np.save(args.checkpoint_path+'_fourier_heatMap_norm_{}.npy'.format(args.norm_fourierHM), fourier_heatMap)