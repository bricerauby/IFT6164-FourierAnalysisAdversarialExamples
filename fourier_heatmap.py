import numpy as np
import scipy.fft
import tqdm
import torch

def compute_basis(h,w):
    """ Function that compute the fourier basis of U_i_j as described in [1] 

    ref : 
    [1] D. Yin, R. Gontijo Lopes, J. Shlens, E. D. Cubuk, and J. Gilmer, 
    “A Fourier Perspective on Model Robustness in Computer Vision,” 
    in Advances in Neural Information Processing Systems, 2019, 
    vol. 32. Accessed: Jan. 24, 2023. [Online]. 
    Available: https://proceedings.neurips.cc/paper/2019/hash/b05b57f6add810d3b7490866d74c0053-Abstract.html
    """ 
    fft_images = {}
    perturbation_basis = {}
    for i in range(0,h//2):
        for j in range(-w//2,w//2):
            image_key = "{},{}".format(i,j)
            fft_images[image_key]= np.zeros((h,w))
            fft_images[image_key][int(np.ceil((h-1)/2 + i)), int(np.ceil((w-1)/2 + j))] = 1
            fft_images[image_key][int(np.floor((h-1)/2 - i)), int(np.floor((w-1)/2 - j))]= 1
            image = scipy.fft.ifft2(scipy.fft.ifftshift(fft_images[image_key], axes=(-2,-1)))
            image = image/ np.sqrt(np.sum(image * np.conjugate(image)))      
            perturbation_basis[image_key] = image.real.copy()
            assert (abs((np.sum(perturbation_basis[image_key].imag / (perturbation_basis[image_key].real+1e-10)))) < 1e-5), image_key + ' is not real'
    return perturbation_basis

def compute_fourier_heat_map(net, test_loader, norm=4, device=None):
    """Function computing the fourier heatmap as described in [1]
    
        ref : 
        [1] D. Yin, R. Gontijo Lopes, J. Shlens, E. D. Cubuk, and J. Gilmer, 
        “A Fourier Perspective on Model Robustness in Computer Vision,” 
        in Advances in Neural Information Processing Systems, 2019, 
        vol. 32. Accessed: Jan. 24, 2023. [Online]. 
        Available: https://proceedings.neurips.cc/paper/2019/hash/b05b57f6add810d3b7490866d74c0053-Abstract.html
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    h,w = test_loader.dataset.__getitem__(0)[0].shape[-2:]

    perturbation_basis = compute_basis(h,w)
    net = net.eval()
    fourier_heat_map = np.zeros((h,w))
    for i in tqdm.tqdm(range(0,h//2)):
        for j in range(-w//2,w//2):
            image_key = "{},{}".format(i,j)
            correct_perturbated = 0
            total = 0
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                r = np.random.choice([-1,1],images.shape[0:2]+(1,1))
                perturbation = torch.from_numpy(norm * r * perturbation_basis[image_key].reshape((1,1)+images.shape[2:])).float().to(device)
                images = images + perturbation
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct_perturbated += (predicted == labels).sum().item()
                total += labels.size(0)
            fourier_heat_map[int(np.ceil((h-1)/2 + i)), int(np.ceil((w-1)/2 + j))] = 1 - correct_perturbated/ total
            fourier_heat_map[int(np.floor((h-1)/2 - i)), int(np.floor((w-1)/2 - j))] = 1 - correct_perturbated/ total
    return fourier_heat_map
    
if __name__=='__main__':
    basis = compute_basis(32,32)
