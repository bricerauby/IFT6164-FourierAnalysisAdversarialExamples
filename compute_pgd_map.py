import torch
import torch.nn as nn

def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40, device=None):
    """ Compute a projected gradient descent attack for model on the images as described in [1]. 
    Function from https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb under MIT License

    Ref:
    [1] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, 
    “Towards Deep Learning Models Resistant to Adversarial Attacks,” 
    presented at the International Conference on Learning Representations, 2018.

    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

def computePGDMap(net, testloader, pdg_params={"eps":8/255, "alpha":2/255, "iters":7}, device=None):
    """
    Compute the average fourier spectrum for PGD perturbations over a dataset given a model as described for fig.7 in [1]

    ref:
    [1] D. Yin, R. Gontijo Lopes, J. Shlens, E. D. Cubuk, and J. Gilmer,
     “A Fourier Perspective on Model Robustness in Computer Vision,” 
     in Advances in Neural Information Processing Systems, 2019.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    correct = 0
    correct_perturbated = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    net.eval()
    perturbation_adv = None
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs_perturbated = pgd_attack(net, inputs, labels, **pdg_params)
        # calculate outputs by running images through the network
        outputs_perturbated = net(inputs_perturbated)
        _, predicted_perturbated = torch.max(outputs_perturbated.data, 1)


        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        if perturbation_adv is None:
            perturbation_adv = ((predicted_perturbated != labels).float().view(-1,1,1,1) * torch.abs(torch.fft.fftshift(torch.fft.fft2(inputs_perturbated - inputs), dim=(-2,-1)))).sum(dim=(0,1))
        else : 
            perturbation_adv += ((predicted_perturbated != labels).float().view(-1,1,1,1) * torch.abs(torch.fft.fftshift(torch.fft.fft2(inputs_perturbated - inputs), dim=(-2,-1)))).sum(dim=(0,1))
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_perturbated += (predicted_perturbated == labels).sum().item()
        correct += (predicted == labels).sum().item()
    pertubated_acc = correct_perturbated/total
    return perturbation_adv.cpu().detach().numpy(), pertubated_acc