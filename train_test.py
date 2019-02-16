import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch

LOSS_ON_AFFINE_MAT = True


def calc_loss(out_warp_im, input_im, out_class, target_class, beta, affine_mat, device):
    MSE = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    if LOSS_ON_AFFINE_MAT:
        identity_affine = torch.eye(2, 3).unsqueeze(0).repeat(affine_mat.shape[0], 1, 1)
        identity_affine = identity_affine.to(device)
        # loss = MSE(affine_mat, identity_affine) + beta*F.nll_loss(out_class, target_class)
        loss = beta*criterion(out_class, target_class)
        mse_loss = MSE(affine_mat, identity_affine)

    else:
        loss = MSE(out_warp_im, torch.squeeze(input_im)) + beta*F.nll_loss(out_class, target_class)
        mse_loss = MSE(out_warp_im, torch.squeeze(input_im))

    return loss, mse_loss


def train(model, device, train_loader, optimizer, epoch, adversarial_target, beta):
    model.train()
    MSE = nn.MSELoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        adversarial_class = torch.squeeze(adversarial_target.expand(1, data.size()[0]))

        optimizer.zero_grad()

        output_warp, output_class, affine_mat = model(data)

        # out_grid_list = []
        # for i in range(0, output_warp.size()[0]):
        #     #TODO: Add cliping?
        #     out_grid_list.append(torchvision.utils.make_grid(output_warp[i,0,:,:])[None, 0,:,:])
        #
        # out_grid = torch.cat(out_grid_list)

        loss, mse_loss = calc_loss(output_warp, torch.squeeze(data), output_class, adversarial_class, beta, affine_mat, device)

        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return loss.item(), mse_loss.item()


# # A simple test procedure to measure STN the performances on MNIST.
def test(model, device, test_loader, adversarial_target, beta, logger):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        mse_loss = 0
        correct = 0
        wrong = 0
        correct_original = 0
        wrong_original = 0
        MSE = nn.MSELoss()
        targets = []

        for data, target in test_loader:
        # for item in test_loader:
        #     print(item.size())
        #     print(item)
            data, target = data.to(device), target.to(device)
            adversarial_class = torch.squeeze(adversarial_target.expand(1, data.size()[0]))

            output_warp, output_class, affine_mat = model(data)

            # out_grid_list = []
            # for i in range(0, output_warp.size()[0]):
            #     out_grid_list.append(torchvision.utils.make_grid(output_warp[0, 0, :, :])[None, 0, :, :])
            #
            # out_grid = torch.cat(out_grid_list)

            # sum up batch loss
            curr_test_loss, curr_mse_loss = calc_loss(output_warp, torch.squeeze(data), output_class, adversarial_class, beta, affine_mat, device)
            test_loss += curr_test_loss
            mse_loss += curr_mse_loss

            pred = output_class.max(1, keepdim=True)[1]
            correct += pred.eq(adversarial_class.view_as(pred)).sum().item()
            wrong += (len(pred) - pred.eq(target.view_as(pred)).sum().item())
            targets.append(torch.unique(target))

            prop_original = model.classifier(data)
            pred_original = prop_original.max(1, keepdim=True)[1]
            correct_original += pred_original.eq(adversarial_class.view_as(pred_original)).sum().item()
            wrong_original += (len(pred_original) - pred_original.eq(target.view_as(pred_original)).sum().item())

        targets_tensor = torch.unique(torch.cat(targets))
        test_loss /= len(test_loader.dataset)
        mse_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Average MSE loss: {:.4f}, Targeted Attack Success: {}/{} ({:.0f}%)'
              .format(test_loss, mse_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))
        print('\nTest set: Not original class: {}/{} ({:.0f}%)'
              .format(wrong, len(test_loader.dataset),
                      100. * wrong / len(test_loader.dataset)))

        print('\nTest set (no warp): Targeted Attack Success: {}/{} ({:.0f}%)'
              .format(correct_original, len(test_loader.dataset),
                      100. * correct_original / len(test_loader.dataset)))
        print('\nTest set (no warp): Not original class: {}/{} ({:.0f}%)'
              .format(wrong_original, len(test_loader.dataset),
                      100. * wrong_original / len(test_loader.dataset)))
        print(targets_tensor)

        logger.info('Test set: Average loss: {:.4f}, Average MSE loss: {:.4f}, Targeted Attack Success: {}/{} ({:.0f}%)'
                    .format(test_loss, mse_loss, correct, len(test_loader.dataset),
                            100. * correct / len(test_loader.dataset)))
        logger.info('Test set: Not original class: {}/{} ({:.0f}%)'
                    .format(wrong, len(test_loader.dataset),
                            100. * wrong / len(test_loader.dataset)))
        logger.info('Test set (no warp): Targeted Attack Success: {}/{} ({:.0f}%)'
                    .format(correct_original, len(test_loader.dataset),
                            100. * correct_original / len(test_loader.dataset)))
        logger.info('Test set (no warp): Not original class: {}/{} ({:.0f}%)'
                    .format(wrong_original, len(test_loader.dataset),
                            100. * wrong_original / len(test_loader.dataset)))
        logger.info(targets_tensor)





