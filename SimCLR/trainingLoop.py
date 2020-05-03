import time
import torch

from lossFunction import simCLRLossFunction

def train_model(featureModel, zModel, dataloader, dataset_sizes,
                criterion, optimizer, scheduler, device,
                num_epochs=25,

                model_name = "Alexnet"):

    since = time.time()

    best_loss = 1e6

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:

            if phase == 'train':
                featureModel.train()  # Set model to training mode
                zModel.train()
            else:
                featureModel.eval()   # Set model to evaluate mode
                zModel.eval()

            running_loss = 0.0

            counter_items = 0

            for batch in dataloader[phase]:

                firstAugmentation = batch[0].to(device)
                secondAugmentation = batch[1].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    first_extract = featureModel(firstAugmentation)
                    second_extract = featureModel(secondAugmentation)

                    zFirstOutput = zModel(first_extract)
                    zSecondOutput = zModel(second_extract)

                    loss = simCLRLossFunction(zFirstOutput, zSecondOutput, criterion)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * firstAugmentation.shape[0]
                counter_items += firstAugmentation.shape[0]

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes[phase] )

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_featureModel_wts = copy.deepcopy(featureModel.state_dict())
                best_zedModel_wts = copy.deepcopy(z_model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    featureModel.load_state_dict(best_featureModel_wts)
    zModel.load_state_dict(best_zedModel_wts)
    return featureModel, zModel