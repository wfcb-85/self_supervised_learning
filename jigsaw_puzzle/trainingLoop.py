import time
import torch
import copy

def train_model(model, dataloader, dataset_sizes, criterion, optimizer,
                scheduler, device, num_epochs=25,
                model_name = "Alexnet"):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            counter_items = 0

            for batch in dataloader[phase]:

                inputsCentralCrops = batch[0].to(device)
                inputsRandomCrops = batch[1].to(device)

                labels = batch[2].to(device)

                labels = labels.view(labels.shape[0])

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    if model_name == "Alexnet":

                        first_extract = model.forward_extraction(inputsCentralCrops)
                        second_extract = model.forward_extraction(inputsRandomCrops)

                        fused_output = model.forward_fuse(first_extract, second_extract)

                    elif model_name == "resnet18":

                        first_extract = model(inputsCentralCrops)
                        second_extract = model(inputsRandomCrops)

                        fused_output = model.fused_output(first_extract, second_extract)

                    _, preds = torch.max(fused_output, 1)

                    loss = criterion(fused_output, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputsCentralCrops.size(0)
                counter_items += inputsCentralCrops.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes[phase] )
            epoch_acc = running_corrects.double() / (dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model