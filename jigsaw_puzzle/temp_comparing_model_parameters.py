from torchvision import models
import torch

def compareModels(model1, model2):

    models_differ = 0

    list_of_mismatched_layers = []
    list_of_unchanged_layers = []

    for keyItem1, keyItem2 in zip(model1.state_dict().items(),
                                  model2.state_dict().items()):

        if torch.equal(keyItem1[1], keyItem2[1]):
            if (keyItem1[0] == keyItem2[0]):
                list_of_unchanged_layers.append(keyItem1[0])
        else:
            models_differ += 1
            if (keyItem1[0] == keyItem2[0]):
                # print("Mismatch found at ", keyItem1[0])
                list_of_mismatched_layers.append(keyItem1[0])
            else:
                raise Exception
    if models_differ == 0:
        print("Models Match perfectly")
    else:
        print("list of mismatched layers ", list_of_mismatched_layers)
        print("list of unchangeed layers ", list_of_unchanged_layers)
        print("-" * 20)

resnet18Model = models.resnet18(pretrained=True)

print(resnet18Model)

print(resnet18Model.parameters())

resnet18model2 = models.resnet18(pretrained=False)

compareModels(resnet18Model, resnet18model2)

otro = models.resnet18(pretrained=False)

print("first time")
compareModels(resnet18model2, otro)

otro.load_state_dict(resnet18model2.state_dict())

print("second time")
compareModels(resnet18model2, otro)