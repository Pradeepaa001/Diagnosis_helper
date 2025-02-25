import syft as sy
import torch
import torchvision.models as models

hook = sy.TorchHook(torch)

hospital_1 = sy.VirtualWorker(hook, id="hospital_1")
hospital_2 = sy.VirtualWorker(hook, id="hospital_2")

def federated_train():
    model_1 = models.resnet18(pretrained=True).send(hospital_1)
    model_2 = models.resnet18(pretrained=True).send(hospital_2)

    for _ in range(5):
        dummy_input = torch.rand(1, 3, 224, 224)
        output_1 = model_1(dummy_input)
        output_2 = model_2(dummy_input)

    model_1.get()
    model_2.get()
