import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import ElectraForSequenceClassification
from typing import Dict


class ElectraModel(BaseModel):
    def __init__(self, model_path: str, id2label: Dict[int,any], label2id: Dict[any,int],
                num_labels=4, problem_type='multi_label_classification', trainable=True):
        self.electra = ElectraForSequenceClassification.from_pretrained(
            model_path,
            problem_type=problem_type,
            ignore_mismatched_sizes=True,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,)
        if not trainable:
            self.set_trainable(trainable)

    def set_trainable(self, trainable: bool):
        requires_grad = True if trainable else False
        for param in self.electra.base_model.parameters():
            param.requires_grad = requires_grad


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
