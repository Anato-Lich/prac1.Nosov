#LBL1 - дополнительные библиотеки.
from torchvision.models import resnet50, ResNet
import torch, torchvision
from typing import Tuple, Dict

from torch.utils.tensorboard import SummaryWriter


#LBL2 - изменена загрузка датасета. Файлы должны быть в корне диска.

#LBL3 - реализация getitem.

#LBL4 - реализация метрик для подсчета по эпохам и вывода в тензорборд.
Возможно, что эти метрики не совсем корректно считаются, т.к. для них рекомендуется оставлять размер батча 1, в реализации батч 64.

#LBL5 - реализация эпохи обучения.

#LBL6 - реализация валидации на тестовом датасете.

#LBL7 - загрузка весов модели. Веса должны быть в корне диска.

#LBL8 - реализация обучения модели с сохранением промежуточных весов в median_weights.

#LBL9 - создание даталоадера для работы с ResNet18.

#LBL10 - вывод слоев модели.

#LBL11 - тензорборд с метриками.
В процессе работы создается папка logs/model, на GoogleDrive не сохраняется. 

Суммарно модель обучалась около 70 эпох, несколько десятков были обучены с 
criterion = torch.nn.CrossEntropyLoss(torch.Tensor([1, 1.3, 1, 1, 1, 1, 1, 1, 1]).to(device))


metrics for 10% of test:
	 accuracy 0.9733:
	 balanced accuracy 0.9733
   
metrics for test:
	 accuracy 0.9147:
	 balanced accuracy 0.9147:
   
metrics for test-tiny:
	 accuracy 0.8667:
	 balanced accuracy 0.8667:
   
   
Написанный код также работает для дообучения ResNet50, однако обучение требует существенно больше времени.

