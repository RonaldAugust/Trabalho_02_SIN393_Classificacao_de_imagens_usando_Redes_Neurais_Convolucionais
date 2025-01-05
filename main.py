from functions import criar_dataloaders, menu_modelo, treinar_validar_e_testar, avaliar_e_imprimir_resultados

import torch
from torchvision import transforms, datasets
from torch import nn, optim
from torch.optim import lr_scheduler

dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\nDispositivo: {0}'.format(dispositivo))

#diretorios
diretorio_origem = 'C:\\Users\\Ronald\\Documents\\Disciplinas\\Visao_computacional\\proj_final\\DATASET_HISTOPATOLOGICAS_PULMAO_COLON_LC25000'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), #redimensionamento
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset_completo = datasets.ImageFolder(diretorio_origem, transform=data_transforms)

classes = dataset_completo.classes

num_classes = len(classes)

#menu para escolher o modelo a ser rodado
modelo = menu_modelo(classes)
modelo = modelo.cuda(dispositivo)


Hyper_parametros = {
    'num_classes': len(classes),
    'class_names': classes,
    'batch_size': 64,
    'lr': 0.001,
    'mm': 0.9,
    'epochs': 50,
    'model_name': modelo,
    'criterion': nn.CrossEntropyLoss()  # Critério de perda
}

# Definindo o otimizador
optimizer = optim.SGD(modelo.parameters(), lr=Hyper_parametros['lr'], momentum=Hyper_parametros['mm'])

# Definindo o scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#divide os dados em treino, teste e validacao 70%, 15%, 15%
dataloaders = criar_dataloaders(dataset_completo, Hyper_parametros.get('batch_size'))
train_dataloader = dataloaders[0]
val_dataloader = dataloaders[1]
test_dataloader = dataloaders[2]

resultado = treinar_validar_e_testar(modelo, train_dataloader, val_dataloader, test_dataloader, optimizer, Hyper_parametros.get('criterion'), scheduler, Hyper_parametros.get('epochs'), dispositivo)
avaliar_e_imprimir_resultados(modelo, val_dataloader, dispositivo, classes, resultado)


