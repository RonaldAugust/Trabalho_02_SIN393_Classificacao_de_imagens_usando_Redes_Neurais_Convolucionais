import time
import torch
import numpy as np
from sklearn import metrics
from torchvision import models
import torch.nn as nn
import numpy as np

def criar_dataloaders(dataset_completo, batch_size):
    train_size = int(0.7 * len(dataset_completo))
    val_size = int(0.15 * len(dataset_completo))
    test_size = len(dataset_completo) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset_completo, [train_size, val_size, test_size], generator=generator
    )

    print("Número de imagens no conjunto de treino:", len(train_dataset))
    print("Número de imagens no conjunto de validação:", len(val_dataset))
    print("Número de imagens no conjunto de teste:", len(test_dataset))

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def menu_modelo(classes):
    num_classes = len(classes)
    
    print("Escolha um modelo para inicializar:")
    print("1. AlexNet")
    print("2. SqueezeNet")
    print("3. ResNet18")
    print("4. Sair")
    
    escolha = input()

    if escolha == "1":
        nome_modelo = "alexnet"
    elif escolha == "2":
        nome_modelo = "SqueezeNet"
    elif escolha == "3":
        nome_modelo = "ResNet18"
    elif escolha == "4":
        exit()
    else:
        print("Opção inválida, saindo...")
        exit()

    modelo = None

    if nome_modelo == "alexnet":
        modelo = models.alexnet(weights='DEFAULT')
        num_ftrs = modelo.classifier[6].in_features
        modelo.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif nome_modelo == "SqueezeNet":
        modelo = models.squeezenet1_1(weights='DEFAULT')
        num_ftrs = modelo.classifier[1].in_channels  
        modelo.classifier[1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=(1, 1), stride=(1, 1))  
        modelo.classifier[1].bias = nn.Parameter(torch.zeros(num_classes)) 

    elif nome_modelo == "ResNet18":
        modelo = models.resnet18(weights='DEFAULT')
        num_ftrs = modelo.fc.in_features
        modelo.fc = nn.Linear(num_ftrs, num_classes)
        
    else:
        print("Nome de modelo inválido, saindo...")
        exit()

    print(f"Modelo {nome_modelo} inicializado com sucesso!\n")
    return modelo

def treinar_validar_e_testar(modelo, train_dataloader, val_dataloader, test_dataloader, optimizer, criterion, scheduler, epochs, DEVICE):
    time_total_start = time.time()

    train_loss_list = []
    train_acc_list = []
        
    val_loss_list = []
    val_acc_list = []

    test_loss_list = []
    test_acc_list = []

    modelo = modelo.to(DEVICE)
    print('treino iniciado\n')

    for epoch in range(epochs):
        time_epoch_start = time.time()
         
        modelo.train()

        loss_epoch_train = 0.0
        
        hits_epoch_train = 0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            torch.set_grad_enabled(True)
            
            outputs = modelo(inputs)
            
            preds = torch.argmax(outputs, dim=1).float()
            
            loss = criterion(outputs, labels)

            loss.backward()
            
            optimizer.step()

            loss_epoch_train += loss.item() * inputs.size(0)
            
            hits_epoch_train += torch.sum(preds == labels.data)


        train_loss = loss_epoch_train / len(train_dataloader.dataset)
        train_acc = hits_epoch_train.double() / len(train_dataloader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        modelo.eval()
        
        loss_epoch_val = 0.0
        
        hits_epoch_val = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = modelo(inputs)
                preds = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                loss_epoch_val += loss.item() * inputs.size(0)
                hits_epoch_val += torch.sum(preds == labels.data)

        scheduler.step()

        val_loss = loss_epoch_val / len(val_dataloader.dataset)
        val_acc = hits_epoch_val.double() / len(val_dataloader.dataset)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        loss_epoch_test = 0.0
        hits_epoch_test = 0

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = modelo(inputs)
                preds = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                loss_epoch_test += loss.item() * inputs.size(0)
                hits_epoch_test += torch.sum(preds == labels.data)

        test_loss = loss_epoch_test / len(test_dataloader.dataset)
        test_acc = hits_epoch_test.double() / len(test_dataloader.dataset)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        time_epoch = time.time() - time_epoch_start
        print(f'Epoch {epoch}/{epochs - 1} - TRAIN Loss: {train_loss:.4f} TRAIN Acc: {train_acc:.4f} - '
              f'VAL Loss: {val_loss:.4f} VAL Acc: {val_acc:.4f} - TEST Loss: {test_loss:.4f} TEST Acc: {test_acc:.4f} '
              f'({time_epoch:.2f}s)')

    time_total_train = time.time() - time_total_start
    print(f'\nTreinamento finalizado! ({int(time_total_train // 60)}m {int(time_total_train % 60)}s)')

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list, test_loss_list, test_acc_list

def avaliar_e_imprimir_resultados(modelo, val_dataloader, dispositivo, class_names, resultado):
    true_val_list = []
    pred_val_list = []
    prob_val_list = []

    for i, (img_list, labelList) in enumerate(val_dataloader):
        if dispositivo.type == 'cuda':
            img_list = img_list.to(dispositivo)
            labelList = labelList.to(dispositivo)

        torch.set_grad_enabled(False)

        outputs = modelo(img_list)

        preds = torch.argmax(outputs, dim=1)

        outputs_prob = nn.functional.softmax(outputs, dim=1)
        prob_val_batch = np.asarray(outputs_prob.cpu())

        if dispositivo.type == 'cuda':
            true_val_batch = np.asarray(labelList.cpu())
            pred_val_batch = np.asarray(preds.cpu())

        for i in range(len(pred_val_batch)):
            true_val_list.append(true_val_batch[i])
            pred_val_list.append(pred_val_batch[i])
            prob_val_list.append(prob_val_batch[i])

    conf_mat_val = metrics.confusion_matrix(true_val_list, pred_val_list)
    print('\nConfusion Matrix (Validation):')
    print(conf_mat_val)

    class_rep_val = metrics.classification_report(
        true_val_list, pred_val_list, 
        target_names=class_names, digits=4, zero_division=0
    )
    print('\nClassification Report (Validation):')
    print(class_rep_val)

    acc_val = metrics.accuracy_score(true_val_list, pred_val_list)
    print('\nValidation Accuracy: {:.4f}'.format(acc_val))

