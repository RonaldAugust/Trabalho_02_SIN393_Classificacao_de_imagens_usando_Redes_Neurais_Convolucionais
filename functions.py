import time
import torch
import numpy as np
from sklearn import metrics
from torchvision import models
import torch.nn as nn

def criar_dataloaders(dataset_completo, batch_size):
    # Conjunto de treinamento: 70 %
    train_size = int(0.7 * len(dataset_completo))
    # Conjunto de validação: 15 %
    val_size = int(0.15 * len(dataset_completo))
    # Conjunto de teste: 15 %
    test_size = len(dataset_completo) - train_size - val_size

    # Dividindo o conjunto completo
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset_completo, [train_size, val_size, test_size], generator=generator
    )

    # Número de imagens em cada conjunto
    print("Número de imagens no conjunto de treino:", len(train_dataset))
    print("Número de imagens no conjunto de validação:", len(val_dataset))
    print("Número de imagens no conjunto de teste:", len(test_dataset))

    # Definindo os dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def menu_modelo(classes):
    num_classes = len(classes)
    
    print("Escolha um modelo para inicializar:")
    print("1. AlexNet")
    print("2. VGG16")
    print("3. ResNet18")
    print("4. Sair")
    
    escolha = input()

    if escolha == "1":
        nome_modelo = "alexnet"
    elif escolha == "2":
        nome_modelo = "vgg"
    elif escolha == "3":
        nome_modelo = "ResNet18"
    elif escolha == "4":
        exit()
    else:
        print("Opção inválida, saindo...")
        exit()

    # Inicializar o modelo com base na escolha
    modelo = None
    tamanho_entrada = 0

    if nome_modelo == "alexnet":
        modelo = models.alexnet(weights='DEFAULT')
        num_ftrs = modelo.classifier[6].in_features
        # Altera o número de neurônios na cadama de saída.
        modelo.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif nome_modelo == "vgg":
        modelo = models.vgg16(weights='DEFAULT')
        num_ftrs = modelo.classifier[6].in_features
        modelo.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
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
# Tempo total do treinamento (treinamento e validação)
    time_total_start = time.time()

# Lista das perdas (loss) e acurácias (accuracy) de treino para cada época.
    train_loss_list = []
    train_acc_list = []
    
# Lista das perdas (loss) e acurácias (accuracy) de validação para cada época.    
    val_loss_list = []
    val_acc_list = []

# Lista das perdas (loss) e acurácias (accuracy) de teste para cada época.
    test_loss_list = []
    test_acc_list = []

    modelo = modelo.to(DEVICE)

    for epoch in range(epochs):
        print('treino iniciado\n')
        time_epoch_start = time.time()
        
        # Inicia contagem de tempo da época 
        modelo.train()

        # Perda (loss) nesta época
        loss_epoch_train = 0.0
        
        # Amostras classificadas corretamente nesta época
        hits_epoch_train = 0

        # Iterar ao longo dos lotes do CONJUNTO DE TREINAMENTO
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zera os parametros do gradiente
            optimizer.zero_grad()
            
            # Habilita o cálculo do gradiente
            torch.set_grad_enabled(True)
            
            # Saída do modelo para o lote
            outputs = modelo(inputs)
            
            # 'outputs' está em porcentagens. Tomar os maximos como resposta.
            preds = torch.argmax(outputs, dim=1).float()
            
            # Calcula a perda (loss)
            loss = criterion(outputs, labels)

            # BACKWARD
            # <-------
            loss.backward()
            
            # Atualiza os parâmetros da rede
            optimizer.step()

            # Atualiza a perda da época
            loss_epoch_train += loss.item() * inputs.size(0)
            
            # Atualiza o número de amostras classificadas corretamente na época.
            hits_epoch_train += torch.sum(preds == labels.data)


        train_loss = loss_epoch_train / len(train_dataloader.dataset)
        train_acc = hits_epoch_train.double() / len(train_dataloader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # VALIDAÇÃO
        
        modelo.eval()
        
        # Pego o numero de perda e o numero de acertos
        loss_epoch_val = 0.0
        
        # Numero de itens corretos
        hits_epoch_val = 0

        # Iterar ao longo dos lotes do CONJUNTO DE VALIDAÇÃO
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
    # Lista com as classes reais e preditas
    true_val_list = []
    pred_val_list = []
    # Lista com as probabilidades
    prob_val_list = []

    # Itera pelos lotes do conjunto de validação
    for i, (img_list, labelList) in enumerate(val_dataloader):
        if dispositivo.type == 'cuda':
            img_list = img_list.to(dispositivo)
            labelList = labelList.to(dispositivo)

        # Desabilita o cálculo do gradiente durante validação e testes.
        torch.set_grad_enabled(False)

        # Forward pass
        outputs = modelo(img_list)

        # Predição
        preds = torch.argmax(outputs, dim=1)

        # Calcula probabilidades
        outputs_prob = nn.functional.softmax(outputs, dim=1)
        prob_val_batch = np.asarray(outputs_prob.cpu())

        # Classes reais e preditas para este lote
        if dispositivo.type == 'cuda':
            true_val_batch = np.asarray(labelList.cpu())
            pred_val_batch = np.asarray(preds.cpu())

        # Adiciona os dados ao conjunto completo
        for i in range(len(pred_val_batch)):
            true_val_list.append(true_val_batch[i])
            pred_val_list.append(pred_val_batch[i])
            prob_val_list.append(prob_val_batch[i])

    # Confusion Matrix
    conf_mat_val = metrics.confusion_matrix(true_val_list, pred_val_list)
    print('\nConfusion Matrix (Validation):')
    print(conf_mat_val)

    # Classification Report
    class_rep_val = metrics.classification_report(
        true_val_list, pred_val_list, 
        target_names=class_names, digits=4, zero_division=0
    )
    print('\nClassification Report (Validation):')
    print(class_rep_val)

    # Accuracy
    acc_val = metrics.accuracy_score(true_val_list, pred_val_list)
    print('\nValidation Accuracy: {:.4f}'.format(acc_val))

    # Imprime os resultados de treinamento, validação e teste
    imprimir_resultados_com_matriz(resultado, true_val_list, pred_val_list, class_names)

def imprimir_resultados_com_matriz(resultado, true_val_list, pred_val_list, class_names):
    # Descompacta os valores da variável resultado
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, test_loss_list, test_acc_list = resultado
    
    # Imprime os resultados de treinamento e validação
    print("\n--- Resultados de Treinamento e Validação ---")
    print("Perdas durante o treinamento por época:")
    print(train_loss_list)
    
    print("\nPerdas durante a validação por época:")
    print(val_loss_list)
    
    print("\nAcurácias durante o treinamento por época:")
    print(train_acc_list)
    
    print("\nAcurácias durante a validação por época:")
    print(val_acc_list)
    
    # Imprime os resultados do teste
    print("\n--- Resultados de Teste ---")
    print(f"Perda no teste: {test_loss_list[-1]:.4f}")
    print(f"Acurácia no teste: {test_acc_list[-1]:.4f}")
    
    # Matriz de Confusão
    conf_mat_val = metrics.confusion_matrix(true_val_list, pred_val_list)
    print('\nConfusion Matrix (Validation):')
    print(conf_mat_val)

    # Relatório de Classificação
    class_rep_val = metrics.classification_report(
        true_val_list, pred_val_list,
        target_names=class_names, digits=4, zero_division=0
    )
    print('\nClassification Report (Validation):')
    print(class_rep_val)

    # Acurácia na Validação
    acc_val = metrics.accuracy_score(true_val_list, pred_val_list)
    print('\nValidation Accuracy: {:.4f}'.format(acc_val))
