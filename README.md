### Trabalho 02 - SIN393: Classificação de Imagens Usando Redes Neurais Convolucionais (CNNs)

Este é o segundo projeto desenvolvido para a disciplina **SIN393 - Visão Computacional**, com foco na classificação de imagens médicas, usando técnicas avançadas de visão computacional e aprendizado profundo para a classificação de imagens médicas de microscopia. Foram implementados e avaliados diferentes modelos de CNNs (AlexNet, SqueezeNet e ResNet18) em um conjunto de dados composto por 25.000 imagens divididas em cinco classes. O objetivo é explorar a eficácia dessas arquiteturas na diferenciação entre tecidos benignos e malignos, contribuindo para a melhoria de diagnósticos médicos automatizados.

---

## Links

- [Dataset com 25.000 imagens](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) Não foi incluído por questões de espaço do github, porém basta baixá-lo e assinalar o caminho relativo dele a à variavel `diretorio_origem`.
  
  ```
   diretorio_origem = path/to/dataset/on/your-computer
  ```
- [Vídeo de apresentação do artigo](https://drive.google.com/file/d/1ql6LeT_kobulxMD2izYUPaoVvCyTo7ky/view?usp=sharing)

## Estrutura do Projeto

- **`functions.py/`**: Arquivo com as funções principais a serem usadas
- **`main`**: Arquivo principal onde ocorre o treinamento e uso de todas as funções previamente definidas.
- **`requirements.txt`**: Lista de dependências necessárias para o ambiente Python.
- **`README.md`**: Instruções detalhadas para configuração e execução do projeto, além de explicação das principais funções feitas.

---

## Pré-requisitos

- **Python**: Versão 3.8 (compatível com o ambiente do projeto).
- **Anaconda**: Gerenciador de pacotes e ambientes virtuais.
- **GPU Nvidia** (opcional): Para aceleração de treinamento com PyTorch.

---

## Como Rodar o Projeto

### 1. Instalar o Anaconda
Baixe e instale o Anaconda através do [link oficial](https://www.anaconda.com/products/individual).

#### Configuração no VSCode
1. Abra o VSCode e pressione `Ctrl+Shift+P`.
2. Digite **Python: Select Interpreter** e selecione o interpretador Python do ambiente Conda criado.

---

### 2. Criar um Ambiente Virtual
Execute o comando abaixo para criar um ambiente virtual específico para o projeto:

```bash
conda create --name projeto_visao python=3.8
```

---

### 3. Ativar o Ambiente
Ative o ambiente virtual criado:

```bash
conda activate projeto_visao
```

#### (Opcional) Configuração para GPU Nvidia
Caso utilize uma GPU Nvidia, instale as dependências adequadas com o comando:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### 4. Instalar Dependências
#### 4.1. Instalar o Pip no Ambiente Conda
Para gerenciar pacotes, instale o Pip no ambiente Conda:

```bash
conda install pip
```

#### 4.2. Instalar Pacotes do `requirements.txt`
Execute o comando abaixo para instalar todas as dependências do projeto:

```bash
pip install -r requirements.txt
```

#### 4.3. Verificar Instalação
Confirme se todos os pacotes foram instalados corretamente:

```bash
pip list
```

---

### 5. Executar o Projeto
Após configurar o ambiente, execute o código principal com:

```bash
python main.py
```

---

## Explicação das principais funções

## **1. `criar_dataloaders`**
Essa função divide o conjunto de dados em três subconjuntos (treinamento, validação e teste) e cria `dataloaders` para facilitar o carregamento em lotes durante o treinamento.

### Parâmetros:
- `dataset_completo`: O conjunto de dados completo a ser dividido.
- `batch_size`: O tamanho dos lotes usados no treinamento e validação.

### Retorna:
- `train_dataloader`: Carregador de dados para o conjunto de treinamento.
- `val_dataloader`: Carregador de dados para o conjunto de validação.
- `test_dataloader`: Carregador de dados para o conjunto de teste.

---

## **2. `menu_modelo`**
Permite selecionar e configurar um dos modelos de CNN disponíveis (AlexNet, SqueezeNet ou ResNet18) com base no número de classes no conjunto de dados.

### Parâmetros:
- `classes`: Lista de classes do conjunto de dados.

### Retorna:
- Um modelo pré-treinado ajustado para o número de classes no problema.

---

## **3. `treinar_validar_e_testar`**
Executa o ciclo completo de treinamento, validação e teste de um modelo em várias épocas, calculando métricas como perda e acurácia para cada etapa.

### Parâmetros:
- `modelo`: O modelo de CNN a ser treinado.
- `train_dataloader`, `val_dataloader`, `test_dataloader`: Carregadores de dados para treinamento, validação e teste.
- `optimizer`: Otimizador para atualizar os pesos do modelo.
- `criterion`: Função de perda para calcular o erro.
- `scheduler`: Agendador para ajustar a taxa de aprendizado.
- `epochs`: Número de épocas para o treinamento.
- `DEVICE`: Dispositivo onde o modelo será executado (CPU ou GPU).

### Retorna:
- Listas contendo as perdas e acurácias para treinamento, validação e teste.

---

## **4. `avaliar_e_imprimir_resultados`**
Avalia o desempenho do modelo no conjunto de validação, gerando a matriz de confusão, o relatório de classificação e a acurácia geral.

### Parâmetros:
- `modelo`: O modelo treinado.
- `val_dataloader`: Carregador de dados para validação.
- `dispositivo`: Dispositivo de execução (CPU ou GPU).
- `class_names`: Lista com os nomes das classes.
- `resultado`: Métricas obtidas durante o treinamento.

### Funcionalidades:
- Calcula a matriz de confusão.
- Gera o relatório de classificação com precisão, recall e F1-score.
- Calcula a acurácia do conjunto de validação.

---

## Fluxo Geral do Projeto
1. **Carregamento do Dataset**:
   - O dataset é carregado e transformado (redimensionamento, normalização) usando o `torchvision.transforms`.

2. **Criação dos DataLoaders**:
   - `criar_dataloaders` divide o dataset e cria carregadores de dados.

3. **Configuração do Modelo**:
   - `menu_modelo` permite selecionar e configurar um modelo pré-treinado.

4. **Treinamento e Validação**:
   - `treinar_validar_e_testar` realiza o treinamento, validação e teste do modelo, ajustando os pesos.

5. **Avaliação**:
   - `avaliar_e_imprimir_resultados` gera métricas e relatórios detalhados para avaliação do modelo.

---


## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorias no código ou na documentação.
