import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def step_function(x):
    if x >= 0:
        return 1
    else:
        return -1

def predict_perceptron(weights, x):
    # calculando produto escalar (valor1 * peso1 + valor2 * peso2 + ...)
    z = np.dot(weights, x)

    # chamando funcao de ativacao
    return step_function(z)

def train_perceptron(data, learning_rate=0.1, epochs=100):
    # inicializando pesos aleatoriamente
    weights = np.random.uniform(-1, 1, size=data.shape[1] - 1)

    for epoch in range(epochs):
        for i in range(len(data)):
            # pegando os valores
            x = data.iloc[i, :-1].values

            # pegando a label
            y = data.iloc[i, -1]

            predicted_class = predict_perceptron(weights, x)

            # atualizando pesos por meio da comparação com o valor real
            error = y - predicted_class
            weights += learning_rate * error * x

    return weights

def train_test_get_metrics(train_data, test_data, learning_rate=0.1, epochs=100):
    # treinando perceptron
    trained_weights = train_perceptron(train_data, learning_rate, epochs)

    # testando perceptron
    test_data['predicted_class'] = test_data.apply(lambda x: predict_perceptron(trained_weights, x[:-1]), axis=1)

    correct_predictions = 0
    for i in range(len(test_data)):
        if test_data.iloc[i, -1] == test_data.iloc[i, -2]:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)

    return trained_weights, test_data, accuracy

def save_results(hold_out_10_results, hold_out_30_results, hold_out_50_results, epoch_qtd, learning_rate):
    folder_name = f'output/ep_{epoch_qtd}_lr_{learning_rate}'
    # criando pasta caso nao exista
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    hold_out_10_results['test_data'].to_csv(f'{folder_name}/10_90_test_data_result.csv', index=False)
    hold_out_30_results['test_data'].to_csv(f'{folder_name}/30_70_test_data_result.csv', index=False)
    hold_out_50_results['test_data'].to_csv(f'{folder_name}/50_50_test_data_result.csv', index=False)

    header_line = f"Hold-out;Pesos treinados;Acuracia\n"
    with open(f'{folder_name}/metrics.out', 'w') as f:
        f.write(header_line)
        f.write(f"10-90;{hold_out_10_results['weights']};{hold_out_10_results['accuracy']}\n")
        f.write(f"30-70;{hold_out_30_results['weights']};{hold_out_30_results['accuracy']}\n")
        f.write(f"50-50;{hold_out_50_results['weights']};{hold_out_50_results['accuracy']}\n")

def main():
    # lendo dataset
    data = pd.read_csv("iris/iris.data", header=None)

    # selecionando características comprimento da sépala (coluna 0) e largura da sépala (coluna 1)
    # e as duas espécies Setosa e Versicolor (primeiras 100 linhas)
    data = data.iloc[:100, [0, 1, 4]]

    # 1 = Setosa
    # -1 = Versicolor
    data[4] = np.where(data[4] == 'Iris-setosa', 1, -1)

    # embaralhando os dados do dataset 
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # separando dados de treino e teste (hold-out)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # iterando por epochs e learning rates para salvar tudo separado em pastas depois
    for epoch_qtd in [10, 100, 1000]:
        for learning_rate in [0.1, 0.2, 0.3]:
            # hold-outs do trabalho
            X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(X, y, test_size=0.9, random_state=42)
            X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(X, y, test_size=0.7, random_state=42)
            X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(X, y, test_size=0.5, random_state=42)

            # juntando em dataset pra continuar usando a funcao de treino
            train_data_10 = pd.DataFrame(np.column_stack((X_train_10, y_train_10)))
            test_data_10 = pd.DataFrame(np.column_stack((X_test_10, y_test_10)))

            train_data_30 = pd.DataFrame(np.column_stack((X_train_30, y_train_30)))
            test_data_30 = pd.DataFrame(np.column_stack((X_test_30, y_test_30)))

            train_data_50 = pd.DataFrame(np.column_stack((X_train_50, y_train_50)))
            test_data_50 = pd.DataFrame(np.column_stack((X_test_50, y_test_50)))

            # treinando e testando perceptron para todos os hold-outs
            trained_weights_10, test_data_10, accuracy_10 = train_test_get_metrics(train_data_10, test_data_10, learning_rate, epoch_qtd)
            trained_weights_30, test_data_30, accuracy_30 = train_test_get_metrics(train_data_30, test_data_30, learning_rate, epoch_qtd)
            trained_weights_50, test_data_50, accuracy_50 = train_test_get_metrics(train_data_50, test_data_50, learning_rate, epoch_qtd)

            hold_out_10_results = {
                'weights': str(trained_weights_10).replace("   ", ",").replace("  ", ",").replace(" ", ","),
                'test_data': test_data_10,
                'accuracy': accuracy_10
            }

            hold_out_30_results = {
                'weights': str(trained_weights_30).replace("   ", ",").replace("  ", ",").replace(" ", ","),
                'test_data': test_data_30,
                'accuracy': accuracy_30
            }

            hold_out_50_results = {
                'weights': str(trained_weights_50).replace("   ", ",").replace("  ", ",").replace(" ", ","),
                'test_data': test_data_50,
                'accuracy': accuracy_50
            }

            # salvando resultados
            save_results(hold_out_10_results, hold_out_30_results, hold_out_50_results, epoch_qtd, learning_rate)

if __name__ == "__main__":
  main()