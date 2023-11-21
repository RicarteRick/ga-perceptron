import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

#region Algoritmo Genetico
def get_fitness_perceptron(weights, data):
  correct_predictions = 0

  for i in range(len(data)):
    x = data.iloc[i, :-1].values # valores das caracteristicas
    y = data.iloc[i, -1] # classe real

    predicted_class = predict_perceptron(weights, x)

    # se classe predita for igual a classe real, incrementa contador de acertos
    if predicted_class == y:
      correct_predictions += 1

  # calcula acuracia
  accuracy = correct_predictions / len(data)
  return accuracy

# seleção por torneio
def select_parents(population, fitness, size=2):
  selected = []

  for i in range(len(population)):
    competitors = np.random.choice(len(population), size=size, replace=False)
    best = competitors[0]

    # seleciona o melhor individuo dentre os competidores (maior fitness)
    for j in competitors[1:]:
      if fitness[j] > fitness[best]:
        best = j
    selected.append(population[best])
  return np.array(selected)

def crossover(parents, alpha):
  offspring = []

  # para cada par de pais, realiza crossover
  for i in range(0, len(parents), 2):
    p1 = parents[i]
    p2 = parents[i+1]

    # realiza crossover aritmetico
    o1 = alpha * p1 + (1 - alpha) * p2 
    o2 = alpha * p2 + (1 - alpha) * p1
    offspring.append(o1)
    offspring.append(o2)
  return np.array(offspring)

def mutation(offspring, sigma=0.1, mutation_rate=0.5):
  mutated = []

  for o in offspring:
    random_number_mutation = random.uniform(0,1)
    m = o

    # realiza mutacao se o numero aleatorio for menor que a taxa de mutacao
    if random_number_mutation <= mutation_rate:
      m = o + np.random.normal(0, sigma, size=len(o))
    mutated.append(m)
  return np.array(mutated)
#endregion

#region Perceptron
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

def initialize_population(pop_size, chrom_size):
  population = np.random.uniform(-1, 1, size=(pop_size, chrom_size))
  return population

def genetic_algorithm(data, pop_size=100, max_gens=1000, k=2, alpha=0.5, sigma=0.1):
  mutation_rate = 0.5

  # inicializar a população
  population = initialize_population(pop_size, data.shape[1] - 1)

  # avaliar a população inicial
  fitness = np.array([get_fitness_perceptron(pop, data) for pop in population])
  
  # inicializar a melhor solução
  best_solution = population[np.argmax(fitness)]
  best_fitness = np.max(fitness)
  
  gen = 0

  # repetir até o critério de parada ser satisfeito
  while gen < max_gens:
    # selecionar os pais
    parents = select_parents(population, fitness, k)
    
    # cruzamento
    offspring = crossover(parents, alpha)
    
    # mutação 
    mutated = mutation(offspring, sigma, mutation_rate)
  
    # avaliar os filhos e substituir a população
    new_fitness = np.array([get_fitness_perceptron(mut, data) for mut in mutated])
    population = mutated
    fitness = new_fitness

    # atualizar a melhor solução
    current_best_solution = population[np.argmax(fitness)]
    current_best_fitness = np.max(fitness)

    if current_best_fitness > best_fitness:
      best_solution = current_best_solution
      best_fitness = current_best_fitness

    gen += 1

  # retornar a melhor solução e seu fitness
  return best_solution, best_fitness

def main():
  generation_qtd = 100

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
      best_weights_10, best_fitness_10 = genetic_algorithm(train_data_10, max_gens=generation_qtd)
      best_weights_30, best_fitness_30 = genetic_algorithm(train_data_30, max_gens=generation_qtd)
      best_weights_50, best_fitness_50 = genetic_algorithm(train_data_50, max_gens=generation_qtd)

      # Avaliar os pesos otimizados no conjunto de teste
      test_accuracy_10 = get_fitness_perceptron(best_weights_10, test_data_10)
      test_accuracy_30 = get_fitness_perceptron(best_weights_30, test_data_30)
      test_accuracy_50 = get_fitness_perceptron(best_weights_50, test_data_50)

      # Imprimir os resultados
      print(f"Epochs: {epoch_qtd}, Learning rate: {learning_rate}, Generation qtd: {generation_qtd}")
      print(f"Hold-out 10%: Train fitness: {best_fitness_10:.4f}, Test accuracy: {test_accuracy_10:.4f}, Weights: {best_weights_10}")
      print(f"Hold-out 30%: Train fitness: {best_fitness_30:.4f}, Test accuracy: {test_accuracy_30:.4f}, Weights: {best_weights_30}")
      print(f"Hold-out 50%: Train fitness: {best_fitness_50:.4f}, Test accuracy: {test_accuracy_50:.4f}, Weights: {best_weights_50}")
      print()

if __name__ == "__main__":
  main()
#endregion