import os
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Carregando os dados de treinamento
def load_data(folder):
    data = []
    for operation in ['subtracao', 'adicao', 'multiplicacao', 'divisao']:
        file_path = f'{folder}/{operation}.csv'
        df = pd.read_csv(file_path)
        data.append(df)
    return data

treinamento_data = load_data('dataset/treinamento')
validacao_data = load_data('dataset/validacao')

# Preparando os dados
def prepare_data(data):
    inputs = []
    outputs = []
    for df in data:
        for _, row in df.iterrows():
            first_number = row['valor1']
            operation = row['operador']
            second_number = row['valor2']
            result = row['resultado']
            inputs.append([first_number, second_number])
            outputs.append(result)
    return np.array(inputs), np.array(outputs)

X_treinamento, y_treinamento = prepare_data(treinamento_data)
X_validacao, y_validacao = prepare_data(validacao_data)
def treinar_epocas_por_arquivo(modelo, pasta_dados, epocas_por_arquivo=10):
    for operacao in ['subtracao', 'adicao', 'multiplicacao', 'divisao']:
        caminho_pasta = os.path.join(pasta_dados, operacao)
        dados = load_data(caminho_pasta)

        for _, linha in dados.iterrows():
            X, y = prepare_data([linha])
            X = X.reshape((X.shape[0], 1, X.shape[1]))

            modelo.fit(X, y, epochs=epocas_por_arquivo, batch_size=batch_size, validation_data=(X_validacao, y_validacao), callbacks=[model_checkpoint])

if __name__ == "__main__":
    dados_treinamento = load_data('dataset/treinamento')
    dados_validacao = load_data('dataset/validacao')

    X_validacao, y_validacao = prepare_data(dados_validacao)

    # Adicione o resto do código relacionado à modelagem e compilação aqui...

    # Callback para salvar o melhor modelo durante o treinamento
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('melhor_modelo.h5', save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', verbose=1)

    # Treinamento por arquivos individuais
    treinar_epocas_por_arquivo(modelo, 'dataset/treinamento', epocas_por_arquivo=10)

# Criando um gráfico de linhas para a perda (loss) durante o treinamento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss no treinamento')
plt.plot(history.history['val_loss'], label='Loss na validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.title('Gráfico de Linhas - Loss no Treinamento e na Validação')
plt.show()