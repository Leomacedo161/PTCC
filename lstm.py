import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Adicionando seis camadas LSTM à rede neural
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, activation='relu', input_shape=(None, 2), return_sequences=True),
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Callback para salvar o melhor modelo durante o treinamento
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', verbose=1)

# Reshape dos dados para serem compatíveis com a entrada da LSTM
X_treinamento = X_treinamento.reshape((X_treinamento.shape[0], 1, X_treinamento.shape[1]))
X_validacao = X_validacao.reshape((X_validacao.shape[0], 1, X_validacao.shape[1]))

# Definindo o tamanho do lote
batch_size = 64  

# Treinando o modelo com o ModelCheckpoint e o tamanho do lote
history = model.fit(X_treinamento, y_treinamento, epochs=10, batch_size=batch_size, validation_data=(X_validacao, y_validacao), callbacks=[model_checkpoint])

# Avaliando o modelo
resultado = model.evaluate(X_validacao, y_validacao)
print(f'Loss na validação: {resultado}')

# Carregando o modelo treinado a partir do melhor checkpoint global
best_model = tf.keras.models.load_model('best_model.h5')

# Carregando os dados de teste
teste_data = load_data('dataset/teste')
X_teste, y_teste = prepare_data(teste_data)

# Reshape dos dados de teste
X_teste = X_teste.reshape((X_teste.shape[0], 1, X_teste.shape[1]))

# Realizando previsões com o melhor modelo
previsoes = best_model.predict(X_teste)

# Avaliando o desempenho no conjunto de teste
resultado_teste = best_model.evaluate(X_teste, y_teste)
print(f'Loss no teste: {resultado_teste}')

# Criando um gráfico de linhas para a perda (loss) durante o treinamento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss no treinamento')
plt.plot(history.history['val_loss'], label='Loss na validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.title('Gráfico de Linhas - Loss no Treinamento e na Validação')
plt.show()
