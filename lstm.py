import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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

# Normalizando os dados
scaler = MinMaxScaler()
X_treinamento = scaler.fit_transform(X_treinamento.reshape(-1, 2)).reshape(X_treinamento.shape)
X_validacao = scaler.transform(X_validacao.reshape(-1, 2)).reshape(X_validacao.shape)

# Adicionando seis camadas LSTM à rede neural
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, activation='relu', input_shape=(None, 2), return_sequences=True),
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilando o modelo com as métricas corrigidas
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

# Callback para salvar o melhor modelo durante o treinamento
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', verbose=1)

# Adicionando o callback EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Reshape dos dados para serem compatíveis com a entrada da LSTM
X_treinamento = X_treinamento.reshape((X_treinamento.shape[0], 1, X_treinamento.shape[1]))
X_validacao = X_validacao.reshape((X_validacao.shape[0], 1, X_validacao.shape[1]))

# Definindo o tamanho do lote
batch_size = 64  

# Treinando o modelo com o ModelCheckpoint, EarlyStopping e o tamanho do lote
history = model.fit(X_treinamento, y_treinamento, epochs=50, batch_size=batch_size, validation_data=(X_validacao, y_validacao), callbacks=[model_checkpoint, early_stopping])

# Avaliando o modelo
resultado = model.evaluate(X_validacao, y_validacao)
print(f'Loss na validação: {resultado[0]}, MSE: {resultado[1]}, MAE: {resultado[2]}')

# Calcular R²
y_pred = model.predict(X_validacao)
ss_res = np.sum(np.square(y_validacao - y_pred))
ss_tot = np.sum(np.square(y_validacao - np.mean(y_validacao)))
r_squared = 1 - (ss_res / ss_tot)

print(f'R² na validação: {r_squared}')

# Carregando o modelo treinado a partir do melhor checkpoint global
best_model = tf.keras.models.load_model('best_model.h5')

# Carregando os dados de teste
teste_data = load_data('dataset/teste')
X_teste, y_teste = prepare_data(teste_data)

# Normalizando os dados de teste
X_teste = scaler.transform(X_teste.reshape(-1, 2)).reshape(X_teste.shape)

# Reshape dos dados de teste
X_teste = X_teste.reshape((X_teste.shape[0], 1, X_teste.shape[1]))

# Realizando previsões com o melhor modelo
previsoes = best_model.predict(X_teste)

# Avaliando o desempenho no conjunto de teste
resultado_teste = best_model.evaluate(X_teste, y_teste)
print(f'Loss no teste: {resultado_teste[0]}, MSE: {resultado_teste[1]}, MAE: {resultado_teste[2]}')

# Calcular R² de teste
y_pred_teste = best_model.predict(X_teste)
ss_res_teste = np.sum(np.square(y_teste - y_pred_teste))
ss_tot_teste = np.sum(np.square(y_teste - np.mean(y_teste)))
r_squared_teste = 1 - (ss_res_teste / ss_tot_teste)

print(f'R² no teste: {r_squared_teste}')

# Criando gráficos para a perda (loss) durante o treinamento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss no treinamento')
plt.plot(history.history['val_loss'], label='Loss na validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.title('Gráfico de Linhas - Loss no Treinamento e na Validação')
plt.show()

# Criando gráficos de barras para as métricas de avaliação
metrics_names = model.metrics_names[1:]  # Exclude 'loss' from metrics_names

# Plotting MSE, MAE, and R²
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].bar('MSE', resultado[1])
axs[0].set_xlabel('Métrica')
axs[0].set_ylabel('Valor')
axs[0].set_title('MSE na Validação')

axs[1].bar('MAE', resultado[2])
axs[1].set_xlabel('Métrica')
axs[1].set_ylabel('Valor')
axs[1].set_title('MAE na Validação')

axs[2].bar('R²', r_squared)
axs[2].set_xlabel('Métrica')
axs[2].set_ylabel('Valor')
axs[2].set_title('R² na Validação')

plt.tight_layout()
plt.show()
