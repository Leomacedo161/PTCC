import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Carregando os dados de treinamento
def load_data(folder, operation):
    file_path = f'{folder}/{operation}.csv'
    df = pd.read_csv(file_path)
    return df

treinamento_data = {operation: load_data('dataset/treinamento', operation) for operation in ['adicao', 'subtracao', 'multiplicacao', 'divisao']}
validacao_data = {operation: load_data('dataset/validacao', operation) for operation in ['adicao', 'subtracao', 'multiplicacao', 'divisao']}

# Preparando os dados
def prepare_data(df):
    inputs = []
    outputs = []
    for _, row in df.iterrows():
        first_number = row['valor1']
        second_number = row['valor2']
        result = row['resultado']
        inputs.append([first_number, second_number])
        outputs.append(result)
    return np.array(inputs), np.array(outputs)

# Normalizando os dados usando a mesma instância para treinamento e validação
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Modelo de Rede Neural
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(None, 2), return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='linear')  
])

# Definindo o tamanho do lote
batch_size = 128

# Ajuste da taxa de aprendizado com agendamento
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9))
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# Callbacks para salvar o melhor modelo e early stopping
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinamento do modelo para cada operação
history = {}
best_models = {}
metrics_results = {}

for operation in ['adicao', 'subtracao', 'multiplicacao', 'divisao']:
    X_treinamento_op, y_treinamento_op = prepare_data(treinamento_data[operation])
    X_validacao_op, y_validacao_op = prepare_data(validacao_data[operation])

    # Normalizando os dados
    X_treinamento_scaled = scaler_X.fit_transform(X_treinamento_op)
    X_validacao_scaled = scaler_X.transform(X_validacao_op)
    y_treinamento_scaled = scaler_y.fit_transform(y_treinamento_op.reshape(-1, 1)).reshape(-1)
    y_validacao_scaled = scaler_y.transform(y_validacao_op.reshape(-1, 1)).reshape(-1)

    X_treinamento_op = X_treinamento_scaled.reshape((X_treinamento_op.shape[0], 1, X_treinamento_op.shape[1]))
    X_validacao_op = X_validacao_scaled.reshape((X_validacao_op.shape[0], 1, X_validacao_op.shape[1]))

    history[operation] = model.fit(X_treinamento_op, y_treinamento_scaled, epochs=5, batch_size=batch_size,
                                   validation_data=(X_validacao_op, y_validacao_scaled),
                                   callbacks=[model_checkpoint, early_stopping])

    # Carregando o modelo treinado a partir do melhor checkpoint global
    best_models[operation] = tf.keras.models.load_model('best_model.h5')

    # Avaliando o desempenho no conjunto de teste
    y_pred_scaled = best_models[operation].predict(X_validacao_op)

    # Invertendo a escala das previsões
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    # Calculando as métricas
    mse = mean_squared_error(y_validacao_op, y_pred)
    mae = mean_absolute_error(y_validacao_op, y_pred)
    r2 = r2_score(y_validacao_op, y_pred)

    metrics_results[operation] = {'mse': mse, 'mae': mae, 'r2': r2}

    # Gráficos lado a lado para MSE, MAE e R²
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Gráfico de Barras - MSE
    axes[0].bar(['MSE'], [metrics_results[operation]['mse']], label=f'MSE ({operation})')
    axes[0].set_xlabel('Métricas')
    axes[0].set_ylabel('Valores')
    axes[0].legend()
    axes[0].set_title(f'Gráfico de Barras - MSE ({operation})')

    # Gráfico de Barras - MAE
    axes[1].bar(['MAE'], [metrics_results[operation]['mae']], label=f'MAE ({operation})')
    axes[1].set_xlabel('Métricas')
    axes[1].set_ylabel('Valores')
    axes[1].legend()
    axes[1].set_title(f'Gráfico de Barras - MAE ({operation})')

    # Gráfico de Barras - R²
    axes[2].bar(['R²'], [metrics_results[operation]['r2']], label=f'R² ({operation})')
    axes[2].set_xlabel('Métricas')
    axes[2].set_ylabel('Valores')
    axes[2].legend()
    axes[2].set_title(f'Gráfico de Barras - R² ({operation})')

    plt.tight_layout()

    # Exibindo os gráficos
    plt.show()

    # Gráfico de Linhas - Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history[operation].history['loss'], label=f'Loss no treinamento ({operation})')
    plt.plot(history[operation].history['val_loss'], label=f'Loss na validação ({operation})')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Gráfico de Linhas - Loss ({operation})')
    plt.show()
