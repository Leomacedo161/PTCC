import random
import csv
import math
import os

def gerar_equacao(operadores, valor_minimo, valor_maximo):
    valor1 = random.randint(valor_minimo, valor_maximo)
    valor2 = random.randint(valor_minimo, valor_maximo)
    operador = random.choice(operadores)

    if operador == '+':
        resultado = valor1 + valor2
    elif operador == '-':
        resultado = valor1 - valor2
    elif operador == '*':
        resultado = valor1 * valor2
    elif operador == '/':
        if valor2 == 0:
            resultado = 0  # Lidar com a divisão por zero
        else:
            resultado = valor1 / valor2

    return valor1, valor2, operador, round(resultado)

def operador_para_nome(operador):
    operador_dict = {
        '+': 'adicao',
        '-': 'subtracao',
        '*': 'multiplicacao',
        '/': 'divisao'
    }
    return operador_dict.get(operador, 'desconhecido')

def gerar_equacao_csv(operador, num_equacoes, nome_arquivo):
    operadores = ['+', '-', '*', '/']
    if operador not in operadores:
        return

    dados = []
    for _ in range(num_equacoes):
        valor1, valor2, op, resultado = gerar_equacao([operador], 1, 999)  # Intervalo de 1 a 999
        dados.append([valor1, valor2, operador, resultado])

    nome_arquivo_sem_extensao = operador_para_nome(operador)
    
    # Dividir os dados em conjuntos de treinamento, validação e teste
    total_dados = len(dados)
    tamanho_treino = math.ceil(0.75 * total_dados)
    tamanho_validacao = math.ceil(0.15 * total_dados)
    tamanho_teste = total_dados - tamanho_treino - tamanho_validacao

    dados_treino = dados[:tamanho_treino]
    dados_validacao = dados[tamanho_treino:tamanho_treino + tamanho_validacao]
    dados_teste = dados[tamanho_treino + tamanho_validacao:]

    if nome_arquivo:
        # Criar pastas para treinamento, validação e teste
        os.makedirs('dataset/treinamento', exist_ok=True)
        os.makedirs('dataset/validacao', exist_ok=True)
        os.makedirs('dataset/teste', exist_ok=True)

        # Criar um arquivo CSV com as colunas "valor1, valor2, operador, resultado"
        with open(f'{nome_arquivo_sem_extensao}.csv', 'w', newline='') as arquivo_csv:
            escritor_csv = csv.writer(arquivo_csv)
            escritor_csv.writerow(["valor1", "valor2", "operador", "resultado"])
            escritor_csv.writerows(dados)

        # Criar arquivos separados para treinamento, validação e teste
        with open(f'dataset/treinamento/{nome_arquivo_sem_extensao}.csv', 'w', newline='') as arquivo_treino:
            escritor_treino = csv.writer(arquivo_treino)
            escritor_treino.writerow(["valor1", "valor2", "operador", "resultado"])
            escritor_treino.writerows(dados_treino)

        with open(f'dataset/validacao/{nome_arquivo_sem_extensao}.csv', 'w', newline='') as arquivo_validacao:
            escritor_validacao = csv.writer(arquivo_validacao)
            escritor_validacao.writerow(["valor1", "valor2", "operador", "resultado"])
            escritor_validacao.writerows(dados_validacao)

        with open(f'dataset/teste/{nome_arquivo_sem_extensao}.csv', 'w', newline='') as arquivo_teste:
            escritor_teste = csv.writer(arquivo_teste)
            escritor_teste.writerow(["valor1", "valor2", "operador", "resultado"])
            escritor_teste.writerows(dados_teste)

if __name__ == '__main__':
    num_equacoes_por_operador = 1000
    operadores = ['+', '-', '*', '/']

    for operador in operadores:
        gerar_equacao_csv(operador, num_equacoes_por_operador, operador)
