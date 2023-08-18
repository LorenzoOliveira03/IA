"""
Implementação da Regressão Linear utilizando o método do gradiente descendente.

Para completar esta tarefa, você deve implementar o código que falta para 3 funções abaixo.
- Uma função de custo, para calcular o erro entre o valor previsto pelo modelo e o valor objetivo.
- Uma função de gradiente, para calcular o valor da derivada da função de custo para os parâmetros w e b.
- Uma função de gradiente descendente, para executar o método do gradiente descendente.

Também são fornecidos um pequeno dataset de casas visto em aula e valores para testar a execução do gradiente descendente.
"""


#biblioteca para lidar com números em python
import numpy as np 
#biblioteca para criar visualizações em gráficos
import matplotlib.pyplot as plt 

#Primeira função a ser desenvolvida é a função para computar o custo do erro do modelo no dataset.

#Lembrando que erro para um exemplo do dataset é dado pela seguinte equação:
#erro_individual = (f(x_i) - y_i)² 
#E o custo total é dado pela seguinte equação:
#(1/2*m) * (somatório de i=1 até m de erro_individual)
#onde
#i = cada exemplo do dataset
#m = total de exemplos do dataset
#f(x_i) = w*x_i+b

def funcao_custo(x,y,w,b):
    """
    Função para computar o custo
    Parâmetros:
        x (ndarray (m,)) : Dataset, m exemplos com uma única feature
        y (ndarray (m,)) : valores objetivos
        w (escalar)      : parâmetro w do modelo    
        b (scalar)       : parâmetro b do modelo
        
    Retorno:
        custo_total (escalar): custo total dos parâmetros w e b no dataset X
    """
    custo_total = 0
    m = x.shape[0]
    #código inicia aqui
    def f(x,y,w,b):
        return w*x+b
    def erro_individual(x,y,w,b):
        return (f(x,y,w,b)-y)**2
        def custo_total(x,y,w,b):
            return (1/(2*m))*erro_individual(x,y,w,b)
    #código termina aqui
    return custo_total

#A função gradiente calcula o valor das derivadas da função de custo em relação a w e b.
#A derivada da função de custo em relação a w é: 
#(1/m) * (somátório de i=1 até m de ((f(x_i) - y_i)*x_i))
#A derivada da função de custo em relação a b é: 
#(1/m) * (somátório de i=1 até m de ((f(x_i) - y_i)))
#onde
#i = cada exemplo do dataset
#m = total de exemplos do dataset
def gradiente(x,y,w,b):
    """
    Função para computar o gradiente com parâmetros específicos w e b
    Parâmetros:
        x (ndarray (m,)) : Dataset, m exemplos com uma única feature
        y (ndarray (m,)) : valores objetivos
        w (escalar)      : parâmetro w do modelo    
        b (escalar)       : parâmetro b do modelo
        
    Retorno:
        derivada_w (escalar): valor da derivada da função de custo em relação a w
        derivada_b (escalar): valor da derivada da função de custo em relação a b
    """
    m = x.shape[0]
    derivada_w = 0
    derivada_b = 0

    #código inicia aqui

    #código termina aqui

    return derivada_w, derivada_b

#A função **gradiente_descendente** minimiza o custo para os parâmetros w e b.
#O gradiente descendente funciona como abaixo.

#Repetir até convergir:
#{
#w = w - learning_rate * derivada_w
#b = b - learning_rate * derivada_b
#}

#Lembrando que w e b precisam ser atualizados simultaneamente.
#Para o nosso caso, teremos um novo parâmetro, chamado de épocas, 
#que define um número máximo de iterações para o método, para evitar que fique tentando convergir infinitamente.


def gradiente_descendente(x,y,w_inicial,b_inicial,learning_rate,epocas):
    """
    Função para encontrar os parâmetros w e b que minimizam o custo de erro do nosso modelo
    Parâmetros:
        x (ndarray (m,))         : Dataset, m exemplos com uma única feature
        y (ndarray (m,))         : valores objetivos
        w_inicial (escalar)      : parâmetro w inicial para começar o aprendizado
        b_inicial (escalar)      : parâmetro b inicial para começar o aprendizado
        learning_rate (escalar)  : taxa de aprendizado do nosso algoritmo (geralmente menor que 0.01)
        epocas(escalar)          : numero máximo de iterações para o método do gradiente descente
        
    Retorno:
        w (escalar): melhor valor encontrado para w
        b (escalar): melhor valor encontrado para b
    """
    m = x.shape[0]
    w = w_inicial
    b = b_inicial

    #código inicia aqui

    #código encerra aqui
    return w,b

#função auxiliar
def teste_de_regressao(x,y):
    """
    Função para testar uma regressão linear de uma variável
    Parâmetros:
        x (ndarray (m,))         : Dataset, m exemplos com uma única feature
        y (ndarray (m,))         : valores objetivos
    """
    #Valores para testar nosso algoritmo de gradiente descente
    w_inicial=0
    b_inicial=0
    learning_rate = 0.00001
    epocas=1000  
    w, b = gradiente_descendente(x,y,w_inicial,b_inicial,learning_rate,epocas)

    print(f"Valor encontrador para w:{w} e para b:{b}")
    
    #Para mostrar em um gráfico, podemos utilizar a biblioteca pyplot
    #Primeiro definimos os nomes dos eixos do gráfico
    plt.ylabel("Custo da casa") #nome do eixo y
    plt.xlabel("Tamanho da casa") #nome do deixo x

    #Depois plotamos todos exemplos do nosso dataset como círculos
    #O parâmetro 'o' diz que queremos que cada exemplo do dataset seja impresso como um círculo
    plt.plot(x,y,'o') 

    #Aqui passamos o dataset x e todos valores previstos pelo nosso modelo, para poder desenhar a reta do nosso modelo no gráfico.
    y_hats = np.zeros(x.shape[0]) #inicializamos um vetor de tamanho m (total de exemplos)
    y_hats = x * w + b #calculamos o y_hat para cada exemplo do dataset, y_hats é um vetor!

    #Aqui plotamos o nosso dataset em custo de casas vs tamanho
    plt.plot(x,y_hats)

    #Para prever um valor que não existe no dataset, podemos testar utilizando a equação linear com os parâmetros que encontramos:
    nova_casa = 150
    valor_previsto = nova_casa * w + b #atenção, aqui valor_previsto é um escalar, não um vetor!

    #Abaixo vamos também colocar no gráfico 
    plt.plot(nova_casa,valor_previsto,'x') #plotamos o valor previsto como um 'x' no gráfico

    #chamada para mostrar o gráfico
    plt.show()

#Você pode testar o modelo algoritmo de regressão com uma variável utilizando o dataset de casas da aula e a função teste_de_regressao

x=np.array([45,60,100,200,300]) #dataset de casas com uma feature
y=np.array([200,300,500,1000,1500]) #valor objetivo do dataset de casas
teste_de_regressao(x,y)