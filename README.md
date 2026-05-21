<h1><strong>Modelos de Machine Learning Aplicados ao Mercado Financeiro</strong></h1>
<h4><strong>RESUMO</strong></h4>
<p align="justify">&emsp;O presente projeto consiste no desenvolvimento de uma ferramenta para a previsão de preços de ações em tempo real. Na análise comparativa, foram implementados dois modelos de machine learning em Python, com foco nas bibliotecas TensorFlow/Keras e Scikit-learn: Redes Neurais Recorrentes Simples (Simple Recurrent Neural Networks - SimpleRNN) e Long Short-Term Memory (LSTM).</p>
<p align="justify">&emsp;Adicionalmente, técnicas como o GridSearch e normalização foram aplicadas para um estudo e análise mais aprofundados dos resultados. Para reforçar a robustez do projeto, foram utilizadas versões do conjunto de dados (dataset) com a inserção de ruído artificial. O estudo permitiu concluir que o uso de modelos de machine learning para a predição do valor de ações é bastante promissor, e este projeto alcançou resultados satisfatórios de acordo com as métricas de avaliação adotadas.</p>
<hr>
<h4><strong>ÍNDICE</strong></h4>
<a href="#1_">1. Introdução</a>
<br>
<a href="#2_">2. Metodologia</a>
<br>
&emsp;<a href="#2.1_">2.1 Dataset's</a>
<br>
&emsp;<a href="#2.2_">2.2 Gridsearch</a>
<br>
&emsp;<a href="#2.3_">2.3 Modelo SimpleRNN</a>
<br>
&emsp;<a href="#2.4_">2.4 Modelo LSTM</a>
<br>
<a href="#3_">3. Resutlados</a>
<br>
<a href="#4_">4. Conclusão</a>
<br>
<hr>
<h4><strong>1. INTRODUÇÃO</strong></h4>
<a id="1_"></a>
<p align="justify">&emsp;O mercado financeiro tem recebido muita atenção recentemente, visto que ele tem se apresentado como uma oportunidade para investidores e empresas. Dentre a grande variedade de ativos financeiros pertencentes a este mercado, ações ganham destaque. A previsão do comportamento de ativos, como ações, em janelas curtas de tempo é de grande interesse para operadores, investidores e desenvolvedores de sistemas de apoio à decisão e é neste contexto em que este projeto se insere.</p>
<br>
<p align="justify">&emsp;Neste projeto foi desenvolvido em sistema preditivo de curto prazo (5 minutos à frente) para a ação VALE3 em tempo real utilizando Redes Neurais Recorrentes (SimpleRNN e LTSM). Este sistema é capaz de prever a tendência de preços do ativo, com simulação gráfica em tempo real das previsões e avaliação rigorosa por métricas padronizadas. O intervalo utilizado para treinamento da rede é de 8 dias até o momento atual, sendo o modelo calibrado com janelas de 1 minuto.</p>
<br>
<p align="justify">&emsp;A escolha da empresa para o projeto teve base no fato da Vale ser globalmente presente e atuar em um poderoso mercado, o de mineração e produção de minério de ferro Além disso, a empresa tem forte impacto no Brasil, estando em funcionamento desde 1942 e construiu uma base sólida com a passagem do tempo, persistindo mesmo após crises recentes.</p>
<hr>

<h4><strong>2. METODOLOGIA</strong></h4>
<a id="2_"></a>
<p align="justify">&emsp;O projeto foi desenvolvido utilizando Python como linguagem de programação combinado com o Jupyter Notebook e fazendo uso TensorFlow (Keras) juntamente com o Scikit-learn e outras bibliotécas como Pandas, NumPy e Matplotlib. Neste capítulo será abordado o desdobramento prático e teórico do projeto, bem como a implementação dos modelos SimpleRNN e LSTM.</p>

<h5><strong>2.1 Dataset's</strong></h5>
<a id="2.1_"></a>
<p align="justify">&emsp;A base de dados foi retirada do API pública yfinance, disponibilizada pelo Yahoo. Os intervalos entre dados escolhido foi de 1 minuto com início a partir de 8 dias anteriores a execução do algoritmo (03/07/2025 até 11/07/2025) e fim no momento em que o algoritmo foi executado. Dessa forma, pode-se ter uma ideia do poder preditivo da ferramenta desenvolvida com poucos dias de dados registrados.</p>
<br>
<p align="justify">&emsp;A fim de explorar ao máximo os modelos de machine learning implementados, uma nova versão do dataset foi criada. Com base na base de dados original, foi aplicado um ruído gaussiano de média 0 (zero) e desvio padrão 0,2 a nova versão, assim, pode-se adquirir um modelo mais robusto e generalista. A comparação dos datasets pde ser vista na Imagem (1) logo abaixo.</p>

<p align="center">Imagem (1)<br>
<img align="center" src="https://github.com/user-attachments/assets/2cb7a505-e147-4e24-8be7-0986d4cf5e5b" alt=0></p>

<p align="justify">&emsp;Ambos os datasets foram submetidos a uma normalização, utilizou-se a ferramenta MinMaxScaler para tal. A normalização ocorreu após a divisão entre treino e teste destes datasets, a proporção utilizada foi de 20% para teste e 80% para treino, dessa forma, normalizamos os dados com base no conjuto de treino e apenas aplicamos a transfomação ao conjunto de teste.</p>

<h5><strong>2.2 Gridsearch</strong></h5>
<a id="2.2_"></a>
<p align="justify">&emsp;O gridsearch é uma ferramente muito utilizada quando o assunto é a busca por ótimos parâmetros de um modelo de machine learning. No total, 4 (quatro) gridsearchs foram realizados, um para cada variação do dataset de cada modelo, os parâmetros testados podem ser vistos na Imagem (2).</p>

<p align="center">Imagem (2)<br>
<img src="https://github.com/user-attachments/assets/f61c6818-4827-40f5-960e-5fc5a9b88825" alt=0></p>

<p align="justify">&emsp;Escolheu-se os parâmetros baseado no valor do Erro Médio Quadrático (EMQ) gerado pelo teste. A partir do gridsearch, chega-se aos melhores parâmetros para cada modelo em cada dataset, os parâmetros escolhidos para cada caso será abordado em sua seção específica.</p>

<h5><strong>2.3 Modelo Simple</strong></h5>
<a id="2.3_"></a>
<p align="justify">&emsp;O Exmplo de um resultado por dataset do gridsearch para o modelo SimpleRNN pode ser vistos na Imagem (3).</p>

<p align="center">Imagem (3)<br>
<img src="https://github.com/user-attachments/assets/d7d778fc-691d-4119-9d15-7a46fd04b160" alt=0></p>

<p align="justify">&emsp;Abaixo, na Imagem (4), pode ser observado o resultado da aplicação do modelo SimpleRNN.</p>

<p align="center">Imagem (4)<br>
<img width="1018" height="398" alt="image5" src="https://github.com/user-attachments/assets/003c565a-ab17-4ef1-9af1-acbfb94fa24f" />

<h5><strong>2.4 Modelo LSTM</strong></h5>
<p align="justify">&emsp;Abaixo, na Imagem (5), pode ser observado o resultado da aplicação do modelo LSTM</p>
  <img width="1018" height="398" alt="image1" src="https://github.com/user-attachments/assets/65105a0f-0a2f-445f-a89c-ff7d37f5e490" />
