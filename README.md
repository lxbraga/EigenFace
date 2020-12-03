<<<<<<< HEAD
> # EigenVargas

*Lucas Geraldo dos Santos Braga* & *João Pedro Giordani Donasolo*
		22/11/2020

# Objetivo:

Reconhecer a face de professores da EMAp com base em fotos achadas em buscas na Web. Para isso, utilizaremos Eigenfaces. A seguir, o passo-a-passo seguido no script:

## Passo a passo:


0. Selecionar nosso dataset, preenchido com imagens de professores, e completo com imagens de outro dataset encontrado na web.
1. Preparar o dataset, recortando as faces dos professores conforme necessário e colocando fundo branco.
2. Dessas imagens, pegar uma para treinamento e deixar as outras como teste (feita renomeando as de teste com números no final).
3. Recortar as imagens em um tamanho padronizado e convertê-las para preto e branco 8bits(feito em outro notebook).
4. Achatar as imagens, de matrizes para vetores
5. Calcular a face média das imagens de treinamento.
6. Normalizar as imagens de treinamento, de cada uma delas, subtrair a face média
7. Calcular a matriz de convariância,
8. Extrair os autovetores,
9. Calcular as Eigenfaces - Autovetores x Faces normalizadas
10. Calcular os pesos das imagens.  

# Fundamentação teórica

A aceitação ou recusa de uma imagem como face ou não se dá pela mensuração da sua diferença da imagem em contraste ao que o programa considera como face, isto é, o erro da projeção da imagem de entrada sobre o espaço vetorial gerado pelas eigenfaces.

O procedimento é feito colpasando as informações das imagens (pixels) em vetores unidimensionais, que irão compor as linhas da matriz. Com a matriz em mãos, podemos realizar sua decomposição SVD para descobrir seus autovalores e autovetores. Os autovetores, presentes na matriz U, terão o número de pixels de uma imagem, e, após o redimensionamento, podem ser visualizados como faces, isto é, "eigenfaces" (fazendo uma alusão ao termo "eigenvectors", ou autovetores, em inglês).

Por fim, o reconhecimento ou não de uma imagem de entrada como face se dará pelo erro da projeção no espaço gerado pelas eigenfaces. No caso deste ser numericamente maior que um limite imposto, será recusado. Caso seja menor, será aceito. De forma complementar, o nosso programa tenta identificar a qual imagem de treinamento o input se relaciona.


# Resultados

Tivemos duas abordagens similares impostas pela natureza do dataset. Na primeira abordagem, tentantos puramente reconhecer faces, e, se possível atrelá-las a um individuo, usando uma maçã como objeto de controle. Obtivemos taxa de sucesso de **11/18**. Na segunda, tentamos dar match dos inputs com os professores corretos, obtendo assim uma taxa de sucesso de **8/12**
