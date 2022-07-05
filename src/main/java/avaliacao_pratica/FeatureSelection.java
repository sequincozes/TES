package avaliacao_pratica;

import weka.attributeSelection.*;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class FeatureSelection {

    static double melhorF1Score = 13.606710433959961;
    static int[] melhorConjunto = {1, 60};

    public static void iwss(Instances treino, Instances teste,
                            AbstractClassifier classificador) throws Exception {
        // Busca Sequencial
        int[] selecao = new int[]{1,60}; // features para serem selecionadas

        double f1Score = testaConjunto(treino, teste, classificador, selecao);

        System.out.println("Resultado: "+f1Score);
        System.out.println("Melhor Resultado: "+melhorF1Score);

        if (f1Score > melhorF1Score) {
            melhorF1Score = f1Score;
            melhorConjunto = selecao;
            System.out.println("Temos um novo melhor resultado!");
        } else {
            System.out.println("Não houve melhoras!");
        }


    }

    public void rankFeatures(Instances treino, int pontoCorte) throws Exception {
        FeatureAvaliada[] allFeatures = new FeatureAvaliada[treino.numAttributes()];
        for (int i = 0; i < treino.numAttributes(); i++) {
            double peso = calcularaInfoGain(treino, i); // estamos usando o Gain Ratio, mas pode-se usar outros
            allFeatures[i] = new FeatureAvaliada(peso, i + 1);
        }

        quickSort(allFeatures, 0, allFeatures.length - 1);
        FeatureAvaliada[] filter = new FeatureAvaliada[pontoCorte];
        int count = 0;
        int posRank = 0;
        for (int j = allFeatures.length; j > allFeatures.length - pontoCorte; j--) {
            filter[count++] = allFeatures[j - 1];
            System.out.println(posRank++ + " - Feature " + filter[count - 1].indiceFeature + ", peso: " + filter[count - 1].valorFeature);
        }
    }

    private static double calcularaGainRatio(Instances instances, int featureIndice) throws Exception {
        GainRatioAttributeEval ase = new GainRatioAttributeEval();
        ase.buildEvaluator(instances);
        return ase.evaluateAttribute(featureIndice);
    }

    private static double calcularaInfoGain(Instances instances, int featureIndice) throws Exception {
        InfoGainAttributeEval ase = new InfoGainAttributeEval();
        ase.buildEvaluator(instances);
        return ase.evaluateAttribute(featureIndice);
    }

    private static double calculaOneR(Instances dataset, int featureIndice) throws Exception {
        OneRAttributeEval ase = new OneRAttributeEval();
        ase.buildEvaluator(dataset);
        return ase.evaluateAttribute(featureIndice);
    }

    public static void quickSort(FeatureAvaliada[] vetor, int inicio, int fim) {
        if (inicio < fim) {
            int posicaoPivo = separar(vetor, inicio, fim);
            quickSort(vetor, inicio, posicaoPivo - 1);
            quickSort(vetor, posicaoPivo + 1, fim);
        }
    }

    private static int separar(FeatureAvaliada[] vetor, int inicio, int fim) {
        FeatureAvaliada pivo = vetor[inicio];
        int i = inicio + 1, f = fim;
        while (i <= f) {
            if (vetor[i].getValorFeature() <= pivo.getValorFeature()) {
                i++;
            } else if (pivo.getValorFeature() < vetor[f].getValorFeature()) {
                f--;
            } else {
                FeatureAvaliada troca = vetor[i];
                vetor[i] = vetor[f];
                vetor[f] = troca;
                i++;
                f--;
            }
        }
        vetor[inicio] = vetor[f];
        vetor[f] = pivo;
        return f;
    }

    class FeatureAvaliada {

        public double valorFeature;
        public int indiceFeature;

        public FeatureAvaliada(double valorFeature, int indiceFeature) {
            this.valorFeature = valorFeature;
            this.indiceFeature = indiceFeature;
        }

        public double getValorFeature() {
            return valorFeature;
        }

        public int getIndiceFeature() {
            return indiceFeature;
        }
    }

    public static double testaConjunto(Instances treino, Instances teste,
                                       AbstractClassifier classificador, int[] selecao) throws Exception {
        // Seleção de features
        Instances treinoReduzido =
                Auxiliar.selecionaFeatures(treino, selecao);
        Instances testReduzido =
                Auxiliar.selecionaFeatures(teste, selecao);

        // Treinar classificador
        AbstractClassifier classificadorTreinado =
                Auxiliar.construir(treinoReduzido, classificador);

        // Testar classificador
        double f1score = Auxiliar.classificarInstancias(
                classificadorTreinado, testReduzido);

        return f1score;
    }

}
