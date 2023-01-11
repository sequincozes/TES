package avaliacao_pratica;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

// https://github.com/sequincozes/TES
public class PrincipalCSNetExtendido {

    public static double normalClass = 0; // isso aqui representa as classes com o valor N
    public static double faultClass = 1;

    public static void main(String args[]) throws Exception {
//        extrairMatrizCSNetEstendido();
//        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53});
//        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53});
        gerarIWSS();


    }

    public static void gerarIWSS() throws Exception {
        int[] allFeatures = new int[]{
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
//                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47
//                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51
//                5, 6, 7, 8, 9, 10, 11, 12, 13, 14,  25, 34, 43, 52, 53
        };

        iwss(allFeatures, true);
        geraCSV(allFeatures);
//        trainAndTestConsideringFaults(new IBk(), features, train, test);
    }


    private static void iwss(int[] features, boolean summarize) throws Exception {
        // Training data
        Instances train = Auxiliar.lerDataset("dataset-c2_20train.csv");
        train = Auxiliar.selecionaFeatures(train, features);

        // Testing data
        Instances test = Auxiliar.lerDataset("dataset-c2_80test.csv");
        test = Auxiliar.selecionaFeatures(test, features);

        // Processing
        System.out.println("-- Resultado com IWSS: ");
        runIwss(train, test, new IBk(), summarize);

    }

    public static void runIwss(Instances train, Instances test,
                               AbstractClassifier classificador, boolean summarize) throws Exception {

        double melhorF1Score = 0;
        int[] melhorConjunto = {};
        ArrayList<String> progresso = new ArrayList<>();

        // Busca Sequencial
//        System.out.println("Starting IWSS with "+train.numAttributes()+" training attributes.");
//        System.out.println("Starting IWSS with "+train.numAttributes()+" testing attributes.");
        for (int i = 1; i < train.numAttributes(); i++) {

            int[] selecao = new int[melhorConjunto.length + 1];

            for (int j = 0; j < melhorConjunto.length; j++) {
                selecao[j] = melhorConjunto[j];
            }

            selecao[melhorConjunto.length] = i;
//            System.out.println("Selecao = " + Arrays.toString(selecao));
            if (!summarize)
                System.out.println("Melhor Conjunto = " + Arrays.toString(melhorConjunto) + " => " + melhorF1Score);

            //////

            Instances treinoReduced = Auxiliar.selecionaFeatures(new Instances(train), selecao);

            Instances testReduced = Auxiliar.selecionaFeatures(new Instances(test), selecao);
            AbstractClassifier classificadorTreinado = Auxiliar.construir(treinoReduced, classificador);
            double f1Score = trainAndTestConsideringFaults(classificadorTreinado, selecao, treinoReduced, testReduced, false);

            /////
//            double f1Score = trainAndTestConsideringFaults(classificador, selecao, treino, teste, true);
            if (!summarize)
                System.out.println(Arrays.toString(selecao) + " => " + f1Score);
            if (f1Score > melhorF1Score) {
                melhorF1Score = f1Score;
                melhorConjunto = selecao;
                System.out.println("Novo melhor resultado: " + Arrays.toString(melhorConjunto) + " com " + melhorF1Score);
            } else {
                if (!summarize)
                    System.out.println("Não houve melhoras!");
            }
            progresso.add(i + ";");
            if (!summarize) {
                System.out.println("Finishing IWSS with " + train.numAttributes());
                System.out.println("------------------");
            }

        }

        //** Final Result output **//
        Instances treinoReduced = Auxiliar.selecionaFeatures(new Instances(train), melhorConjunto);
        Instances testReduced = Auxiliar.selecionaFeatures(new Instances(test), melhorConjunto);
        AbstractClassifier classificadorTreinado = Auxiliar.construir(treinoReduced, classificador);
        trainAndTestConsideringFaults(classificador, melhorConjunto, treinoReduced, testReduced, true);

    }

    private static void gerarMatrizConfusao(int[] features) throws Exception {
        Instances datasetTreinamento = Auxiliar.lerDataset("dataset-c2_20train.csv");
        Instances teste = Auxiliar.lerDataset("dataset-c2_80test.csv");

        datasetTreinamento = Auxiliar.selecionaFeatures(datasetTreinamento, features);
        teste = Auxiliar.selecionaFeatures(teste, features);

        AbstractClassifier classificadorC = new J48(); // nova instância de um classificador qualquer
        AbstractClassifier classificadorTreinado = Auxiliar.construir(datasetTreinamento, classificadorC);
        // Resultados
        float VP = 0; // quando o IDS diz que está acontecendo um ataque, e realmente está
        float VN = 0; // quando o IDS diz que NÃO está acontecendo um ataque, e realmente NÃO está
        float FP = 0; // quando o IDS diz que está acontecendo um ataque, PORÉM NÃO ESTÁ
        float FN = 0; // quando o IDS diz que NÃO está acontecendo um ataque, PORÉM ESTÁ!
        long beginNano = System.nanoTime();

        int[][] confusionMatrix = new int[5][5];
        for (int i = 0; i < teste.size(); i++) { //percorre cada uma das amostras de teste
            try {
                Instance testando = teste.instance(i);
                double resultado = Auxiliar.testarInstancia(classificadorTreinado, testando);
                double esperado = testando.classValue();

                if (resultado == normalClass || resultado == faultClass) { // sabemos que é negativo (O IDS nao detectou ataque)
                    if (esperado == normalClass || esperado == faultClass) {
                        VN = VN + 1; // o IDS acertou (pode ter confundido falta com normal, mas nao confundiu com ataque)
                    } else {
                        FN = FN + 1;
                    }
                } else { // sabemos que é positivo (o IDS detectou algo)
                    if (esperado == normalClass || esperado == faultClass) {
                        FP = FP + 1;
                    } else {
                        VP = VP + 1;
                    }
                }

                // Confusion matrix:
                confusionMatrix[(int) esperado][(int) resultado] = confusionMatrix[(int) esperado][(int) resultado] + 1;
            } catch (ArrayIndexOutOfBoundsException a) {
                System.err.println("Erro: " + a.getLocalizedMessage());
                System.err.println("DICA: " + "Tem certeza que o número de classes está definido corretamente?");
                System.exit(1);
            } catch (Exception e) {
                System.err.println("Erro: " + e.getLocalizedMessage());
                System.exit(1);
            }
        }
        long endNano = System.nanoTime();
        float totalNano = Float.valueOf(endNano - beginNano) / 1000; // converte para microssegundos
        float acuracia = (VP + VN) * 100 / (VP + VN + FP + FN); // quantos acertos o IDS teve
        float recall = (VP * 100) / (VP + FN); // quantas vezes eu acertei dentre as vezes REALMENTE ESTAVA acontecendo um ataque
        float precision = (VP * 100) / (VP + FP); // quantas vezes eu acertei dentre as vezes que eu DISSE que estava acontecendo
        float f1score = 2 * (recall * precision) / (recall + precision);
        System.out.println(Arrays.toString(features) + ';' + f1score / 100 + ";" + precision / 100 + ";" + recall / 100 + ';' + acuracia / 100 + ';' + totalNano + ';' + totalNano / teste.size());
//        System.out.println("FP: " + FP);
//        System.out.println("VP: " + VP);
//        System.out.println("VN: " + VN);
//        System.out.println("FN: " + FN);

        for (int iEsperado = 0; iEsperado < confusionMatrix.length; iEsperado++) {
            for (int iResultado = 0; iResultado < confusionMatrix.length; iResultado++) {
                System.out.print(confusionMatrix[iEsperado][iResultado] + ";");
            }
            System.out.println();
        }

    }

    public static void extrairMatrizCSNetEstendido() throws Exception {
        System.out.println("Features" + ';' + "F1-Score" + ";" + "Precision" + ";" + "Recall" + ';' + "Acuracia" + ';' + "TotalNano" + ';' + "Time/sample");
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53});
        gerarMatrizConfusao(new int[]{6, 7, 4, 14, 5, 3, 53, 2, 8, 1, 21, 30, 34, 25, 39, 44, 15, 16, 48, 43, 52, 13, 12, 45});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4});
        gerarMatrizConfusao(new int[]{5, 6, 7, 8});
        gerarMatrizConfusao(new int[]{9, 10, 11, 12, 13, 14});
        gerarMatrizConfusao(new int[]{15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});
        gerarMatrizConfusao(new int[]{16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        gerarMatrizConfusao(new int[]{25, 34, 43, 52, 53});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25, 34, 43, 52, 53});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 25, 34, 43, 52, 53});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53, 13});
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53, 14});
        gerarMatrizConfusao(new int[]{1, 3, 4, 5, 6, 7, 53});
    }

    public static void extrairResultadosCSNetEstendido() throws Exception {
        System.out.println("Features" + ';' + "F1-Score" + ";" + "Precision" + ";" + "Recall" + ';' + "Acuracia" + ';' + "TotalNano" + ';' + "Time/sample");
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53});
        geraCSV(new int[]{6, 7, 4, 14, 5, 3, 53, 2, 8, 1, 21, 30, 34, 25, 39, 44, 15, 16, 48, 43, 52, 13, 12, 45});
        geraCSV(new int[]{1, 2, 3, 4});
        geraCSV(new int[]{5, 6, 7, 8});
        geraCSV(new int[]{9, 10, 11, 12, 13, 14});
        geraCSV(new int[]{15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});
        geraCSV(new int[]{16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        geraCSV(new int[]{25, 34, 43, 52, 53});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25, 34, 43, 52, 53});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 25, 34, 43, 52, 53});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53, 13});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53, 14});
        geraCSV(new int[]{1, 3, 4, 5, 6, 7, 53});
    }

    public static void geraCSV(int[] features) throws Exception {

        // Prepara datasets
        Instances train = Auxiliar.lerDataset("dataset-c2_20train.csv");
        Instances test = Auxiliar.lerDataset("dataset-c2_80test.csv");
        train = Auxiliar.selecionaFeatures(train, features);
        test = Auxiliar.selecionaFeatures(test, features);

        trainAndTestConsideringFaults(new IBk(), features, train, test, true);

    }

    private static double trainAndTestConsideringFaults(AbstractClassifier classifier, int[] features, Instances datasetTreinamento, Instances teste, boolean printCSV) throws Exception {

        // Treina
        AbstractClassifier classificadorTreinado = Auxiliar.construir(datasetTreinamento, classifier);

        // Testa
        float VP = 0; // quando o IDS diz que está acontecendo um ataque, e realmente está
        float VN = 0; // quando o IDS diz que NÃO está acontecendo um ataque, e realmente NÃO está
        float FP = 0; // quando o IDS diz que está acontecendo um ataque, PORÉM NÃO ESTÁ
        float FN = 0; // quando o IDS diz que NÃO está acontecendo um ataque, PORÉM ESTÁ!
        long beginNano = System.nanoTime();

        int[][] confusionMatrix = new int[5][5];
        for (int i = 0; i < teste.size(); i++) { //percorre cada uma das amostras de teste
            try {
                Instance testando = teste.instance(i);
                double resultado = Auxiliar.testarInstancia(classificadorTreinado, testando);
                double esperado = testando.classValue();

                if (resultado == normalClass || resultado == faultClass) { // sabemos que é negativo (O IDS nao detectou ataque)
                    if (esperado == normalClass || esperado == faultClass) {
                        VN = VN + 1; // o IDS acertou (pode ter confundido falta com normal, mas nao confundiu com ataque)
                    } else {
                        FN = FN + 1;
                    }
                } else { // sabemos que é positivo (o IDS detectou algo)
                    if (esperado == normalClass || esperado == faultClass) {
                        FP = FP + 1;
                    } else {
                        VP = VP + 1;
                    }
                }

                // Confusion matrix:
                confusionMatrix[(int) esperado][(int) resultado] = confusionMatrix[(int) esperado][(int) resultado] + 1;
            } catch (ArrayIndexOutOfBoundsException a) {
                a.printStackTrace();
                System.err.println("Erro: " + a.getLocalizedMessage());
                System.err.println("DICA: " + "Tem certeza que o número de classes está definido corretamente?");
                System.exit(1);
            } catch (Exception e) {
                System.err.println("Erro: " + e.getLocalizedMessage());
                System.exit(1);
            }
        }
        long endNano = System.nanoTime();
        float totalNano = Float.valueOf(endNano - beginNano) / 1000; // converte para microssegundos
        float acuracia = (VP + VN) * 100 / (VP + VN + FP + FN); // quantos acertos o IDS teve
        float recall = (VP * 100) / (VP + FN); // quantas vezes eu acertei dentre as vezes REALMENTE ESTAVA acontecendo um ataque
        float precision = (VP * 100) / (VP + FP); // quantas vezes eu acertei dentre as vezes que eu DISSE que estava acontecendo
        float f1score = 2 * (recall * precision) / (recall + precision);
        if (printCSV) {
            System.out.println(Arrays.toString(features) + ';' + f1score / 100 + ";" + precision / 100 + ";" + recall / 100 + ';' + acuracia / 100 + ';' + totalNano + ';' + totalNano / teste.size());
        }
//        System.out.println("DEBUG: Amostras GeraCSV: " + (VP + VN + FP + FN) + " / Features: " + Arrays.toString(features) + "/ F1-Score: " + f1score);
//        System.out.println("Outros TESTE:");
//        System.out.println(FeatureSelection.testaConjunto(datasetTreinamento, teste, new IBk(), features));
//        System.out.println(Auxiliar.classificarInstancias(classificadorTreinado, teste));
        return f1score;
    }
}
