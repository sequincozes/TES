package avaliacao_pratica;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.Arrays;

// https://github.com/sequincozes/TES
public class PrincipalCSNetExtendido {

    public static double normalClass = 0; // isso aqui representa as classes com o valor N
    public static double faultClass = 1;

    public static void main(String args[]) throws Exception {
//        extrairResultadosCSNetEstendido();
        gerarMatrizConfusao(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53});
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53});

//        Instances datasetTreinamento = Auxiliar.lerDataset("dataset-c2_20train.csv");
//        Instances datasetTestes = Auxiliar.lerDataset("dataset-c2_80test.csv");
//        datasetTreinamento.addAll(datasetTestes);
//        new avaliacao_pratica.FeatureSelection().rankFeatures(datasetTreinamento, 53);
        //        Construção do modelo de classificação (treinamento)
//        AbstractClassifier classificador = new J48(); // nova instância de um classificador qualquer
//        AbstractClassifier classificadorTreinado = Auxiliar.construir(datasetTreinamento, classificador);
//
//        FeatureSelection.iwss(datasetTreinamento, datasetTestes, classificadorTreinado);

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
                    if (esperado == normalClass || esperado == faultClass ){
                        VN = VN + 1; // o IDS acertou (pode ter confundido falta com normal, mas nao confundiu com ataque)
                    } else {
                        FN = FN + 1;
                    }
                } else { // sabemos que é positivo (o IDS detectou algo)
                    if (esperado == normalClass || esperado == faultClass ) {
                       FP = FP +1;
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
        System.out.println("FP: " + FP);
        System.out.println("VP: " + VP);
        System.out.println("VN: " + VN);
        System.out.println("FN: " + FN);

        for (int iEsperado = 0; iEsperado < confusionMatrix.length; iEsperado++) {
            for (int iResultado = 0; iResultado < confusionMatrix.length; iResultado++) {
                System.out.print(confusionMatrix[iEsperado][iResultado] + ",");
            }
            System.out.println();
        }

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
                    if (esperado == normalClass || esperado == faultClass ){
                        VN = VN + 1; // o IDS acertou (pode ter confundido falta com normal, mas nao confundiu com ataque)
                    } else {
                        FN = FN + 1;
                    }
                } else { // sabemos que é positivo (o IDS detectou algo)
                    if (esperado == normalClass || esperado == faultClass ) {
                        FP = FP +1;
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
    }
}