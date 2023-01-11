package avaliacao_pratica;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.Arrays;

import static avaliacao_pratica.PrincipalCSNetExtendido.geraCSV;

// https://github.com/sequincozes/TES
public class Principal {

    public static double normalClass = 0; // isso aqui representa as classes com o valor N
    public static double faultClass = 1;

    public static void main(String args[]) throws Exception {
//        Instances datasetTreinamento = Auxiliar.lerDataset("/home/silvio/Downloads/ds2.csv");
//        new FeatureSelection().rankFeatures(datasetTreinamento,179);


        extrairResultadosCSNetEstendido();

//        anterior();

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

    public static void extrairResultadosCSNetEstendido() throws Exception {
        System.out.println("Features" + ';' + "F1-Score" + ";" + "Precision" + ";" + "Recall" + ';' + "Acuracia" + ';' + "TotalNano" + ';' + "Time/sample");
// GOOSE, GOOSE++, SV&GOOSE
        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
//Voltage, GOOSE++, SV&GOOSE,
        geraCSV(new int[]{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});

//Current, GOOSE++, SV&GOOSE,
        geraCSV(new int[]{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});

//Application, GOOSE++, SV&GOOSE,
        geraCSV(new int[]{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25, 34, 43, 52, 53});


//        SV&GOOSE com GOSSE++
//        geraCSV(new int[]{5, 6, 7, 8, 9, 10, 11, 12, 13, 14});


        //        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,  52, 53});
//        geraCSV(new int[]{6,7,4,14,5,3,53,2,8,1,21,30,34,25,39,44,15,16,48,43,52,13,12,45});
//        geraCSV(new int[]{1, 2, 3, 4});
//        geraCSV(new int[]{5, 6, 7, 8});
//        geraCSV(new int[]{9, 10, 11, 12, 13, 14});
//        geraCSV(new int[]{15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});
//        geraCSV(new int[]{16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
//        geraCSV(new int[]{25, 34, 43, 52, 53});
//        geraCSV(new int[]{1, 2, 3, 4,5, 6, 7, 8,9, 10, 11, 12, 13, 14});
//        geraCSV(new int[]{1, 2, 3, 4,5, 6, 7, 8,9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});
//        geraCSV(new int[]{1, 2, 3, 4,5, 6, 7, 8,9, 10, 11, 12, 13, 14, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
//        geraCSV(new int[]{1, 2, 3, 4,5, 6, 7, 8,9, 10, 11, 12, 13, 14, 25, 34, 43, 52, 53});
//        geraCSV(new int[]{1, 2, 3, 4,5, 6, 7, 8,9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
//        geraCSV(new int[]{1, 2, 3, 4,5, 6, 7, 8,9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53});
//        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8});
//        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47});
//        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
//        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 25, 34, 43, 52, 53});
//        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51});
//        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53});
//        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53, 13});
//        geraCSV(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47, 16, 21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51, 25, 34, 43, 52, 53, 14});
//        geraCSV(new int[]{1, 3, 4, 5, 6, 7, 53});
    }

    public static void anterior() throws Exception {

        // Leitura de datasets para a memória
        Instances datasetTreinamento = Auxiliar.lerDataset("dataset-c2_20train.csv");
        Instances datasetTestes = Auxiliar.lerDataset("dataset-c2_80test.csv");

//        new avaliacao_pratica.FeatureSelection().rankFeatures(datasetTreinamento, 53);

//        int[] features = new int[]{1, 2, 3, 4,5, 6, 7, 8,9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 26, 27, 28, 29, 35, 36, 37, 38, 44, 45, 46, 47};
//        datasetTreinamento = Auxiliar.selecionaFeatures(datasetTreinamento, features);
//        datasetTestes = Auxiliar.selecionaFeatures(datasetTestes, features);

//        // Construção do modelo de classificação (treinamento)
//        AbstractClassifier classificador = new J48(); // nova instância de um classificador qualquer
//        AbstractClassifier classificadorTreinado = Auxiliar.construir(datasetTreinamento, classificador);

//        FeatureSelection.iwss(datasetTreinamento, datasetTestes, classificador);

        new FeatureSelection().rankFeatures(datasetTreinamento, 10);

//
//        // Testes e processamento de resultados
//        processaResultados(classificadorTreinado, datasetTestes);
    }

    public static void processaResultados(AbstractClassifier classificador, Instances teste) {
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
                double resultado = Auxiliar.testarInstancia(classificador, testando);
                double esperado = testando.classValue();
                if (resultado == esperado) { // já sabemos que o resultado é verdadeiro
                    if (resultado == normalClass || resultado == faultClass) {
                        VN = VN + 1; // O IDS diz que NÃO está acontecendo um ataque, e realmente NÃO está
                    } else {
                        VP = VP + 1; // o IDS diz que está acontecendo um ataque, e realmente está
                    }
                } else { // sabemos que é um "falso"
                    if (resultado == normalClass || resultado == faultClass) {
                        FN = FN + 1; // o IDS diz que NÃO está acontecendo um ataque, PORÉM ESTÁ!
                    } else {
                        FP = FP + 1; // o IDS diz que está acontecendo um ataque, PORÉM NÃO ESTÁ
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
        System.out.println(" ### Tempo de Processamento");
        System.out.println("     - Tempo total de processamento: " + totalNano + " microssegundos");
        System.out.println("     - Tempo de processamento por amostra: " + totalNano / teste.size() + " microssegundos");

        System.out.println(" ### Desempenho na classificação");
        float acuracia = (VP + VN) * 100 / (VP + VN + FP + FN); // quantos acertos o IDS teve
        float recall = (VP * 100) / (VP + FN); // quantas vezes eu acertei dentre as vezes REALMENTE ESTAVA acontecendo um ataque
        float precision = (VP * 100) / (VP + FP); // quantas vezes eu acertei dentre as vezes que eu DISSE que estava acontecendo
        float f1score = 2 * (recall * precision) / (recall + precision);
        System.out.println("     - VP: " + VP + ", VN: " + VN + ", FP: " + FP + ", VN: " + FN);
        System.out.println("     - F1-Score: " + f1score + "%");
        System.out.println("     - Recall: " + recall + "%");
        System.out.println("     - Precision: " + precision + "%");
        System.out.println("     - Accuracy: " + acuracia + "%");


    }

    public static void geraCSV(int[] features) throws Exception {
        Instances datasetTreinamento = Auxiliar.lerDataset("dataset-c2_20train.csv");
        Instances teste = Auxiliar.lerDataset("dataset-c2_80test.csv");

        datasetTreinamento = Auxiliar.selecionaFeatures(datasetTreinamento, features);
        teste = Auxiliar.selecionaFeatures(teste, features);

        AbstractClassifier classificadorC = new IBk();//J48(); // nova instância de um classificador qualquer
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
                if (resultado == esperado) { // já sabemos que o resultado é verdadeiro
                    if (resultado == normalClass || resultado == faultClass) {
                        VN = VN + 1; // O IDS diz que NÃO está acontecendo um ataque, e realmente NÃO está
                    } else {
                        VP = VP + 1; // o IDS diz que está acontecendo um ataque, e realmente está
                    }
                } else { // sabemos que é um "falso"
                    if (resultado == normalClass || resultado == faultClass) {
                        FN = FN + 1; // o IDS diz que NÃO está acontecendo um ataque, PORÉM ESTÁ!
                    } else {
                        FP = FP + 1; // o IDS diz que está acontecendo um ataque, PORÉM NÃO ESTÁ
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
