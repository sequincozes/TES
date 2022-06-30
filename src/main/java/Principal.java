import weka.classifiers.AbstractClassifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class Principal {

    public static double normalClass = 0;

    public static void main(String args[]) throws Exception {
        // Leitura de datasets para a memória
        Instances datasetTreinamento = Auxiliar.lerDataset("ereno1ktrain.arff");
        Instances datasetTestes = Auxiliar.lerDataset("ereno1ktest.arff");

        datasetTreinamento = Auxiliar.selecionaFeatures(datasetTreinamento, new int[]{1, 2, 3, 4, 5, 6});
        datasetTestes = Auxiliar.selecionaFeatures(datasetTestes, new int[]{1, 2, 3, 4, 5, 6});

        // Construção do modelo de classificação (treinamento)
        AbstractClassifier classificador = new IBk(); // nova instância de um classificador qualquer
        AbstractClassifier classificadorTreinado = Auxiliar.construir(datasetTreinamento, classificador);

        // Testes e processamento de resultados
        processaResultados(classificadorTreinado, datasetTestes);
    }


    public static void processaResultados(AbstractClassifier classificador, Instances teste) {
        // Resultados
        float VP = 0;
        float VN = 0;
        float FP = 0;
        float FN = 0;
        long beginNano = System.nanoTime();

        int[][] confusionMatrix = new int[2][2];
        for (int i = 0; i < teste.size(); i++) {
            try {
                Instance testando = teste.instance(i);
                double resultado = classificador.classifyInstance(testando);
                double esperado = testando.classValue();
                if (resultado == esperado) {
                    if (resultado == normalClass) {
                        VN = VN + 1;
                    } else {
                        VP = VP + 1;
                    }
                } else { // bad prediction
                    if (resultado == normalClass) {
                        FN = FN + 1;
                    } else {
                        FP = FP + 1;
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
        System.out.println("     - Tempo de processamento por amostra: " + Float.valueOf(endNano - beginNano) / teste.size() + " microssegundos");

        System.out.println(" ### Desempenho na classificação");
        float acuracia = (VP + VN) * 100 / (VP + VN + FP + FN);
        float recall = (VP * 100) / (VP + FN);
        float precision = (VP * 100) / (VP + FP);
        float f1score = 2 * (recall * precision) / (recall + precision);
        System.out.println("     - VP: " + VP + ", VN: " + VN + ", FP: " + FP + ", VN: " + FN);
        System.out.println("     - F1-Score: " + f1score + "%");
        System.out.println("     - Recall: " + recall + "%");
        System.out.println("     - Precision: " + precision + "%");
        System.out.println("     - Accuracy: " + acuracia + "%");


    }
}
