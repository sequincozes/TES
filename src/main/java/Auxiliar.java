import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class Auxiliar {

    public static double testarInstancia(AbstractClassifier classificador, Instance amostra) throws Exception {
        return classificador.classifyInstance(amostra);
    }

    public static AbstractClassifier construir(Instances treinamento, AbstractClassifier classificador) throws Exception {
        classificador.buildClassifier(treinamento);
        return classificador;
    }

    public static Instances lerDataset(String dataset) throws IOException {
        FileReader fr = new FileReader(dataset);
        BufferedReader br = new BufferedReader(fr);
        Instances datasetInstances = new Instances(br);
        datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);
        return datasetInstances;
    }

    public static Instances selecionaFeatures(Instances amostras, int[] features) {
        Arrays.sort(features);
        for (int i = amostras.numAttributes() - 1; i > 0; i--) {
            if (amostras.numAttributes() <= features.length) {
                System.err.println("O número de features precisa ser maior que o filtro.");
                System.exit(1);
                return amostras;
            }
            boolean deletar = true;
            for (int j : features) {
                if (i == j) {
                    deletar = false;
                }
            }
            if (deletar) {
                amostras.deleteAttributeAt(i - 1);
            }
        }
        amostras.setClassIndex(amostras.numAttributes() - 1);
        return new Instances(amostras);
    }

}
