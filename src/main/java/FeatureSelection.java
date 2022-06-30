import weka.attributeSelection.*;
import weka.core.Instances;

public class FeatureSelection {

    public static void rankComGR(Instances treino) throws Exception {
        for (int i = 0; i < treino.get(0).numAttributes(); i++) {
            System.out.println("Feature: " + i + ", valor: " + calculaOneR(treino, i));
        }
    }
    public static void rankComGainRatio(Instances treino) throws Exception {
        for (int i = 0; i < treino.get(0).numAttributes(); i++) {
            System.out.println("Feature: " + i + ", valor: " + calcularaGainRatio(treino, i));
        }
    }
    public static void rankComInfoGain(Instances treino) throws Exception {
        for (int i = 0; i < treino.get(0).numAttributes(); i++) {
            System.out.println("Feature: " + i + ", valor: " + calcularaInfoGain(treino, i));
        }
    }

    public static double calcularaGainRatio(Instances instances, int featureIndice) throws Exception {
        GainRatioAttributeEval ase = new GainRatioAttributeEval();
        ase.buildEvaluator(instances);
        return ase.evaluateAttribute(featureIndice);
    }

    public static double calcularaInfoGain(Instances instances, int featureIndice) throws Exception {
        InfoGainAttributeEval ase = new InfoGainAttributeEval();
        ase.buildEvaluator(instances);
        return ase.evaluateAttribute(featureIndice);
    }

    public static double calculaOneR(Instances dataset, int featureIndice) throws Exception {
        OneRAttributeEval ase = new OneRAttributeEval();
        ase.buildEvaluator(dataset);
        return ase.evaluateAttribute(featureIndice);
    }

}
