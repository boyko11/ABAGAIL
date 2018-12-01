package opt.test;

import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;


public class NNOptimizationTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 30, hiddenLayer = 10, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork network = factory.createClassificationNetwork(
            new int[] {inputLayer, hiddenLayer, outputLayer});
    private static NeuralNetworkOptimizationProblem neuralNetworkOptimizationProblem =
            new NeuralNetworkOptimizationProblem(set, network, measure);

    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {

        OptimizationAlgorithm optimizationAlgorithm = null;
        if(args.length == 0) {
            System.out.println("Specify one of the following: RHC, SA, GA: e.g. java NNOptimizationTest RHC");
            return;
        }

        final String optimizationAlgoToUse = args[0];

        switch(optimizationAlgoToUse) {
            case "RHC":
                optimizationAlgorithm = new RandomizedHillClimbing(neuralNetworkOptimizationProblem);
                break;

            case "SA":
                optimizationAlgorithm = new SimulatedAnnealing(1E11, .95, neuralNetworkOptimizationProblem);
                break;

            case "GA":
                optimizationAlgorithm = new StandardGeneticAlgorithm(200, 100, 10,
                        neuralNetworkOptimizationProblem);
                break;

            default:
                optimizationAlgorithm = new RandomizedHillClimbing(neuralNetworkOptimizationProblem);
                break;

        }




        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        train(optimizationAlgorithm, network, optimizationAlgoToUse); //trainer.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        Instance optimalInstance = optimizationAlgorithm.getOptimal();
        network.setWeights(optimalInstance.getData());

        double predicted, actual;
        start = System.nanoTime();
        for(int j = 0; j < instances.length; j++) {
            network.setInputValues(instances[j].getData());
            network.run();

            predicted = Double.parseDouble(instances[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        results +=  "\nResults for " + optimizationAlgoToUse + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println("Iteration " + i + " Error: " + df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[569][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/breast_cancer.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[30]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 30; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
