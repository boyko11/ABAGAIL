package opt.test;

import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import util.linalg.Vector;


public class NNOptimizationTest {

    private static final Integer number_of_records = 569;
    private static final Integer number_of_features = 30;
    private static Instance[] instances = initializeInstances(number_of_records, number_of_features);

    private static final double percentageTraining = 0.8;
    private static Instance[][] training_testing_split = getTrainingAndTestingInstances(percentageTraining, instances);

    private static int inputLayer = 30, hiddenLayer = 100, outputLayer = 1, trainingIterations = 2000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(training_testing_split[0]);

    private static BackPropagationNetwork network = factory.createClassificationNetwork(
            new int[] {inputLayer, hiddenLayer, outputLayer});
    private static NeuralNetworkOptimizationProblem neuralNetworkOptimizationProblem =
            new NeuralNetworkOptimizationProblem(set, network, measure);

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
        for(int j = 0; j < training_testing_split[0].length; j++) {
            network.setInputValues(training_testing_split[0][j].getData());
            network.run();

            predicted = Double.parseDouble(training_testing_split[0][j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }

        String training_results =  "\nTRAINING Results for " + optimizationAlgoToUse + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(correct/(correct+incorrect)*100);
        System.out.println(training_results);

        correct = 0;
        incorrect = 0;
        start = System.nanoTime();
        for(int j = 0; j < training_testing_split[1].length; j++) {
            network.setInputValues(training_testing_split[1][j].getData());
            network.run();

            predicted = Double.parseDouble(training_testing_split[1][j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        String testing_results =  "\nTESTING Results for " + optimizationAlgoToUse + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

        System.out.println(testing_results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        double previousIterationFitnessValue = 0;
        for(int i = 0; i < trainingIterations; i++) {
            Vector weights_before_train = oa.getOptimal().getData();
            double fitnessValue = oa.train();
            if(previousIterationFitnessValue == fitnessValue) {
//                System.out.println("No Change after Train. " + i);
//                System.out.println("Resetting weights and skipping training.");
                network.setWeights(weights_before_train);
                continue;
            }
            previousIterationFitnessValue = fitnessValue;

            double train_error = 0;
            for(int j = 0; j < training_testing_split[0].length; j++) {
                network.setInputValues(training_testing_split[0][j].getData());
                network.run();

                Instance output = training_testing_split[0][j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                train_error += measure.value(output, example);
            }
            if(i % 100 == 0) {

                double test_error = 0;
                for(int j = 0; j < training_testing_split[1].length; j++) {
                    network.setInputValues(training_testing_split[1][j].getData());
                    network.run();

                    Instance output = training_testing_split[1][j].getLabel(), example = new Instance(network.getOutputValues());
                    example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                    test_error += measure.value(output, example);
                }

                System.out.println("Iteration " + i + " Train Error: " + df.format(train_error/training_testing_split[0].length) +
                        " Test  Error: " + df.format(test_error/training_testing_split[1].length));
            }
        }
    }

    private static Instance[] initializeInstances(Integer number_of_records, Integer number_of_attributes) {

        double[][][] attributes = new double[number_of_records][][];

        try {
//            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/breast_cancer.csv")));
            BufferedReader br = new BufferedReader(new FileReader(
                    new File("src/opt/test/breast_cancer_zero_mean_unit_var.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[number_of_attributes]; // 30 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < number_of_attributes; j++)
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
    private static Instance[][] getTrainingAndTestingInstances(Double percentTraining, Instance[] allInstances) {

        int number_of_training_records = (int) (allInstances.length * percentTraining);
        int number_of_testing_records = allInstances.length - number_of_training_records;
        Instance[][] train_test_split = new Instance[2][];
        train_test_split[0] = new Instance[number_of_training_records];
        train_test_split[1] = new Instance[number_of_testing_records];
        List<Integer> indices = IntStream.range(0, allInstances.length).boxed().collect(Collectors.toList());
        Collections.shuffle(indices);
        for(int index = 0; index < number_of_training_records; index++) {
            train_test_split[0][index] = allInstances[indices.get(index)];
        }
        for(int index = number_of_training_records, index1 = 0; index < allInstances.length; index++, index1++) {
            train_test_split[1][index1] = allInstances[indices.get(index)];
        }
        return train_test_split;
    }

}
