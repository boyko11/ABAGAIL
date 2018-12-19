package project2_cs7641;


import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import util.linalg.Vector;

import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class NNOptimizationTestBoyko {

    public static void main(String[] args) throws IOException {

        if(args.length == 0) {
            System.out.println("Specify one of the following: RHC, SA, GA: e.g. java NNOptimizationTest RHC");
            return;
        }
        boolean generate_lc_by_training_sizes = false;
        if(args.length > 1) {
            if(args[1].equals("training_sizes")) {
                generate_lc_by_training_sizes = true;
            }
        }

        String learning_curve_by_train_sizes_file = "learning_curve_training_sizes_" + args[0] + ".csv";

        String learning_curve_by_iterations_file = "learning_curve_iterations_" + args[0] + ".csv";
        new File(learning_curve_by_iterations_file).delete();

        int trainingIterations = 1300;
        if(args[0].equals("SA")) {
            trainingIterations = 2500;
        }
        if(args[0].equals("GA")) {
            trainingIterations = 175;
        }
        int iteration_recording_step = 10;
        Integer number_of_records = 569;
        Integer number_of_features = 30;

        double max_percentage_training = 0.8;

        Instance[] instances = initializeInstances(number_of_records, number_of_features);
        List<Double> training_times = new ArrayList<>();
        List<Double> last_train_accuracy = new ArrayList<>();
        List<Double> last_test_accuracy = new ArrayList<>();
//        for(int i = 0; i < 3; i++) {

            //System.out.println("Cross validation run " + i);

            Instance[][] training_testing_split = getTrainingAndTestingInstances(max_percentage_training, instances);

            List<Integer> number_of_training_records = new ArrayList<>();
            List<Double> train_accuracy = new ArrayList<>();
            List<Double> test_accuracy = new ArrayList<>();

            //for learning curve by iterations
            double train_increment = 1.0;
            //for Learning Curve by number of training records
            if(generate_lc_by_training_sizes) {
                train_increment = 1.0 / 10.0;
            }
            for (double current_train_percentage = train_increment; current_train_percentage <= 1.0;
                 current_train_percentage += train_increment) {

                List<Double> train_records_and_train_and_test_accuracy_and_train_time = train_and_test(
                        training_testing_split, trainingIterations, current_train_percentage, args,
                        learning_curve_by_iterations_file, iteration_recording_step);
                number_of_training_records.add(train_records_and_train_and_test_accuracy_and_train_time.get(0).intValue());
                train_accuracy.add(train_records_and_train_and_test_accuracy_and_train_time.get(1));
                test_accuracy.add(train_records_and_train_and_test_accuracy_and_train_time.get(2));
                if(current_train_percentage == 1.0) {
                    training_times.add(train_records_and_train_and_test_accuracy_and_train_time.get(3));
                    last_train_accuracy.add(train_records_and_train_and_test_accuracy_and_train_time.get(1));
                    last_test_accuracy.add(train_records_and_train_and_test_accuracy_and_train_time.get(2));
                }
            }
            if(generate_lc_by_training_sizes) {
                append_to_results_file(learning_curve_by_train_sizes_file, number_of_training_records, train_accuracy,
                        test_accuracy);
            }

//        }

        System.out.println("Training times: ");
        for(Double training_time : training_times) {
            System.out.println(training_time);
        }

        System.out.println("Average Training Time: " + training_times.stream().mapToDouble(time -> time).average());
        System.out.println("Average Training Accuracy: " + last_train_accuracy.stream().mapToDouble(time -> time).average());
        System.out.println("Average Testing Accuracy: " + last_test_accuracy.stream().mapToDouble(time -> time).average());

    }

    private static void append_to_results_file(String results_file, List<Integer> x_axis_values,
                                               List<Double> train_accuracy, List<Double> test_accuracy) throws IOException {

        new File(results_file).createNewFile();
        BufferedWriter bw = new BufferedWriter(new FileWriter(results_file, true));
        bw.write(String.join(",", x_axis_values.stream().map(Object::toString).collect(Collectors.toList())));
        bw.newLine();
        bw.write(String.join(",", train_accuracy.stream().map(score -> BigDecimal.valueOf(score)
                .setScale(3, RoundingMode.HALF_UP).toString()).collect(Collectors.toList())));
        bw.newLine();
        bw.write(String.join(",", test_accuracy.stream().map(score -> BigDecimal.valueOf(score)
                .setScale(3, RoundingMode.HALF_UP).toString()).collect(Collectors.toList())));
        bw.newLine();
        bw.close();
    }


    private static List<Double> train_and_test(Instance[][] training_testing_split, int trainingIterations,
                                               double current_train_percentage, String[] args,
                                               String learning_curve_by_iterations_file, int iteration_recording_step) throws IOException {

        int inputLayer = 30, hiddenLayer = 100, outputLayer = 1;
        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

        ErrorMeasure measure = new SumOfSquaresError();

        Instance[] trainingRecords = training_testing_split[0];
        Instance[] testingRecords = training_testing_split[1];

        int number_of_training_records_to_use = (int) (trainingRecords.length * current_train_percentage);
        Instance[] trainingRecords_to_use = new Instance[number_of_training_records_to_use];
        List<Integer> indices = IntStream.range(0, trainingRecords.length).boxed().collect(Collectors.toList());
        Collections.shuffle(indices, new Random());
        for(int index = 0; index < number_of_training_records_to_use; index++) {
            trainingRecords_to_use[index] = trainingRecords[indices.get(index)];
        }

        DataSet set = new DataSet(trainingRecords);

        BackPropagationNetwork network = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
        NeuralNetworkOptimizationProblem neuralNetworkOptimizationProblem =
                new NeuralNetworkOptimizationProblem(set, network, measure);

        DecimalFormat df = new DecimalFormat("0.000");

        OptimizationAlgorithm optimizationAlgorithm = null;

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
        List<Integer> iterations_toRecord = IntStream.range(0, trainingIterations)
                .filter(x -> x % iteration_recording_step == 0)
                .boxed().collect(Collectors.toList());
        List<List<Double>> accuracy_per_iteration =
                train(optimizationAlgorithm, network, optimizationAlgoToUse, trainingIterations, trainingRecords,
                        testingRecords, measure, iteration_recording_step); //trainer.train();

        append_to_results_file(learning_curve_by_iterations_file, iterations_toRecord, accuracy_per_iteration.get(0),
                accuracy_per_iteration.get(1));

        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        Instance optimalInstance = optimizationAlgorithm.getOptimal();
        network.setWeights(optimalInstance.getData());

        double predicted, actual;
        for(int j = 0; j < trainingRecords_to_use.length; j++) {
            network.setInputValues(trainingRecords_to_use[j].getData());
            network.run();

            predicted = Double.parseDouble(trainingRecords_to_use[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }

        Double training_accuracy = correct/(correct+incorrect)*100;
        String training_results =  "\nTRAINING Results for " + optimizationAlgoToUse + ": \nTraining Records: "
                + current_train_percentage + " : " + number_of_training_records_to_use + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(training_accuracy);
        System.out.println(training_results);

        correct = 0;
        incorrect = 0;
        start = System.nanoTime();
        for(int j = 0; j < testingRecords.length; j++) {
            network.setInputValues(testingRecords[j].getData());
            network.run();

            predicted = Double.parseDouble(testingRecords[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        double testing_accuracy = correct/(correct+incorrect)*100;
        String testing_results =  "\nTESTING Results for " + optimizationAlgoToUse +": \nTesting Records: " + testingRecords.length + "\nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(testing_accuracy) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

        System.out.println(testing_results);

        List<Double> results = new ArrayList<>();
        results.add(new Double(trainingRecords_to_use.length));
        results.add(training_accuracy);
        results.add(testing_accuracy);
        results.add(trainingTime);
        return results;
    }

    private static List<List<Double>> train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName,
                                            int trainingIterations, Instance[] train_records, Instance[] test_records, ErrorMeasure measure,
                                            Integer iteration_step_to_record) {

        List<Double> train_accuracy_per_iteration = new ArrayList<>();
        List<Double> test_accuracy_per_iteration = new ArrayList<>();
        List<List<Double>> accuracy_per_iteration = new ArrayList<>();
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        double previousIterationFitnessValue = 0;
        for(int i = 0; i < trainingIterations; i++) {
            Vector weights_before_train = oa.getOptimal().getData();
            double fitnessValue = oa.train();
            if(previousIterationFitnessValue == fitnessValue) {
                network.setWeights(weights_before_train);
                //continue;
            }
            previousIterationFitnessValue = fitnessValue;

            double train_error = 1/fitnessValue;

            if(i % iteration_step_to_record == 0) {

                double predicted, actual;
                double correct = 0, incorrect = 0;
                for(int j = 0; j < train_records.length; j++) {
                    network.setInputValues(train_records[j].getData());
                    network.run();

                    predicted = Double.parseDouble(train_records[j].getLabel().toString());
                    actual = Double.parseDouble(network.getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                }
                Double training_accuracy = correct/(correct+incorrect)*100;
                train_accuracy_per_iteration.add(training_accuracy);


                double test_error = 0;
                predicted = 0;
                actual = 0;
                correct =  0;
                incorrect = 0;
                for(int j = 0; j < test_records.length; j++) {
                    network.setInputValues(test_records[j].getData());
                    network.run();
                    predicted = Double.parseDouble(test_records[j].getLabel().toString());
                    actual = Double.parseDouble(network.getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                    Instance output = test_records[j].getLabel(), example = new Instance(network.getOutputValues());
                    example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                    test_error += measure.value(output, example);
                }

                Double test_accuracy = correct/(correct+incorrect)*100;
                test_accuracy_per_iteration.add(test_accuracy);

                DecimalFormat df = new DecimalFormat("0.000");
                System.out.println("Iteration " + i + " Train Error: " + df.format(train_error) +
                        " Test  Error: " + df.format((test_error/test_records.length)*train_records.length));
            }
        }
        accuracy_per_iteration.add(train_accuracy_per_iteration);
        accuracy_per_iteration.add(test_accuracy_per_iteration);
        return accuracy_per_iteration;
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
        Collections.shuffle(indices, new Random());
        for(int index = 0; index < number_of_training_records; index++) {
            train_test_split[0][index] = allInstances[indices.get(index)];
        }
        for(int index = number_of_training_records, index1 = 0; index < allInstances.length; index++, index1++) {
            train_test_split[1][index1] = allInstances[indices.get(index)];
        }
        return train_test_split;
    }

}
