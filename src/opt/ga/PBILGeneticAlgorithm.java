package opt.ga;

import dist.DiscreteDistribution;
import opt.OptimizationAlgorithm;
import shared.Instance;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


/**
 * Genetic algorithms are pretty stupid.
 * This is based on the version in Andrew Moore's tutorial.
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class PBILGeneticAlgorithm extends OptimizationAlgorithm {

    /**
     * The random number generator
     */
    private static final Random random = new Random();

    /**
     * The population size
     */
    private int populationSize;

    /**
     * The number of population to mate
     * each time step
     */
    private int toMate;

    /**
     * The number of population to mutate
     * each time step
     */
    private double mutationRate;

    /**
     * The population
     */
    private Instance[] population;

    /**
     * The values of the population
     */
    private double[] values;

    /**
     * Make a new genetic algorithm
     * @param populationSize the size
     * @param toMate the number to mate each iteration
     * @param mutationRate the number to mutate each iteration
     * @param gap the problem to solve
     */
    public PBILGeneticAlgorithm(int populationSize, int toMate, double mutationRate, GeneticAlgorithmProblem gap) {
        super(gap);
        this.toMate = toMate;
        this.mutationRate = mutationRate;
        this.populationSize = populationSize;
        population = new Instance[populationSize];
        for (int i = 0; i < population.length; i++) {
            population[i] = gap.random();
        }
        values = new double[populationSize];
        for (int i = 0; i < values.length; i++) {
            values[i] = gap.value(population[i]);
        }
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        GeneticAlgorithmProblem ga = (GeneticAlgorithmProblem) getOptimizationProblem();
        double[] probabilities = new double[population.length];
        // calculate probability distribution over the population
        double sum = 0;
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = values[i];
            sum += probabilities[i];
        }
        if (Double.isInfinite(sum)) {
            return sum;
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
        DiscreteDistribution dd = new DiscreteDistribution(probabilities);

        // make the children
        double[] newValues = new double[populationSize];
        Instance[] newPopulation = new Instance[populationSize];
        for (int i = 0; i < toMate; i++) {
            // pick the mates
            Instance a = population[dd.sample(null).getDiscrete()];
            Instance b = population[dd.sample(null).getDiscrete()];
            // make the kid
            newPopulation[i] = ga.mate(a, b);
            newValues[i] = -1;
        }
        // elite for the rest
        for (int i = toMate; i < newPopulation.length; i++) {
            int j = dd.sample(null).getDiscrete();
            newPopulation[i] = (Instance) population[j].copy();
            newValues[i] = values[j];
        }
        // mutate
        //get mutation indices
        int total_number_of_bits = populationSize * population[1].size();
        int total_number_of_bits_to_mutate = (int) Math.round(new Double(total_number_of_bits) * mutationRate);

        List<Integer> range = IntStream.range(0, total_number_of_bits).boxed().collect(Collectors.toList());
        Collections.shuffle(range);
        List<Integer> bits_to_mutate = range.subList(0, total_number_of_bits_to_mutate);
        for(int index = 0; index < bits_to_mutate.size(); index++) {

            int instance_to_mutate = bits_to_mutate.get(index)/population[1].size();
            int bit_to_mutate = bits_to_mutate.get(index) % population[1].size();
            //System.out.println("instance_to_mutate and bit: " + instance_to_mutate + " " + bit_to_mutate);
            double mutatedValue = (newPopulation[instance_to_mutate].getData().get(bit_to_mutate) + 1) % 2;
            newPopulation[instance_to_mutate].getData().set(bit_to_mutate, mutatedValue);
            newValues[instance_to_mutate] = -1;

        }
//        for (int i = 0; i < toMutate; i++) {
//        	int j = random.nextInt(newPopulation.length);
//            ga.mutate(newPopulation[j]);
//            newValues[j] = -1;
//        }

        // calculate the new values
        for (int i = 0; i < newValues.length; i++) {
            if (newValues[i] == -1) {
                newValues[i] = ga.value(newPopulation[i]);
            }
        }
        // the new generation
        population = newPopulation;
        values = newValues;
        return sum / populationSize;
    }

    /**
     * @see OptimizationAlgorithm#getOptimalData()
     */
    public Instance getOptimal() {
        GeneticAlgorithmProblem ga = (GeneticAlgorithmProblem) getOptimizationProblem();
        double bestVal = values[0];
        int best = 0;
        for (int i = 1; i < population.length; i++) {
            double value = values[i];
            if (value > bestVal) {
                bestVal = value;
                best = i;
            }
        }
        return population[best];
    }

}
