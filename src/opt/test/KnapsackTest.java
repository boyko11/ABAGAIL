package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
         MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {


        double rhc_sum = 0;
        double sa_sum = 0;
        double ga_sum = 0;
        double mimic_sum = 0;

        double rhc_train_time_sum = 0;
        double sa_train_time_sum = 0;
        double ga_train_time_sum = 0;
        double mimic_train_time_sum = 0;

        for(int a = 0; a < 10; a++) {
            int[] copies = new int[NUM_ITEMS];
            Arrays.fill(copies, COPIES_EACH);
            double[] values = new double[NUM_ITEMS];
            double[] weights = new double[NUM_ITEMS];
            for (int i = 0; i < NUM_ITEMS; i++) {
                values[i] = random.nextDouble() * MAX_VALUE;
                weights[i] = random.nextDouble() * MAX_WEIGHT;
            }
            int[] ranges = new int[NUM_ITEMS];
            Arrays.fill(ranges, COPIES_EACH + 1);

            EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);

            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();
            rhc_train_time_sum += fit.getTrain_time();
            double rhc_opt_value = ef.value(rhc.getOptimal());
            rhc_sum += rhc_opt_value;
            System.out.println("RHC: " + rhc_opt_value);

            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            sa_train_time_sum += fit.getTrain_time();
            double sa_opt_value = ef.value(sa.getOptimal());
            sa_sum += sa_opt_value;
            System.out.println("SA: " + sa_opt_value);
//
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            ga_train_time_sum += fit.getTrain_time();
            double ga_opt_value = ef.value(ga.getOptimal());
            ga_sum += ga_opt_value;
            System.out.println("GA: " + ga_opt_value);

            MIMIC mimic = new MIMIC(200, 100, pop);
            fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            mimic_train_time_sum += fit.getTrain_time();
            double mimic_opt_value = ef.value(mimic.getOptimal());
            mimic_sum += mimic_opt_value;
            System.out.println("MIMIC: " + mimic_opt_value);
        }

        System.out.println("Avg Opt Value:");
        System.out.println("RHC: " + rhc_sum/10.0);
        System.out.println("SA: " + sa_sum/10.0);
        System.out.println("GA: " + ga_sum/10.0);
        System.out.println("MIMIC: " + mimic_sum/10.0);
        System.out.println("--------------------------");
        System.out.println("Avg Train Time:");
        System.out.println("RHC: " + rhc_train_time_sum/10.0);
        System.out.println("SA: " + sa_train_time_sum/10.0);
        System.out.println("GA: " + ga_train_time_sum/10.0);
        System.out.println("MIMIC: " + mimic_train_time_sum/10.0);
    }

}
