package opt.test;

import java.util.Arrays;

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
 * @author Daniel Cohen dcohen@gatech.edu
 * @version 1.0
 */
public class TwoColorsTest {
    /** The number of colors */
    private static final int k = 11;
    /** The N value */
    private static final int N = 100*k;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, k+1);
        EvaluationFunction ef = new TwoColorsEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double optimumValue = N - 2;
        System.out.println("Optimum value: " + optimumValue);
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 2000);
//        fit.setConvergence_value(optimumValue);
//        fit.setEvaluationFunction(ef);
//        fit.train();
//        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        
//        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
//        fit = new FixedIterationTrainer(sa, 4000);
//        fit.setConvergence_value(optimumValue);
//        fit.setEvaluationFunction(ef);
//        fit.train();
//        System.out.println("SA: " + ef.value(sa.getOptimal()));
//
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(3000, 600, 0, gap);
        fit = new FixedIterationTrainer(ga, 4000);
        fit.setConvergence_value(optimumValue);
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("SA: " + ef.value(ga.getOptimal()));
//
//        MIMIC mimic = new MIMIC(100, 20, pop);
//        fit = new FixedIterationTrainer(mimic, 500);
//        fit.setConvergence_value(optimumValue);
//        fit.setEvaluationFunction(ef);
//        fit.train();
//        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
    }
}
