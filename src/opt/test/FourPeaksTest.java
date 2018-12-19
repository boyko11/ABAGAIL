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
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 100;
    /** The t value */
    private static final int T = 17;
    
    public static void main(String[] args) {

        System.out.println("FourPeaksTest start.");

        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double optimumValue = (2*N - (T + 1));
        System.out.println("Optimum value to find: " + optimumValue);


        System.out.println("RHC: ");
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 3500);
        fit.setConvergence_value(optimumValue);
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("Optimum after training: " + ef.value(rhc.getOptimal()));
        System.out.println("-------------------------------------");

        System.out.println("SA: ");
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        fit = new FixedIterationTrainer(sa, 5000);
        fit.setConvergence_value(optimumValue);
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("Optimum after training: " + ef.value(sa.getOptimal()));
        System.out.println("-------------------------------------");

        System.out.println("GA: ");
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(10000, 5000, 200, gap);
        fit = new FixedIterationTrainer(ga, 20000);
        fit.setConvergence_value(optimumValue);
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("Optimum after training: " + ef.value(ga.getOptimal()));
        System.out.println("-------------------------------------");
/*
        PBILGeneticAlgorithm pbil_ga = new PBILGeneticAlgorithm(2000, 1600, 0.001, gap);
        fit = new FixedIterationTrainer(pbil_ga, 5000);
        fit.train();
        System.out.println("PBIL GA: " + ef.value(pbil_ga.getOptimal()));*/


        System.out.println("MIMIC: ");
        MIMIC mimic = new MIMIC(5000, 200, pop);
        fit = new FixedIterationTrainer(mimic, 100);
        fit.setConvergence_value(optimumValue);
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("Optimum after training: " + ef.value(mimic.getOptimal()));
        System.out.println("-------------------------------------");

        System.out.println("FourPeaksTest end.");
    }
}
