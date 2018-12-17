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
    private static final int N = 50;
    /** The t value */
    private static final int T = 5;
    
    public static void main(String[] args) {
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
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.setConvergence_value(optimumValue);
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.setConvergence_value(2*N - (T + 1));
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("SA: " + ef.value(sa.getOptimal()));

        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20000, 10000, 200, gap);
        fit = new FixedIterationTrainer(ga, 20000);
        fit.setConvergence_value(optimumValue);
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("GA: " + ef.value(ga.getOptimal()));
//
////        PBILGeneticAlgorithm pbil_ga = new PBILGeneticAlgorithm(2000, 1600, 0.001, gap);
////        fit = new FixedIterationTrainer(pbil_ga, 5000);
////        fit.train();
////        System.out.println("PBIL GA: " + ef.value(pbil_ga.getOptimal()));
//
        MIMIC mimic = new MIMIC(5000, 2000, pop);
        fit = new FixedIterationTrainer(mimic, 2000);
        fit.setConvergence_value(optimumValue);
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
    }
}
