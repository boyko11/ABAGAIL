package opt.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesTest {
    /** The n value */
    private static final int N = 80;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        FixedIterationTrainer fit = new FixedIterationTrainer(null, 200);


//        int rhc_iter_sum = 0;
//        long rhc_time_sum = 0l;
//        int sa_iter_sum = 0;
//        long sa_time_sum = 0l;
//
//
//        for(int i = 0; i < 1000; i++) {
//            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
//            fit = new FixedIterationTrainer(rhc, 200);
//            fit.setConvergence_value(new Double(N));
//            fit.train();
//            System.out.println(ef.value(rhc.getOptimal()));
//            rhc_iter_sum += fit.getConverge_iter();
//            rhc_time_sum += fit.getTrain_time();
//
//            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
//            fit = new FixedIterationTrainer(sa, 400);
//            fit.setConvergence_value(new Double(N));
//            fit.train();
//            System.out.println(ef.value(sa.getOptimal()));
//            sa_iter_sum += fit.getConverge_iter();
//            sa_time_sum += fit.getTrain_time();
//        }
//        System.out.println("Avg RHC iterations: " + new Double(rhc_iter_sum)/1000.0);
//        System.out.println("Avg RHC train time ms: " + rhc_time_sum/Math.pow(10,6)/1000.0);
//        System.out.println("Avg SA iterations: " + new Double(sa_iter_sum)/1000.0);
//        System.out.println("Avg SA train time ms: " + sa_time_sum/Math.pow(10, 6)/1000.0);
        
//        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(500, 200, 10, gap);
//        fit = new FixedIterationTrainer(ga, 3000);
//        fit.setConvergence_value(new Double(N));
//        fit.train();
//        System.out.println(ef.value(ga.getOptimal()));

        MIMIC mimic = new MIMIC(50, 10, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.setConvergence_value(new Double(N));
        fit.train();
        System.out.println(ef.value(mimic.getOptimal()));
    }
}