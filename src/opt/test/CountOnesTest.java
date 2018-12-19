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

        System.out.println("CountOnesTest start.");

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


        System.out.println("RHC: ");
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 400);
        fit.setConvergence_value(new Double(N));
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("Optimum after training: " + ef.value(rhc.getOptimal()));
        System.out.println("-------------------------------------");

        System.out.println("SA: ");
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 800);
        fit.setConvergence_value(new Double(N));
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("Optimum after training: " + ef.value(sa.getOptimal()));
        System.out.println("-------------------------------------");

        System.out.println("GA: ");
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(500, 200, 10, gap);
        fit = new FixedIterationTrainer(ga, 3000);
        fit.setConvergence_value(new Double(N));
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("Optimum after training: " + ef.value(ga.getOptimal()));
        System.out.println("-------------------------------------");

        System.out.println("MIMIC: ");
        MIMIC mimic = new MIMIC(50, 10, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.setConvergence_value(new Double(N));
        fit.setEvaluationFunction(ef);
        fit.train();
        System.out.println("Optimum after training: " + ef.value(mimic.getOptimal()));
        System.out.println("-------------------------------------");

        System.out.println("CountOnesTest end.");
    }
}