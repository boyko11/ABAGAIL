package shared;

import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
import opt.example.CountOnesEvaluationFunction;

/**
 * A fixed iteration trainer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FixedIterationTrainer implements Trainer {
    
    /**
     * The inner trainer
     */
    private Trainer trainer;

    private EvaluationFunction evaluationFunction;

    /**
     * The number of iterations to train
     */
    private int iterations;

    private double convergence_value = 0;
    private int converge_iter = 0;
    private double train_time = 0.0;

    /**
     * Make a new fixed iterations trainer
     * @param t the trainer
     * @param iter the number of iterations
     */
    public FixedIterationTrainer(Trainer t, int iter) {
        trainer = t;
        iterations = iter;
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        long start_time = System.currentTimeMillis();
        double sum = 0;
        this.converge_iter = iterations;
        for (int i = 0; i < iterations; i++) {
            double optimal_value_after_training = trainer.train();
            double current_optimal_value = optimal_value_after_training;
            sum += optimal_value_after_training;
            if(this.evaluationFunction != null) {
                current_optimal_value = this.evaluationFunction.value(((OptimizationAlgorithm) trainer).getOptimal());
                if (this.convergence_value == current_optimal_value) {
                    System.out.println("Converged in " + i + " iterations");
                    this.converge_iter = i + 1;
                    break;
                }
            }
//            if(i % 100 == 0) {
//                System.out.println("Iteration " + i + " current optimum: " + optimal_value_after_training + " " + current_optimal_value);
//            }
        }
        long end_time = System.currentTimeMillis();
        this.train_time = end_time - start_time;
        System.out.println("Training Time ms: " + this.train_time);
        return sum / iterations;
    }

    public void setConvergence_value(double convergence_value) {
        this.convergence_value = convergence_value;
    }

    public void setEvaluationFunction(EvaluationFunction evaluationFunction) {
        this.evaluationFunction = evaluationFunction;
    }


    public int getConverge_iter() {
        return this.converge_iter;
    }

    public double getTrain_time() {
        return this.train_time;
    }
}
