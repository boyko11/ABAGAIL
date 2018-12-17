package opt;

import shared.Instance;
import dist.Distribution;

/**
 * A generic hill climbing problem
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class GenericHillClimbingProblem extends GenericOptimizationProblem implements HillClimbingProblem {
    
    /**
     * The neighbor function
     */
    private NeighborFunction neigh;

    /**
     * Make a new hill climbing problem
     * @param eval the evaulation function
     * @param dist the initial distribution
     * @param neigh the neighbor function
     */
    public GenericHillClimbingProblem(EvaluationFunction eval, Distribution dist,
               NeighborFunction neigh) {
        super(eval, dist);
        this.neigh = neigh;
    }

    /**
     * @see opt.HillClimbingProblem#neighbor(shared.Instance)
     */
    public Instance neighbor(Instance d) {
        return neigh.neighbor(d);
    }

    public Instance[] twoNeighbors(Instance d) {

        Instance[] neighbors = new Instance[2];
        neighbors[0] = neigh.neighbor(d);
        neighbors[1] = neigh.neighbor(d);

        return neighbors;
    }

}
