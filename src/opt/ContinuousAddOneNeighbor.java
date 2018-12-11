package opt;

import dist.Distribution;

import shared.Instance;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * A continuous add one neighbor function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ContinuousAddOneNeighbor implements NeighborFunction {
    /**
     * The amount to add to the value
     */
    private double amount;
    
    /**
     * Continuous add one neighbor
     * @param amount the amount to add
     */
    public ContinuousAddOneNeighbor(double amount) {
        this.amount = amount;
    }
    
    /**
     * Continuous add one neighbor
     */
    public ContinuousAddOneNeighbor() {
        this(1);
    }

    /**
     *
     */
    public Instance neighbor(Instance d) {
        int i = Distribution.random.nextInt(d.size());
        Instance cod = (Instance) d.copy();
        double step = Distribution.random.nextDouble() * amount - amount / 2;
        cod.getData().set(i, cod.getContinuous(i) + step);
        return cod;
    }

    @Override
    public Instance[] twoNeighbors(Instance d) {

        int i = Distribution.random.nextInt(d.size());
        Instance cod1 = (Instance) d.copy();
        Instance cod2 = (Instance) d.copy();
        double step = Distribution.random.nextDouble() * amount - amount / 2;
        cod1.getData().set(i, cod1.getContinuous(i) + step);
        cod2.getData().set(i, cod2.getContinuous(i) - step);

        return new Instance[]{cod1, cod2};
    }
}
