code: https://github.com/boyko11/ABAGAIL
    This is a modified fork of ABAGAIL: https://github.com/pushkar/ABAGAIL

The main submission files:
    https://github.com/boyko11/ABAGAIL/blob/master/ABAGAIL-Boyko.jar
    https://github.com/boyko11/ABAGAIL/blob/master/src/project2_cs7641/NNOptimizationTestBoyko.java
    https://github.com/boyko11/ABAGAIL/blob/master/src/plotting_service.py
    https://github.com/boyko11/ABAGAIL/blob/master/src/opt/test/CountOnesTest.java
    https://github.com/boyko11/ABAGAIL/blob/master/src/opt/test/FlipFlopTest.java
    https://github.com/boyko11/ABAGAIL/blob/master/src/opt/test/FourPeaksTest.java
    https://github.com/boyko11/ABAGAIL/blob/master/src/opt/test/KnapsackTest.java
    https://github.com/boyko11/ABAGAIL/blob/master/src/opt/test/breast_cancer_zero_mean_unit_var.csv

Remaining Modifications of the original ABAGAIL files are discussed in btodorov6-analisys.pdf

Run Instructions:

Assumes installed:
Java >= 8, python >= 3, numpy, matplotlib

    git clone https://github.com/boyko11/ABAGAIL
    cd to the cloned project directory

Files to run:

    src/plotting_service.py
    project2_cs7641.NNOptimizationTestBoyko.java
    opt.test.CountOnesTest.java
    opt.test.FlipFlopTest.java
    opt.test.FourPeaksTest.java
    opt.test.KnapsackTest.java

The standardized breast cancer dataset:

    /opt/test/breast_cancer_zero_mean_unit_var.csv


RHC in place of backpropagation:

    To generate data for the Learning Curve graph by iterations:

    java -classpath ABAGAIL-Boyko.jar project2_cs7641.NNOptimizationTestBoyko RHC
    Note: this only generates data for the plot in a .csv file: learning_curve_iterations_RHC.csv, not the actual plot

FOR SA or GA, pass SA or GA instead of RHC, e.g.

    java -classpath ABAGAIL-Boyko.jar project2_cs7641.NNOptimizationTestBoyko SA

    Note: GA takes about 3.5 minutes to run on my machine

Note: The following command will take a longer time, especially for GA -
To generate DATA for the Learning Curve graph by iterations AND training sizes:

    java -classpath ABAGAIL-Boyko.jar project2_cs7641.NNOptimizationTestBoyko RHC training_sizes
    Note: this only generates data for the plot in a .csv file: learning_curve_training_sizes_RHC.csv, not the actual plot


FOR SA or GA, pass SA or GA instead of RHC, e.g.

    java -classpath ABAGAIL-Boyko.jar project2_cs7641.NNOptimizationTestBoyko SA training_sizes
    Note: GA takes about 25 minutes to run

To Generate the actual graphs which will be .png files
From the directory where the java command was run:

    python src/plotting_service.py RHC

    This generates two .png Learning Curve Graph files in the run directory:

    learning_curve_iterations_RHC.png
    learning_curve_training_sizes_RHC.png

FOR SA or GA, replace RHC with SA or GA, e.g.

    python src/plotting_service.py GA

To duplicate the results from the CountOnesTest:

    java -classpath ABAGAIL-Boyko.jar opt.test.CountOnesTest

To duplicate the results from Alternating Ones:

    java -classpath ABAGAIL-Boyko.jar opt.test.FlipFlopTest

To duplicate the results from FourPeaksTest:

    java -classpath ABAGAIL-Boyko.jar opt.test.FourPeaksTest

    Note: I've limited Mimic to only 100 iterations, so it will likely not converge as the run documented in the report
    - that one ran for 2000 iterations, which took 7 minutes

To duplicate the results from KnapsackTest:

    java -classpath ABAGAIL-Boyko.jar opt.test.KnapsackTest









