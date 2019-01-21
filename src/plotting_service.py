import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path

def plot_learning_curve(title, x_label, results_file, dots, fill):

    if os.path.isfile(results_file):
        data = np.genfromtxt(results_file, delimiter=',')
    else:
        print("File {0} not found. Did not generate learning plot for it.".format(results_file))
        return

    number_of_lines = data.shape[0]
    train_size_indices = range(0, number_of_lines, 3)
    train_score_indices = range(1, number_of_lines, 3)
    test_score_indices = range(2, number_of_lines, 3)

    train_sizes = data[0, :]
    train_scores = data[train_score_indices, :] / 100
    test_scores = data[test_score_indices, :] /100

    plt.figure()
    plt.title(title)
    # if ylim is not None:
    #     plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel("Error")

    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    train_errors_mean = np.mean(train_errors, axis=0)
    train_errors_std = np.std(train_errors, axis=0)
    test_errors_mean = np.mean(test_errors, axis=0)
    test_errors_std = np.std(test_errors, axis=0)

    plt.grid()

    if fill:
        plt.fill_between(train_sizes, train_errors_mean - train_errors_std,
                         train_errors_mean + train_errors_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_errors_mean - test_errors_std,
                         test_errors_mean + test_errors_std, alpha=0.1, color="g")

    if dots:
        plt.plot(train_sizes, train_errors_mean, 'o-', color="r", label="Training error")
        plt.plot(train_sizes, test_errors_mean, 'o-', color="g", label="Test error")
    else:
        plt.plot(train_sizes, train_errors_mean, color="r", label="Training error")
        plt.plot(train_sizes, test_errors_mean, color="g", label="Test error")

    plt.legend(loc="best")
    #plt.show()
    plot_file = results_file.replace('.csv','.png')
    plt.savefig(plot_file)
    print('Finished Plotting {0}'.format(plot_file))


algo_results_to_plot = sys.argv[1].strip()
print("Plotting {0} results.".format(algo_results_to_plot))

title = 'Learning curve by training set sizes'
x_label = 'Training examples'
results_file = 'learning_curve_training_sizes_{0}.csv'.format(algo_results_to_plot)
dots = True
fill = True
plot_learning_curve(title, x_label, results_file, dots, fill)


title = 'Learning curve by iterations'
x_label = 'Iterations'
results_file = 'learning_curve_iterations_{0}.csv'.format(algo_results_to_plot)
dots = False
fill = False
plot_learning_curve(title, x_label, results_file, dots, fill)
