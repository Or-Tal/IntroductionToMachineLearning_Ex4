from adaboost import *
from ex4_tools import *
import matplotlib.pyplot as plt


def Q10(train_set, test_set, title, T=500):
    """
    generates 5000 samples without noise, trains an Adaboost classifier over this data
    using DecisionStump weak learner with T=500, tests it on 200 test samples
    ans to questions
    """
    def vectorised_prediction_err(max_t, args):
        """
        vectorised prediction fuction
        args = [Ada_classifier, X, y]
        """
        return args[0].error(args[1], args[2], max_t[0])

    train_x, train_y = train_set
    test_x, test_y = test_set

    # train Adaboost classifier
    classifier = AdaBoost(DecisionStump, T)
    weights = classifier.train(train_x, train_y)

    # get error vectors
    num_classifiers = np.arange(1, T)
    T_arr = [num_classifiers]
    train_error = np.apply_along_axis(vectorised_prediction_err, 0, T_arr, [classifier, train_x, train_y])
    test_error = np.apply_along_axis(vectorised_prediction_err, 0, T_arr, [classifier, test_x, test_y])

    # get idx of min test_error - for Q12
    min_err = np.amin(test_error)
    idx = np.where(test_error == min_err)[0][0]

    # plot
    plt.plot(num_classifiers, train_error, c='black')
    plt.plot(num_classifiers, test_error, c='red')
    plt.legend(["Train error", "Test error"])
    plt.title(title)
    plt.xlabel("T = committee size")
    plt.ylabel("Classification error")
    plt.show()

    return weights, classifier, idx, min_err


def Q11(classifier, train_set, test_set, title):
    """
    Plot the decisions of the learned classifiers with T in{5; 10; 50; 100; 200; 500} together with
    the test data.
    """
    train_x, train_y = train_set
    test_x, test_y = test_set

    Ts = [5, 10, 50, 100, 200, 500]
    plt.figure(figsize=(16, 8))
    for i, t in enumerate(Ts):
        plt.subplot(2, 3, i + 1)
        decision_boundaries(classifier, test_x, test_y, num_classifiers=t)
        plt.title(title + "num_classifiers = {}".format(t))
    plt.show()


def Q12(train_set, classifier, min_t, title, weights=None):
    """Out of the different values you used for T, and T_hat, the one that minimizes the test error.
    What is T_hat and what is its test error? Plot the decision boundaries of this classier together
    with the training data."""
    train_x, train_y = train_set
    plt.figure(figsize=(8,6))
    decision_boundaries(classifier, train_x, train_y, min_t, weights)
    plt.title(title)
    plt.show()


def Q10_to_13(train_set, test_set, Qnums, noise_ratio):
    """
    mini-main function for running Q10-Q13 code blocks
    will also use for repeating the process in Q14
    """
    # Q10
    title = "Q{}: Error as a function of committee size, noise_ratio = {}".format(Qnums[0], noise_ratio)
    weights, classifier, min_t, min_err = Q10(train_set, test_set, title)

    # Q11
    title = "Q{} decision boundaries: noise = {}, ".format(Qnums[1], noise_ratio)
    Q11(classifier, train_set, test_set, title)

    # Q12
    title = "Q{}: decision boundaries: T_hat={}, test_err={}, noise_ratio={}".format(Qnums[2], min_t, min_err, noise_ratio)
    Q12(train_set, classifier, min_t, title)

    # Q13
    title = "Q{}: decision boundaries, training set scaled by weight, noise_ratio = {}".format(Qnums[3], noise_ratio)
    Q12(train_set, classifier, 500, title, (weights / np.amax(weights)) * 10)


if __name__ == "__main__":
    # gen train, test sets
    # -- Q10 - Q13
    train_size, test_size = 5000, 200
    train_x, train_y = generate_data(train_size, 0)
    test_x, test_y = generate_data(test_size, 0)
    train_set, test_set = [train_x, train_y], [test_x, test_y]
    Qnums = [10, 11, 12, 13]
    Q10_to_13(train_set, test_set, Qnums, 0)


    # -- Q14
    Qnums = [14, 14, 14, 14]
    for noise_ratio in [0.01, 0.4]:
        train_x, train_y = generate_data(train_size, noise_ratio)
        test_x, test_y = generate_data(test_size, noise_ratio)
        train_set, test_set = [train_x, train_y], [test_x, test_y]
        # -- repeat Q10-Q13
        Q10_to_13(train_set, test_set, Qnums, noise_ratio)
