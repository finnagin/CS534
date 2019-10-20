import numpy as np
from preprocess import preprocess
from Helper_Class import Helper_Class as helper
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--parts", "-p", type=int, nargs='*', help="The parts you want to run this on seperated by spaces", default=[0, 1, 2, 3])
parser.add_argument("--hide", action='store_true', help="Add if you want to hide the plots")

args = parser.parse_args()

color_list = ["#984ea3",
                "#ff7f00",
                "#4daf4a",
                "#e41a1c",
                "#377eb8",
                "#ffff33",
                "#a65628",
                "#f781bf",
                "#999999"]

# Part 0

# loads the data
train = preprocess("data/PA1_train.csv")
val = preprocess("data/PA1_dev.csv", norm=train.norm)
test = preprocess("data/PA1_test.csv", test=True, norm=train.norm)



# Print stats to console
if 0 in args.parts:
    print("Starting Part 0...")
    train.get_stats()


# Part 1
myhelperclass = helper()
w_0 = np.zeros(5)

# set stopping criterias
epsilon = 5
max_iter = 12000
max_grad = 10**50

# set regularization constant to zero
lambda_0 = 0


# set initial weights to zero
X_train_shape = train.X_norm.shape
X_val_shape = val.X_norm.shape
w_0 = np.zeros(X_train_shape[1])

# define list of learning rates to try
alpha_list = [10**0,10**(-1),10**(-2),10**(-3),10**(-4),10**(-5),10**(-6),10**(-7)]

if 1 in args.parts:
    print("Starting Part 1...")
    # run descent algorithm using each learning rate and compute SSE for training and validation set
    SSE_val_final = []
    final_w_for_alpha = []
    SSE_trains = []
    for alpha in alpha_list:
        
        # run gradient descent using given input values
        w_vecs, w_grad_vecs, w_grad_norms = myhelperclass.run_gradient_descent(train.y_norm, train.X_norm, w_0, alpha, lambda_0, epsilon, max_iter, max_grad)
        
        # compute Sum of Square errors for training and test set for ever weight vector generated using 
        # gradient descent 
        SSE_train = myhelperclass.calculuate_SSE(w_vecs, train.y_norm, train.X_norm)
        SSE_val = myhelperclass.calculuate_SSE(w_vecs, val.y_norm, val.X_norm)

        # create list of ending weight vectors, 
        final_w_for_alpha.append(w_vecs[-1])
        
        # create integer based spacing to graph SSE for each weight vector associated
        # with a given iteration
        x_axis = np.linspace(0,len(SSE_train)-1,len(SSE_train))

        print("(Training SSE, Validation SSE) for Learning Rate " +str(alpha) + ":")
        print("  " + str(SSE_train[-1]) + ", " + str(SSE_val[-1]))

        if alpha <= 10**(-5):
            if not args.hide:
                plt.figure()
                plt.plot(x_axis[1:], SSE_train[1:], label="Training Data", color=color_list[0]) 
                plt.plot(x_axis[1:], SSE_val[1:], label="Validation Data", color=color_list[1])
                plt.title("SSE for training and validation data, learning rate: " + str(alpha) + " lambda: " + str(lambda_0))
                plt.xlabel("Iteration")
                plt.ylabel("SSE")
                plt.legend()
                plt.savefig("data/sse_val_train_alpha_" + str(alpha) + "_lambda_" + str(lambda_0) + ".png", dpi=100)

        SSE_trains.append(SSE_train)

        """
        plt.figure()

        plt.plot(x_axis, w_grad_norms)
        plt.title("Norm of gradient, alpha: " + str(alpha))
        plt.xlabel("training iteration")
        plt.ylabel("gradient norm")
        plt.show()
        """

        #print(w_grad_norms[-1])
        
        # create list of final sum of squared errors
        SSE_val_final.append(SSE_val[-1])

    if not args.hide:
        plt.figure()
        idx = 0
        sse = SSE_trains[-5]
        plt.plot(np.linspace(0,len(sse)-1,len(sse)), sse, label="alpha=" + str(alpha_list[-5]), color = color_list[idx])
        plt.title("SSE for Training Data at Learning Rate: " + str(alpha_list[-5]))
        plt.ylim(0,10**10)
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.legend()
        #plt.save("val_train_plot_learning_rates.png")
        plt.savefig("data/train_sse_alpha_" + str(alpha_list[-5]) + ".png", dpi=100)


        plt.figure()
        sse = SSE_trains[-4]
        idx = 0
        plt.plot(np.linspace(0,len(sse)-1,len(sse)), sse, label="alpha=" + str(alpha_list[-4]), color = color_list[idx])
        plt.title("SSE for Training Data at Learning Rate: " + str(alpha_list[-4]))
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.legend()
        #plt.save("val_train_plot_learning_rates.png")
        plt.savefig("data/train_sse_alpha_" + str(alpha_list[-4]) + ".png", dpi=100)

        plt.figure()
        idx = 0
        for sse in SSE_trains[-3:]:
            plt.plot(np.linspace(0,len(sse)-1,len(sse)), sse, label="alpha=" + str(alpha_list[-3+idx]), color = color_list[idx])
            idx += 1
        plt.title("SSE for Training Data at Convergent Learning Rates")
        plt.ylim(0,40)
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.legend()
        #plt.save("val_train_plot_learning_rates.png")
        plt.savefig("data/train_sse_convergent_rates.png", dpi=100)

        
    # select best learning rate based upon best SSE for final weight using validation data
    alpha_best = alpha_list[SSE_val_final.index(min(SSE_val_final))]    
    print("The best learning rate based upon validation SSE is " + str(alpha_best))

    # select best final weight vector using best SSE for final weights using validation data
    best_idx=SSE_val_final.index(min(SSE_val_final))
    w_best = final_w_for_alpha[best_idx]
    print("The best final weight based upon validation SSE is" +str(w_best))
    sorted_idx = np.argsort(-np.abs(w_best))
    print("The values of w sorted by magnitude:")
    for idx in sorted_idx:
        print("  "+train.keys[idx]+": "+str(w_best[idx]))
    




# Part 2

# set learning rate using best alpha from #Part 1
alpha = 10**(-5)

X_train_shape = train.X_norm.shape
w_0 = np.zeros(X_train_shape[1])

if 2 in args.parts:
    print("Starting Part 2...")
    # define list of learning rates to try
    lambda_list = [0, 10**(-3), 10**(-2), 10**(-1), 1, 10, 100]

    # run descent algorithm using each learning rate and compute SSE for training and validation set

    SSE_val_final = []
    final_w_for_lambda_0 = []

    for lambda_0 in lambda_list:
        
        # run gradient descent using given input values

        w_vecs, w_grad_vecs, w_grad_norms = myhelperclass.run_gradient_descent(train.y_norm, train.X_norm, w_0, alpha, lambda_0, epsilon, max_iter, max_grad)
        
        # compute Sum of Square errors for training and test set for ever weight vector generated using 
        # gradient descent 
        
        SSE_train = myhelperclass.calculuate_SSE(w_vecs, train.y_norm, train.X_norm)
        SSE_val = myhelperclass.calculuate_SSE(w_vecs, val.y_norm, val.X_norm)

        # create list of ending weight vectors, 
        
        final_w_for_lambda_0.append(w_vecs[-1])
        
        # create integer based spacing to graph SSE for each weight vector associated
        # with a given iteration
        
        x_axis = np.linspace(0,len(SSE_train)-1,len(SSE_train))
        if not args.hide:
            plt.figure()

            plt.plot(x_axis[1:], SSE_train[1:], label="Training Data", color=color_list[0]) 
            plt.plot(x_axis[1:], SSE_val[1:], label="Validation Data", color=color_list[1])
            plt.title("SSE for training and validation data, alpha: " + str(alpha) + " lambda: " + str(lambda_0))
            plt.xlabel("Iteration")
            plt.ylabel("SSE")
            plt.legend()
            plt.savefig("data/sse_val_train_alpha_" + str(alpha) + "_lambda_" + str(lambda_0) + ".png", dpi=100)

        """
        plt.figure()

        plt.plot(x_axis, w_grad_norms)
        plt.title("Norm of gradient, lambda:" + str(lambda_0))
        plt.xlabel("Iteration")
        plt.ylabel("gradient norm")
        plt.show()
        """

        if lambda_0 in [0,10**(-2),10]:
            sorted_idx = np.argsort(-np.abs(w_vecs[-1]))
            print("The values of w for lamba = "+str(lambda_0)+":")
            for idx in range(len(w_vecs[-1])):
                print("  "+train.keys[idx]+": "+str(w_vecs[-1][idx]))

        # create list of final sum of squared errors
        SSE_val_final.append(SSE_val[-1])
        print(str(lambda_0)+":")
        print("  " + str(SSE_val[-1]))
        
    # select best learning rate based upon best SSE for final weight using validation data
    lambda_best = lambda_list[SSE_val_final.index(min(SSE_val_final))]
    print("The best regularization constant based upon validation SSE is " + str(lambda_best))
        
    # select best final weight vector using best SSE for final weights using validation data
    #w_best = final_w_for_lambda_0[SSE_val_final.index(min(SSE_val_final))]
    #print("The best final weight based upon validation SSE is" +str(w_best))
    #sorted_idx = np.argsort(-np.abs(w_best))
    #print("The values of w sorted by magnitude")
    #for idx in sorted_idx:
    #    print("  "+train.keys[idx]+": "+str(w_best[idx]))


# Part 3

if 3 in args.parts:
    print("Starting Part 3...")
    # set stopping criterias
    epsilon = 0
    max_iter = 10000

    # set regularization constant to zero
    lambda_0 = 0

    # set initial weights to zero
    X_train_shape = train.X_norm.shape
    w_0 = np.zeros(X_train_shape[1])

    # define list of learning rates to try
    alpha_list = [10**0,0,10**(-3),10**(-6),10**(-9),10**(-15)]

    # run descent algorithm using each learning rate and compute SSE for training and validation set
    SSE_val_final = []
    final_w_for_alpha = []
    for alpha in alpha_list:
        
        # run gradient descent using given input values
        w_vecs, w_grad_vecs, w_grad_norms = myhelperclass.run_gradient_descent(train.y, train.X, w_0, alpha, lambda_0, epsilon, max_iter,max_grad)
        
        # compute Sum of Square errors for training and test set for ever weight vector generated using 
        # gradient descent 
        SSE_train = myhelperclass.calculuate_SSE(w_vecs, train.y, train.X)
        SSE_val = myhelperclass.calculuate_SSE(w_vecs, val.y, val.X)

        # create list of ending weight vectors, 
        final_w_for_alpha.append(w_vecs[-1])
        
        # create integer based spacing to graph SSE for each weight vector associated
        # with a given iteration
        x_axis = np.linspace(0,len(SSE_train)-1,len(SSE_train))

        print("(Training SSE, Validation SSE) for Learning Rate " +str(alpha) + " on unnormalized data:")
        print("  " + str(SSE_train[-1]) + ", " + str(SSE_val[-1]))

        if not args.hide:
            plt.figure()
            plt.plot(x_axis[1:], SSE_train[1:], label="Training Data", color=color_list[0]) 
            plt.plot(x_axis[1:], SSE_val[1:], label="Validation Data", color=color_list[1])
            plt.title("SSE for training and validation data, alpha: " + str(alpha) + " lambda: " + str(lambda_0))
            plt.xlabel("Iteration")
            plt.ylabel("SSE")
            plt.legend()
            plt.savefig("data/sse_unnormalized_alpha_" + str(alpha) + "_lambda_" + str(lambda_0) + ".png", dpi=100)

        """
        plt.figure()

        plt.plot(x_axis, w_grad_norms)
        plt.title("Norm of gradient. alpha:" + str(alpha) + " lambda: "+str(lambda_0))
        plt.xlabel("Iteration")
        plt.ylabel("gradient norm")
        plt.show()
        """

        # create list of final sum of squared errors
        SSE_val_final.append(SSE_val[-1])

if 2 in args.parts:
    with open("pred.csv","w") as fid:
        min_val, max_val = train.norm['price']
        for x in test.X_norm:
            y = np.dot(w_best,x)
            y = y*(max_val-min_val)+min_val
            fid.write(str(y)+"\n")