import numpy as np
from preprocess import preprocess
import Gradient_Descent as gd
from Helper_Class import Helper_Class as helper
import matplotlib.pyplot as plt


# Part 0

# loads the data
train = preprocess("data/PA1_train.csv")
val = preprocess("data/PA1_dev.csv", norm=train.norm)
test = preprocess("data/PA1_test.csv", test=True, norm=train.norm)

# Print stats to console
train.get_stats()


# Part 1
myhelperclass = helper()
w_0 = np.zeros(5)

# set stopping criterias
epsilon = .5
max_iter = 1000
max_grad = 10**50

# set regularization constant to zero
lambda_0 = 0

# set initial weights to zero
X_train_shape = train.X_norm.shape
w_0 = np.zeros(X_train_shape[1])

# define list of learning rates to try
alpha_list = [10**0,10**(-1),10**(-2),10**(-3),10**(-4),10**(-5),10**(-6),10**(-7)]

# run descent algorithm using each learning rate and compute SSE for training and validation set
SSE_val_final = []
final_w_for_alpha = []
for alpha in alpha_list:
    
    # run gradient descent using given input values
    [w_vecs, w_grad_vecs, w_grad_norms] = myhelperclass.run_gradient_descent(train.y_norm, train.X_norm, w_0, alpha, lambda_0, epsilon, max_iter, max_grad)
    
    # compute Sum of Square errors for training and test set for ever weight vector generated using 
    # gradient descent 
    SSE_train = myhelperclass.calculuate_SSE(w_vecs, train.y_norm, train.X_norm)
    SSE_val = myhelperclass.calculuate_SSE(w_vecs, val.y_norm, val.X_norm)

    # create list of ending weight vectors, 
    final_w_for_alpha.append(w_vecs[len(SSE_train)-1])
    
    # create integer based spacing to graph SSE for each weight vector associated
    # with a given iteration
    x_axis = np.linspace(0,len(SSE_train)-1,len(SSE_train))

    plt.figure()

    plt.plot(x_axis,SSE_train, x_axis, SSE_val)
    plt.title("SSE for training and validation data, alpha: " + str(alpha) + " lambda: " + str(lambda_0))
    plt.xlabel("training iteration")
    plt.ylabel("SSE")
    plt.legend(["SSE_Training", "SSE_Testing"])
    plt.show()

    plt.figure()

    plt.plot(x_axis, w_grad_norms)
    plt.title("Norm of gradient")
    plt.xlabel("training iteration")
    plt.ylabel("gradient norm")
    plt.show()
    
    # create list of final sum of squared errors
    SSE_val_final.append(SSE_val[len(SSE_val)-1])
    
# select best learning rate based upon best SSE for final weight using validation data
alpha_best = alpha_list[SSE_val_final.index(min(SSE_val_final))]    
print("The best learning rate based upon validation SSE is " + str(alpha_best))

# select best final weight vector using best SSE for final weights using validation data
w_best = final_w_for_alpha[SSE_val_final.index(min(SSE_val_final))]
print("The best final weight based upon validation SSE is" +str(w_best))
print("The feature with the greatest weight is in position " + str(np.max((np.abs(w_best)))))




# Part 2

# set learning rate - must be selected
alpha = 10**(-5)

# define list of learning rates to try

lambda_list = [0, 10**(-3), 10**(-2), 10**(-1), 1, 10, 100]

# run descent algorithm using each learning rate and compute SSE for training and validation set

SSE_val_final = []
final_w_for_lambda_0 = []

for lambda_0 in lambda_list:
    
    # run gradient descent using given input values

    [w_vecs, w_grad_vecs, w_grad_norms] = myhelperclass.run_gradient_descent(train.y_norm, train.X_norm, w_0, alpha, lambda_0, epsilon, max_iter, max_grad)
    
    # compute Sum of Square errors for training and test set for ever weight vector generated using 
    # gradient descent 
    
    SSE_train = myhelperclass.calculuate_SSE(w_vecs, train.y_norm, train.X_norm)
    SSE_val = myhelperclass.calculuate_SSE(w_vecs, val.y_norm, val.X_norm)

    # create list of ending weight vectors, 
    
    final_w_for_lambda_0.append(w_vecs[len(SSE_train)-1])
    
    # create integer based spacing to graph SSE for each weight vector associated
    # with a given iteration
    
    x_axis = np.linspace(0,len(SSE_train)-1,len(SSE_train))

    plt.figure()

    plt.plot(x_axis,SSE_train, x_axis, SSE_val)
    plt.title("SSE for training and validation data, alpha: " + str(alpha) + " lambda: " + str(lambda_0))
    plt.xlabel("training iteration")
    plt.ylabel("SSE")
    plt.legend(["SSE_Training", "SSE_Testing"])
    plt.show()

    plt.figure()

    plt.plot(x_axis, w_grad_norms)
    plt.title("Norm of gradient")
    plt.xlabel("training iteration")
    plt.ylabel("gradient norm")
    plt.show()
    
    # create list of final sum of squared errors
    SSE_val_final.append(SSE_val[len(SSE_val)-1])
    
# select best learning rate based upon best SSE for final weight using validation data
lambda_best = lambda_list[SSE_val_final.index(min(SSE_val_final))]
print("The best regularization constant based upon validation SSE is " + str(lambda_best))
    
# select best final weight vector using best SSE for final weights using validation data
w_best = final_w_for_lambda_0[SSE_val_final.index(min(SSE_val_final))]
print("The best final weight based upon validation SSE is" +str(w_best))


# Part 3
w_0 = np.zeros(5)

# set stopping criterias
epsilon = 0
max_iter = 10000

# set regularization constant to zero
lambda_0 = 

# set initial weights to zero
X_train_shape = train.X_norm.shape
w_0 = np.zeros(X_train_shape[1])

# define list of learning rates to try
alpha_list = [10**2,10**(1),10**(0),10**(-1),10**(-2),10**(-3)]

# run descent algorithm using each learning rate and compute SSE for training and validation set
SSE_val_final = []
final_w_for_alpha = []
for alpha in alpha_list:
    
    # run gradient descent using given input values
    [w_vecs, w_grad_vecs, w_grad_norms] = myhelperclass.run_gradient_descent(train.y, train.X, w_0, alpha, lambda_0, epsilon, max_iter,max_grad)
    
    # compute Sum of Square errors for training and test set for ever weight vector generated using 
    # gradient descent 
    SSE_train = myhelperclass.calculuate_SSE(w_vecs, train.y, train.X)
    SSE_val = myhelperclass.calculuate_SSE(w_vecs, val.y, val.X)

    # create list of ending weight vectors, 
    final_w_for_alpha.append(w_vecs[len(SSE_train)-1])
    
    # create integer based spacing to graph SSE for each weight vector associated
    # with a given iteration
    x_axis = np.linspace(0,len(SSE_train)-1,len(SSE_train))

    plt.figure()

    plt.plot(x_axis,SSE_train, x_axis, SSE_val)
    plt.title("SSE for training and validation data, alpha: " + str(alpha) + " lambda: " + str(lambda_0))
    plt.xlabel("training iteration")
    plt.ylabel("SSE")
    plt.legend(["SSE_Training", "SSE_Testing"])
    plt.show()

    plt.figure()

    plt.plot(x_axis, w_grad_norms)
    plt.title("Norm of gradient")
    plt.xlabel("training iteration")
    plt.ylabel("gradient norm")
    plt.show()
    
    # create list of final sum of squared errors
    SSE_val_final.append(SSE_val[len(SSE_val)-1])
    
# select best learning rate based upon best SSE for final weight using validation data
alpha_best = alpha_list[SSE_val_final.index(min(SSE_val_final))]    
print("The best learning rate based upon validation SSE is " + str(alpha_best))

# select best final weight vector using best SSE for final weights using validation data
w_best = final_w_for_alpha[SSE_val_final.index(min(SSE_val_final))]
print("The best final weight based upon validation SSE is" +str(w_best))
print("The feature with the greatest weight is in position " + str(np.max((np.abs(w_best)))))
