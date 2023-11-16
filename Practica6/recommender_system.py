# coding=utf-8

def checkCostFunction(cofiCostFunc,lamb):
    import numpy as np
    # Create small problem
    X_t = np.random.random((4, 3))
    Theta_t = np.random.random((5, 3))

    # Zap out most entries
    Y = np.dot(X_t,Theta_t.T)
    Y[(np.random.random(np.shape(Y)) > .5)] = 0
    R = np.zeros_like(Y)
    R[Y != 0] = 1

    # Run gradient checking
    X = np.random.random(np.shape(X_t))
    Theta = np.random.random(np.shape(Theta_t))
    n_users = np.size(Y, 1)
    n_movies = np.size(Y,0)
    n_features = np.size(Theta_t,1)

    numgrad = computeNumericalGradient(cofiCostFunc,np.append(X.flatten(), Theta.flatten()),
                                       Y,R,n_users, n_movies,n_features,lamb)    
    J, grad = cofiCostFunc(np.append(X.flatten(), Theta.flatten()),Y,R,n_users,n_movies,n_features,lamb)
    
    print('The above two columns you get should be very similar.')
    print('(Left - Your Numerical Gradient, Right - Analytical Gradient)\n')    
    for i in range(grad.shape[0]):
        print((numgrad[i],grad[i]))

    print('If your backpropagation implementation is correct, then ')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: %0.8e\n' %(np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)))    

def computeNumericalGradient(J,theta,*argv):
    # COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    # and gives us a numerical estimate of the gradient.
    #   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    #   gradient of the function J around theta. Calling y = J(theta) should
    #   return the function value at theta.

    # Notes: The following code implements numerical gradient checking, and 
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical 
    #        approximation of) the partial derivative of J with respect to the 
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
    #        be the (approximately) the partial derivative of J with respect 
    #        to theta(i).)
    
    import numpy as np
    
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    e = 1e-4
    
    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = e
        (loss1,g) = J(theta - perturb,*argv)
        (loss2,g) = J(theta + perturb,*argv)
        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1)/(2*e)
        perturb[p] = 0

    return numgrad

