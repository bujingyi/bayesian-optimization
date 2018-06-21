# Bayesian Optimization
**Bayesian optimizer for hyperparameter tuning**

### 1. Hyperparameter Tunning As a Mathematical Optimization
Among variaous hyperparameter optimization methods, **`Bayesian Optimization`** is probably the third famous (top2 are `Grid Search` and `Random Search` beyond doubt). Unlike the searching strategies, `Baysian Optimization` view hyperparameter tunning as a mathematical optimization problem. 

If we take the hyperparameters as inputs of some real-valued function, the loss of the well trained model under these hyperparameters as the output of that function, then the problem becomes a optimization problem and the best hyperparameter combination is the minimizer of the real-valued function. However we don't know anything about the objective function. All we can do is throw a set of hyperparameter and get a loss (after the model is trained). The objective function is a black box to us.

### 2. Unkown Objective Function
Since the objective function is unknown, the Bayesian strategy is to treat it as a random function and place a `prior distribution` over it. The `prior` is an assumption we made about the function. It captures our beliefs about the behaviour of the function. Each trial of the evaluation of a set of hyperparameter is also an evaluation of the function. After gathering the function evaluations, the `prior` is updated to form the `posterior distribution` over the unknown objective function. The more function evaluations, the more information we know about the function, and the closer we get to the best hyperparameter combinations. Before continuing, we need to answer two questions:

* What is a distribution of a function?  
* How to perform function evaluation?

We will figure them out in the next two sections respectively.

### 3. Stochastic Process - Distribution of Random Functions