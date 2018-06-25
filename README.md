# Bayesian Optimization
** *Bayesian optimizer for hyperparameter tuning* **

### 1. Hyperparameter tunning as a mathematical optimization
Among variaous hyperparameter optimization methods, **`Bayesian Optimization`** is probably the third famous (top2 are `Grid Search` and `Random Search` beyond doubt). Unlike the searching strategies, `Baysian Optimization` view hyperparameter tunning as a mathematical optimization problem. 

If we take the hyperparameters as inputs of some real-valued function, the loss of the well trained model under these hyperparameters as the output of that function, then the problem becomes a optimization problem and the best hyperparameter combination is the minimizer of the real-valued function. However we don't know anything about the objective function. All we can do is throw a set of hyperparameter and get a loss (after the model is trained). The objective function is a black box to us.

### 2. Unknown objective function
Since the objective function is unknown, the Bayesian strategy is to treat it as a `random function` and place a `prior distribution` over it. The `prior` is an assumption we made about the function. It captures our beliefs about the behaviour of the function. Each trial of the evaluation of a set of hyperparameter is also an evaluation of the function. After gathering the function evaluations, the `prior` is updated to form the `posterior distribution` over the unknown objective function. The more function evaluations, the more information we know about the function, and the closer we get to the best hyperparameter combinations. Before continuing, we need to answer two questions:

* *What is a `prior` or `posterior` distribution of a random function?*  
* *How to perform function evaluation?*

We will figure them out in the next two sections respectively.

>There are two major choices that must be made when performing Bayesian optimization. First, one must select a prior over functions that will express assumptions about the function being optimized. Second, we must choose an acquisition function, which is used to construct a utility function from the model posterior, allowing us to determine the next point to evaluate.

### 3. Stochastic Process - Distribution of Random Functions
>In probability theory and related fields, a stochastic or random process is a mathematical object usually defined as a collection of random variables indexed by some set...

Stochastic process has several interpretations. The most common one is the index set is interpreted as time. Here we will use another interpretatioin - `random function`. 
>The term random function is also used to refer to a stochastic or random process, because a stochastic process can also be interpreted as a random element in a function space.

Under this interpretation, stochastic process can be regarded as the distribution of a random function. Next is to choose one specific stochastic process as the `prior` distribution of the objective function. In Baysian Optimization, the most wildly used is `Gaussian Process (GP)`. 

#### Gaussian Process
*TODO: some explanations about GP...*

### 4. Function Evaluation - Acquisition Function
We assume that the objective function is drawn from a Gaussian process prior and each hyperparameter combination together with its corresponding loss makes up a sample data. This prior and these data induce a posterior over functions. The acquisition function determines what point should be evaluated next via a proxy optimization, where several different functions have been proposed:
* **Probability of Improvement** is to maximize the probability of improving over the best current value.
* **Expected Improvement** is to maximize the expected improvement (EI) over the current best.
* **Upper Confidence Bound** considers the `exploitation vs. exploration tradeoff`. The idea is to exploiting upper confidence bounds to construct acquisition functions that minimize regret over the course of their optimization.

### 5. Let's put everything together - Bayesian optimization procedure
1. Choose some `prior` distribution over the space of possible objective functions
2. Combine `prior` and the likelihood to get a `posterior` distributed over the objective function given some evaluations.
3. Use the `posterior` to decide where to take the next evaluation according to some prechosen acquisition function.
4. Augment the data (evaluations).
Iterate between 2 and 4 until converged.

#### Reference
[Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/pdf/1206.2944.pdf)  
[Bayesian optimization - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_optimization)  
[Stochastic process - Wikipedia](https://en.wikipedia.org/wiki/Stochastic_process)
