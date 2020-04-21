# Machine Learning
# History
In 1642, one of the first mechanical adding machine was designed by **Blaise Pascal**. It used a system of gears and wheels similar to those found in odometers and other counting devices. Pascal's adder also known as **Pascaline**, could both add and subtract and was invented to calculate taxes.

In 1694 **Gottfried Wilhelm Von Lebniez** produced a similar machine to that of **Pascaline**, that was more accurate and could perform all four arithematic functions. Lebneiz also created **binary system** used by all modern computers. 

**Storing data** was the next challenge to be met. In 1801 the first use of storing data was in a weaving loom invented by **Joseph Marie Jacquard** that used metal cards punched with holes to position threads. A collection of these cards coded a program that directed the loom. This allowed for a process to be repeated with a consistent result every time.

In 1847, **George Boole** created a way of representing this using **Boolean operators (AND, OR, NOR)** and having responses represented by true or false, yes or no, and represented in binary as 1 or 0. Web searches still use these operators today.

In **1890 Herman Hollerith** created the first combined system of mechanical calculation and punch cards to rapidly calculate statistics gathered from millions of people.

In 1945 **Mark I** built at **IBM** and designed by **Howard Aiken**, was the first combined electric and mechanical computer. The **Mark I** could **store 72 numbers** and it could perform complex multiplication in 6 seconds and division in 16.

In 1946 the first **fully electronic computer** was built by **John Mauchly** and **John Eckert** and named **ENIAC**, short for *Electronic Numerical Integrator and Computer*.

In 1952 **Arthur Samuel** was an **IBM** scientist who used the game of **checkers** to create the **first learning program**. His program became a better player after many games against itself and a variety of human players in a **'supervised learning mode'**. The program observed which moves were winning strategies and adapted its programming to incorporate those strategies. 

In 1957 **Frank Rosenblatt** designed the perceptron which is a type of *neural network*. A neural network acts like your brain; the brain contains billions of cells called neurons that are connected together in a network. The perceptron connects a web of points where simple decisions are made that come together in the larger program to solve more complex problems.
In 1967 the first **pattern regonition program** were designed based on a type of algorithm called the *nearest neighbour*. An algorithm is a sequence of instructions and steps. When the program is given a new object it compares this with data from the training set and classifies the object to the *nearest neighbour*, or most similar object in memory.

In 1981 **Gerald Dejong** introduced *explanation based learning*, prior knowledge of the world is provided by training examples which makes this a type of supervised learning. The program analyzes the training data and discards irrelevant information to form a general rule to follow. For example in chess if the program is told that it needs to focus on the queen, it will discard all piesces that don't have immediate effect upon her.

In 1990's Machine learning applications in *data mining, adaptive software and web applications, text learning , and language learning* were started. Advances continued in machine learning algorithms within the general areas of supervised learning and unsupervised learning. As well, reinforcement learning algorithms were developed.

In 2000's the new millenium brought an explosion of *adaptive programming*. Anywhere adaptive programs are needed, machine learning  is there. These programs are capable of recognizihng patterns, learning from experience, abstracting new information from data, and optimizing the efficiency and accuracy of its processing and output.

## Time Line of MACHINE LEARNING History 

![Sample diagram](assets/history_of_ml.SVG)


# Machine Learning: 

### Definition

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
"Machine learning is defined as field of study that gives computers the ability to learn without being explicitly programmed"- Arthur Samauel
“Machine Learning at its most basic is the practice of using algorithms to parse data, learn from it, and then make a determination or prediction about something in the world.” – Nvidia 
“Machine learning is the science of getting computers to act without being explicitly programmed.” – Stanford
“Machine learning is based on algorithms that can learn from data without relying on rules-based programming.”- McKinsey & Co.
“Machine learning algorithms can figure out how to perform important tasks by generalizing from examples.” – University of Washington
“The field of Machine Learning seeks to answer the question “How can we build computer systems that automatically improve with experience, and what are the fundamental laws that govern all learning processes?” – Carnegie Mellon University

## Types of Machine Learning
![Sample Diagram](assets/ML_DIAGRAM1.SVG)

### Supervised machine learning 

These algorithms can apply what has been learned in the past to new data using labeled examples to predict future events. Starting from the analysis of a known training dataset, the learning algorithm produces an inferred function to make predictions about the output values. The system is able to provide targets for any new input after sufficient training. The learning algorithm can also compare its output with the correct, intended output and find errors in order to modify the model accordingly.


![Sample Diagram](assets/ML_DIAGRAM2.SVG)


### Unsupervised machine learning 

These algorithms are used when the information used to train is neither classified nor labeled. Unsupervised learning studies how systems can infer a function to describe a hidden structure from unlabeled data. The system doesn’t figure out the right output, but it explores the data and can draw inferences from datasets to describe hidden structures from unlabeled data.
![Sample Diagram](assets/ML_DIAGRAM3.SVG)

### Reinforcement machine learning 

These algorithms are learning methods that interacts with its environment by producing actions and discovers errors or rewards. Trial and error search and delayed reward are the most relevant characteristics of reinforcement learning. This method allows machines and software agents to automatically determine the ideal behavior within a specific context in order to maximize its performance. Simple reward feedback is required for the agent to learn which action is best; this is known as the reinforcement signal.

![Sample Diaggram](assets/ML_DIAGRAM4.SVG)

## Other Machine learning Algorithms

 ### Nearest Neighbour Classification

Nearest Neighbors is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in *pattern recognition, data mining and intrusion detection*.

It is widely disposable in real-life scenarios since it is non-parametric, meaning, it does not make any underlying assumptions about the distribution of data (as opposed to other algorithms such as GMM, which assume a Gaussian distribution of the given data).

# Linear Regression
Linear regression attempts to model the relationship between two variables by fitting a linear equation to observed data. One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable. For example, a modeler might want to relate the weights of individuals to their heights using a linear regression model.

Before attempting to fit a linear model to observed data, a modeler should first determine whether or not there is a relationship between the variables of interest. This does not necessarily imply that one variable causes the other (for example, higher SAT scores do not cause higher college grades), but that there is some significant association between the two variables. 

A scatterplot can be a helpful tool in determining the strength of the relationship between two variables. If there appears to be no association between the proposed explanatory and dependent variables (i.e., the scatterplot does not indicate any increasing or decreasing trends), then fitting a linear regression model to the data probably will not provide a useful model. A valuable numerical measure of association between two variables is the correlation coefficient, which is a value between -1 and 1 indicating the strength of the association of the observed data for the two variables.

A linear regression line has an equation of the form Y = a + bX, where X is the explanatory variable and Y is the dependent variable. The slope of the line is b, and a is the intercept (the value of y when x = 0).

## Least-Squares Regression
Linear least squares (LLS) is the least squares approximation of linear functions to data. It is a set of formulations for solving statistical problems involved in linear regression, including variants for ordinary (unweighted), weighted, and generalized (correlated) residuals. 

Numerical methods for linear least squares include inverting the matrix of the normal equations and orthogonal decomposition methods.

### Explanation
In statistics and mathematics, linear least squares is an approach to fitting a mathematical or statistical model to data in cases where the idealized value provided by the model for any data point is expressed linearly in terms of the unknown parameters of the model. The resulting fitted model can be used to summarize the data, to predict unobserved values from the same system, and to understand the mechanisms that may underlie the system.

Mathematically, linear least squares is the problem of approximately solving an overdetermined system of linear equations A x = b, where b is not an element of the column space of the matrix A. The approximate solution is realized as an exact solution to A x = b', where b' is the projection of b onto the column space of A. 

The best approximation is then that which minimizes the sum of squared differences between the data values and their corresponding modeled values. The approach is called linear least squares since the assumed function is linear in the parameters to be estimated. Linear least squares problems are convex and have a closed-form solution that is unique, provided that the number of data points used for fitting equals or exceeds the number of unknown parameters, except in special degenerate situations. In contrast, non-linear least squares problems generally must be solved by an iterative procedure, and the problems can be non-convex with multiple optima for the objective function. If prior distributions are available, then even an underdetermined system can be solved using the Bayesian MMSE estimator.

In statistics, linear least squares problems correspond to a particularly important type of statistical model called linear regression which arises as a particular form of regression analysis. One basic form of such a model is an ordinary least squares model. The present article concentrates on the mathematical aspects of linear least squares problems, with discussion of the formulation and interpretation of statistical regression models and statistical inferences related to these being dealt with in the articles just mentioned. See outline of regression analysis for an outline of the topic.

