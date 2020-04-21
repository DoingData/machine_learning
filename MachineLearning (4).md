# Machine Learning
# History
In 1642, one of the first mechanical adding machine was designed by Blaise Pascal. It used a system of gears and wheels similar to those found in odometers and other counting devices. Pascal's adder also known as Pascaline, could both add and subtract and was invented to calculate taxes.

In 1694 Gottfried Wilhelm Von Lebniez produced a similar machine to that of Pascaline, that was more accurate and could perform all four arithematic functions. Lebneiz also created binary system used by all modern computers. 
Storing data was the next challenge to be met. In 1801 the first use of storing data was in a weaving loom invented by Joseph Marie Jacquard that used metal cards punched with holes to position threads. A collection of these cards coded a program that directed the loom. This allowed for a process to be repeated with a consistent result every time.
In 1847, George Boole created a way of representing this using Boolean operators (AND, OR, NOR) and having responses represented by true or false, yes or no, and represented in binary as 1 or 0. Web searches still use these operators today.
In 1890 Herman Hollerith created the first combined system of mechanical calculation and punch cards to rapidly calculate statistics gathered from millions of people.
In 1945 Mark I built at IBM and designed by Howard Aiken, was the first combined electric and mechanical computer. The Mark I could store 72 numbers and it could perform complex multiplication in 6 seconds and division in 16.
In 1946 the first fully electronic computer was built by John Mauchly and John Eckert and named ENIAC, short for Electronic Numerical Integrator and Computer.
In 1952 Arthur Samuel was an IBM scientist who used the game of checkers to create the first learning program. His program became a better player after many games against itself and a variety of human players in a 'supervised learning mode'. The program observed which moves were winning strategies and adapted its programming to incorporate those strategies. 
In 1957 Frank Rosenblatt designed the perceptron which is a type of neural network. A neural network acts like your brain; the brain contains billions of cells called neurons that are connected together in a network. The perceptron connects a web of points where simple decisions are made that come together in the larger program to solve more complex problems.
In 1967 the first pattern regonition program were designed based on a type of algorithm called the nearest neighbour. An algorithm is a sequence of instructions and steps. When the program is given a new object it compares this with data from the training set and classifies the object to the nearest neighbour, or most similar object in memory.
In 1981 Gerald Dejong introduced explanation based learning, prior knowledge of the world is provided by training examples which makes this a type of supervised learning. The program analyzes the training data and discards irrelevant information to form a general rule to follow. For example in chess if the program is told that it needs to focus on the queen, it will discard all piesces that don't have immediate effect upon her.
In 1990's Machine learning applications in data mining, adaptive software and web applications, text learning , and language learning were started. Advances continued in machine learning algorithms within the general areas of supervised learning and unsupervised learning. As well, reinforcement learning algorithms were developed.
In 2000's the new millenium brought an explosion of adaptive programming. Anywhere adaptive programs are needed, machine learning learning is there. These programs are capable of recognizihng patterns, learning from experience, abstracting new information from data, and optimizing the efficiency and accuracy of its processing and output.
# Machine Learning: Definition

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.
"Machine learning is defined as field of study that gives computers the ability to learn without being explicitly programmed"- Arthur Samauel
“Machine Learning at its most basic is the practice of using algorithms to parse data, learn from it, and then make a determination or prediction about something in the world.” – Nvidia 
“Machine learning is the science of getting computers to act without being explicitly programmed.” – Stanford
“Machine learning is based on algorithms that can learn from data without relying on rules-based programming.”- McKinsey & Co.
“Machine learning algorithms can figure out how to perform important tasks by generalizing from examples.” – University of Washington
“The field of Machine Learning seeks to answer the question “How can we build computer systems that automatically improve with experience, and what are the fundamental laws that govern all learning processes?” – Carnegie Mellon University

# Types of Machine Learning

# Supervised machine learning 

These algorithms can apply what has been learned in the past to new data using labeled examples to predict future events. Starting from the analysis of a known training dataset, the learning algorithm produces an inferred function to make predictions about the output values. The system is able to provide targets for any new input after sufficient training. The learning algorithm can also compare its output with the correct, intended output and find errors in order to modify the model accordingly.
Imagine a computer is a child, we are its supervisor (e.g. parent, guardian, or teacher), and we want the child (computer) to learn what a pig looks like. We will show the child several different pictures, some of which are pigs and the rest could be pictures of anything (cats, dogs, etc).
When we see a pig, we shout “pig!” When it’s not a pig, we shout “no, not pig!” After doing this several times with the child, we show them a picture and ask “pig?” and they will correctly (most of the time) say “pig!” or “no, not pig!” depending on what the picture is. That is supervised machine learning.
Supervised machine learning algorithms are used to solve classification or regression problems.

1. Classification Problems
A classification problem has a discrete value as its output. For example, “likes pineapple on pizza” and “does not like pineapple on pizza” are discrete. There is no middle ground. The analogy above of teaching a child to identify a pig is another example of a classification problem.
2. Regression Problems
A regression problem has a real number (a number with a decimal point) as its output. For example, we could use the data to estimate someone’s weight given their height.
## Nearest Neighbour Classification
 
Nearest Neighbors is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining and intrusion detection.

It is widely disposable in real-life scenarios since it is non-parametric, meaning, it does not make any underlying assumptions about the distribution of data (as opposed to other algorithms such as GMM, which assume a Gaussian distribution of the given data).
 
KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) with some mathematics we might have learned in our childhood— calculating the distance between points on a graph.

There are other ways of calculating distance, and one way might be preferable depending on the problem we are solving. However, the straight-line distance (also called the Euclidean distance) is a popular and familiar choice.
### The KNN Algorithm
1. Load the data
2. Initialize K to your chosen number of neighbors
3. For each example in the data
3. Calculate the distance between the query example and the current example from the data.
3. Add the distance and the index of the example to an ordered collection
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
5. Pick the first K entries from the sorted collection
6. Get the labels of the selected K entries
7. If regression, return the mean of the K labels
8. If classification, return the mode of the K labels

#### Choosing the right value for K
To select the K that’s right for your data, we run the KNN algorithm several times with different values of K and choose the K that reduces the number of errors we encounter while maintaining the algorithm’s ability to accurately make predictions when it’s given data it hasn’t seen before.
Here are some things to keep in mind:
As we decrease the value of K to 1, our predictions become less stable. 
Just think for a minute, imagine K=1 and we have a query point surrounded by several reds and one green (I’m thinking about the top left corner of the colored plot above), but the green is the single nearest neighbor. Reasonably, we would think the query point is most likely red, but because K=1, KNN incorrectly predicts that the query point is green.
Inversely, as we increase the value of K, our predictions become more stable due to majority voting / averaging, and thus, more likely to make more accurate predictions (up to a certain point). Eventually, we begin to witness an increasing number of errors. It is at this point we know we have pushed the value of K too far.
In cases where we are taking a majority vote (e.g. picking the mode in a classification problem) among labels, we usually make K an odd number to have a tiebreaker.
#### Advantages
The algorithm is simple and easy to implement.
There’s no need to build a model, tune several parameters, or make additional assumptions.
The algorithm is versatile. It can be used for classification, regression, and search (as we will see in the next section).
#### Disadvantages
The algorithm gets significantly slower as the number of examples and/or predictors/independent variables increase.

# unsupervised machine learning 

These algorithms are used when the information used to train is neither classified nor labeled. Unsupervised learning studies how systems can infer a function to describe a hidden structure from unlabeled data. The system doesn’t figure out the right output, but it explores the data and can draw inferences from datasets to describe hidden structures from unlabeled data.
## Clustering
Clustering is one of the most common exploratory data analysis technique used to get an intuition about the structure of the data. It can be defined as the task of identifying subgroups in the data such that data points in the same subgroup (cluster) are very similar while data points in different clusters are very different. In other words, we try to find homogeneous subgroups within the data such that data points in each cluster are as similar as possible according to a similarity measure such as euclidean-based distance or correlation-based distance. The decision of which similarity measure to use is application-specific.
Clustering analysis can be done on the basis of features where we try to find subgroups of samples based on features or on the basis of samples where we try to find subgroups of features based on samples. 
Clustering is used in market segmentation; where we try to fined customers that are similar to each other whether in terms of behaviors or attributes, image segmentation/compression; where we try to group similar regions together, document clustering based on topics, etc.

### K means Algorithm
K means algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the inter-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.


# Reinforcement machine learning 

These algorithms is a learning method that interacts with its environment by producing actions and discovers errors or rewards. Trial and error search and delayed reward are the most relevant characteristics of reinforcement learning. This method allows machines and software agents to automatically determine the ideal behavior within a specific context in order to maximize its performance. Simple reward feedback is required for the agent to learn which action is best; this is known as the reinforcement signal.

# other Machine learning Algorithms

 


