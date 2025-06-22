# Naive Bayes Classifier

Demonstrate a Naive Bayes classifier to identify spam email

I demonstrate the use of the Scikit-Learn implementation of Naive Bayes Classifier using the CountVectorizer module to learn the difference between spam and non-spam emails based on a dataset of labeled examples.

## Features

- Classifies emails as **spam** or **not spam**
- Trains on labeled text data
- Uses sklearn Naive Bayes algorithm for classification
- Outputs accuracy and performance metrics

## Bayes Theorem

The theorem states that the posterior probability of A given B is equal to B given A times the probability of A divided by the probability of B.
There is one assumption however and that is that all features are independent, which isn't actually true, which is why its called 'Naive', but this assumption enables efficient space and time computation, and in practice works well enough.

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

Where 
- $P(A \mid B)$ : Probability of A given B
- $P(B \mid A)$ : Probability of B given A
- $P(A)$ : Probability of A
- $P(B)$ : Probability of B

As it relates to spam detection given a feature vector $X=(x_1,x_2,..,x_n)$ and a class variable $y$, Bayes Theorem states that:

$$
P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}
$$

Where 
- $P(y \mid X)$ : Probability of a class y given words X
- $P(X \mid y)$ : Probability of words X given class y
- $P(y)$ : Probability of class y
- $P(X)$ : Probability of words X

Were interested in calculating the posterior probability $P(y \mid X)$ from the liklihood of $P(X \mid y)$ and prior probabilities $P(y)$ $P(X)$.

Using the chain rule:

$$
P(X \mid y) = P(x_1,x_2,..,x_n|y)
$$
$$
   = P(x_1|x_2,..,x_n,y) * P(x_2|x_3,..,x_n,y) * .. * P(x_n|y)
$$

Due to the Naive Bayes assumption of independence the conditional probabilities are independent of each other, greatly simplifying:

$$
P(X \mid y) = P(x_1|y) * P(x_2|y) * .. * P(x_n|y)
$$

Thus we have:

$$
P(y \mid X) = \frac{P(x_1|y) * P(x_2|y) * .. * P(x_n|y) \cdot P(y)}{P(X)}
$$

And since $P(X)$ remains a constant for all y it has no effect, and can be optimized away, leaving us with:

$$
P(y \mid x_1,x_2,..,x_n) \propto P(y) \prod_{i=1}^{n} P(x_i|y)
$$

The most likely classification is then decided by which class has a higher joint probability of all the words it contains being spam or not-spam times the prior probablity of a document being that class.

$$
y = \underset{y}{argmax} \space P(y) \prod_{i=1}^{n} P(x_i|y)
$$

## 
