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
It then computes the liklihood of each feature times the liklihood of each class and chooses which class is more likely.

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

Where 
- $P(A \mid B)$ : Probability of A given B
- $P(B \mid A)$ : Probability of B given A
- $P(A)$ : Probability of A
- $P(B)$ : Probability of B


$$
P(is spam \mid words) = \frac{P(words \mid is spam) \cdot P(is spam)}{P(words)}
$$

Since the probability of all words is the same for spam and non-spam it can be ignored.
So the probability of a document being spam is the joint probability of all the words it contains being higher than the joint probability of all the words in non-spam documents.
Because the document length would change the joint probabiilty the score should be normalized by the document length.

