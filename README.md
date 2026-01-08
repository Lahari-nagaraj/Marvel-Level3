# Task1

In this task, I worked on understanding and implementing the Decision Tree based ID3 algorithm. Initially, I studied the basic concepts of decision trees such as nodes, branches, entropy, information gain, and how decisions are made at each level of the tree. I learned that entropy is used to measure the uncertainty in data, and information gain helps in selecting the best attribute for splitting the dataset. After understanding the theory, I implemented the ID3 algorithm from scratch using Python, without using any machine learning libraries. This helped me clearly understand how the tree is built recursively by choosing the attribute with the highest information gain at every step. I also tested the model by predicting outcomes for new input values, which gave correct results. Through this task, I gained a clear practical understanding of how decision trees work internally and how mathematical measures guide the decision-making process. Overall, this task helped me connect theoretical concepts with hands-on implementation.
<img width="1917" height="989" alt="Screenshot 2026-01-06 081104" src="https://github.com/user-attachments/assets/6c6bb179-dba3-4147-acd0-4bb816f5eb61" />


<img width="870" height="523" alt="Screenshot 2026-01-06 080757" src="https://github.com/user-attachments/assets/e83f7c35-a337-4190-a1c3-276db80e8a81" />

<img width="755" height="572" alt="Screenshot 2026-01-06 080630" src="https://github.com/user-attachments/assets/b707febe-8d05-4087-a006-86097dbe1479" />
<img width="340" height="552" alt="Screenshot 2026-01-06 080527" src="https://github.com/user-attachments/assets/6bdac569-66d0-4ffb-82f5-8ed533fe9f7a" />


# Task 2

In this task, I studied and implemented the Naive Bayes classification algorithm with a focus on text classification. I first understood the theoretical concepts such as Bayes’ theorem, prior probability, likelihood, and the independence assumption used in Naive Bayes. Using the sklearn library, I implemented the Multinomial Naive Bayes classifier and observed how text data is converted into numerical form using the Bag of Words model. The trained model was able to correctly classify messages as spam or ham based on learned word patterns. To strengthen my understanding, I also implemented Naive Bayes from scratch using Python, which helped me clearly understand how word frequencies and probabilities are calculated internally. This task helped me understand why Naive Bayes works efficiently for text-based problems despite its simplifying assumptions. Overall, the experiment gave me both conceptual clarity and practical exposure to probabilistic classification.


# Task3

I explored various ensemble learning techniques and applied them to the Titanic dataset to predict passenger survival. Ensemble techniques combine multiple machine learning models to improve performance by reducing bias and variance. I first preprocessed the Titanic data by handling missing values and encoding categorical features. Then I implemented several popular ensemble models including Random Forest, Gradient Boosting, and AdaBoost using the sklearn library. I also built a Stacking classifier that combined multiple base models with a logistic regression meta-learner. Each model was trained using the training data and evaluated using a validation set. The performance of each ensemble method was compared using accuracy scores, and a bar chart was created to visualize the results. Overall, ensemble methods showed strong performance and improved prediction accuracy compared to single models. This exercise helped me understand how ensemble techniques work in practice and how they can be beneficial for real-world classification tasks.
