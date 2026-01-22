<img width="932" height="575" alt="Screenshot 2026-01-18 140527" src="https://github.com/user-attachments/assets/7a9cee00-5b92-408e-a6bb-19c77c463d35" /># Task1

In this task, I worked on understanding and implementing the Decision Tree based ID3 algorithm. Initially, I studied the basic concepts of decision trees such as nodes, branches, entropy, information gain, and how decisions are made at each level of the tree. I learned that entropy is used to measure the uncertainty in data, and information gain helps in selecting the best attribute for splitting the dataset. After understanding the theory, I implemented the ID3 algorithm from scratch using Python, without using any machine learning libraries. This helped me clearly understand how the tree is built recursively by choosing the attribute with the highest information gain at every step. I also tested the model by predicting outcomes for new input values, which gave correct results. Through this task, I gained a clear practical understanding of how decision trees work internally and how mathematical measures guide the decision-making process. Overall, this task helped me connect theoretical concepts with hands-on implementation.
<img width="1917" height="989" alt="Screenshot 2026-01-06 081104" src="https://github.com/user-attachments/assets/6c6bb179-dba3-4147-acd0-4bb816f5eb61" />


<img width="870" height="523" alt="Screenshot 2026-01-06 080757" src="https://github.com/user-attachments/assets/e83f7c35-a337-4190-a1c3-276db80e8a81" />

<img width="755" height="572" alt="Screenshot 2026-01-06 080630" src="https://github.com/user-attachments/assets/b707febe-8d05-4087-a006-86097dbe1479" />
<img width="340" height="552" alt="Screenshot 2026-01-06 080527" src="https://github.com/user-attachments/assets/6bdac569-66d0-4ffb-82f5-8ed533fe9f7a" />


# Task 2

In this task, I studied and implemented the Naive Bayes classification algorithm with a focus on text classification. I first understood the theoretical concepts such as Bayes’ theorem, prior probability, likelihood, and the independence assumption used in Naive Bayes. Using the sklearn library, I implemented the Multinomial Naive Bayes classifier and observed how text data is converted into numerical form using the Bag of Words model. The trained model was able to correctly classify messages as spam or ham based on learned word patterns. To strengthen my understanding, I also implemented Naive Bayes from scratch using Python, which helped me clearly understand how word frequencies and probabilities are calculated internally. This task helped me understand why Naive Bayes works efficiently for text-based problems despite its simplifying assumptions. Overall, the experiment gave me both conceptual clarity and practical exposure to probabilistic classification.

<img width="932" height="575" alt="Screenshot 2026-01-18 140527" src="https://github.com/user-attachments/assets/c2773f82-d620-4f7b-8086-28282adb4268" />

<img width="938" height="638" alt="Screenshot 2026-01-18 140550" src="https://github.com/user-attachments/assets/a151db4a-e1ad-48ad-8f3e-d0c84858f3dd" />
<img width="1882" height="889" alt="Screenshot 2026-01-18 140634" src="https://github.com/user-attachments/assets/b058e1be-0a24-41e2-a2c0-cbcede21a49b" />

# Task3

I explored various ensemble learning techniques and applied them to the Titanic dataset to predict passenger survival. Ensemble techniques combine multiple machine learning models to improve performance by reducing bias and variance. I first preprocessed the Titanic data by handling missing values and encoding categorical features. Then I implemented several popular ensemble models including Random Forest, Gradient Boosting, and AdaBoost using the sklearn library. I also built a Stacking classifier that combined multiple base models with a logistic regression meta-learner. Each model was trained using the training data and evaluated using a validation set. The performance of each ensemble method was compared using accuracy scores, and a bar chart was created to visualize the results. Overall, ensemble methods showed strong performance and improved prediction accuracy compared to single models. This exercise helped me understand how ensemble techniques work in practice and how they can be beneficial for real-world classification tasks.

<img width="844" height="542" alt="Screenshot 2026-01-19 215314" src="https://github.com/user-attachments/assets/867e3a32-7b68-4dda-a07b-282775bba300" />

<img width="908" height="895" alt="Screenshot 2026-01-19 214626" src="https://github.com/user-attachments/assets/87137428-4650-458b-baec-1e302c9b591c" />
<img width="934" height="921" alt="Screenshot 2026-01-19 214305" src="https://github.com/user-attachments/assets/1de8e652-ac86-4b9d-a89a-7fed6bc985b1" />

<img width="957" height="537" alt="Screenshot 2026-01-19 215408" src="https://github.com/user-attachments/assets/67aacb62-a87c-4198-ba0d-146100ee66e6" />


# Task 4

In this task, I studied and implemented advanced tree-based ensemble algorithms including Random Forest, Gradient Boosting Machine (GBM), and XGBoost. I first understood the theoretical working of each algorithm and how they differ in handling bias, variance, and learning strategy. Using the Titanic dataset, I performed data preprocessing by handling missing values and encoding categorical features. Random Forest was implemented to reduce variance by combining multiple independent decision trees. GBM was used to improve prediction accuracy by sequentially learning from previous errors. XGBoost, being an optimized version of gradient boosting, provided efficient training with regularization to prevent overfitting. The performance of all models was evaluated using accuracy scores and compared visually. Through this task, I gained practical understanding of how ensemble learning improves model performance and why Random Forest, GBM, and XGBoost are widely used in real-world machine learning problems.
<img width="914" height="802" alt="Screenshot 2026-01-19 222419" src="https://github.com/user-attachments/assets/1ca54b7a-bd63-4a7b-b260-e3224c3d6c42" />

<img width="918" height="822" alt="Screenshot 2026-01-19 222358" src="https://github.com/user-attachments/assets/616c87aa-a1f4-4a43-a3a7-b629e68dad7a" />


# Task 5

In this task, I worked on hyperparameter tuning to improve the performance of a machine learning model. I selected the Titanic survival prediction problem and used a Random Forest classifier. Initially, the model was trained using default hyperparameters to obtain a baseline accuracy. After observing the baseline performance, I applied hyperparameter tuning techniques using GridSearchCV and RandomizedSearchCV. Parameters such as the number of trees, maximum depth, minimum samples split, and feature selection strategy were tuned to find the optimal combination. Cross-validation was used during tuning to ensure reliable performance evaluation. The tuned model showed an improvement in accuracy compared to the baseline model. This task helped me understand how hyperparameters influence model learning and how systematic tuning can significantly enhance model performance. Overall, I gained practical experience in optimizing machine learning models rather than relying on default settings.
<img width="1228" height="830" alt="Screenshot 2026-01-19 223538" src="https://github.com/user-attachments/assets/01c0d9e7-55c3-44c5-879c-2916ac0e1c95" />

<img width="1230" height="906" alt="Screenshot 2026-01-19 223529" src="https://github.com/user-attachments/assets/aab5e777-9906-4246-85bf-368f8a50305f" />

<img width="954" height="620" alt="Screenshot 2026-01-19 223507" src="https://github.com/user-attachments/assets/00dba897-f09d-4e10-9b69-8c5c801ca026" />


# Task 6

In this task, I studied and implemented image classification using the K-Means clustering algorithm. Since K-Means is an unsupervised learning algorithm, it does not use labeled data during training and instead identifies patterns based on similarity between data points.
For this experiment, I used the MNIST dataset, which consists of handwritten digit images from 0 to 9. Each image is represented as a 28×28 grayscale image and was converted into numerical feature vectors before applying the algorithm. The data was scaled to improve clustering performance. K-Means clustering was then applied with the number of clusters set to 10, corresponding to the ten digit classes.
After clustering, each cluster was mapped to the most frequent digit label present within it for evaluation purposes. The clustering results were evaluated by comparing the predicted cluster labels with the actual digit labels. The learned centroids were visualized as average digit images, which provided a clear understanding of how K-Means groups similar handwritten digits. Although the accuracy obtained was lower than supervised learning methods, the results were reasonable for an unsupervised approach. This task helped me understand how clustering algorithms can be applied to image data and how meaningful patterns can be discovered without using labeled information.
<img width="942" height="880" alt="Screenshot 2026-01-20 215309" src="https://github.com/user-attachments/assets/10920333-a04c-4b83-a14e-d4a4030dafe5" />
<img width="933" height="890" alt="Screenshot 2026-01-20 215323" src="https://github.com/user-attachments/assets/2d20973d-428d-4317-a863-5c6b39fae697" />

<img width="937" height="890" alt="Screenshot 2026-01-20 215529" src="https://github.com/user-attachments/assets/1214e830-1223-4f27-a8f3-05868c8af6ab" />

# Task7

From this task, this is what I learned and implemented. I understood the concept of anomaly detection and how it is used to identify abnormal or erroneous data points that differ from normal behavior. I learned the difference between supervised and unsupervised anomaly detection techniques and why unsupervised methods are commonly used in real-world applications. I generated a synthetic toy dataset using Python to simulate normal data and anomalies. I also learned how to visualize data using scatter plots to clearly identify outliers. This task helped me understand the complete workflow of anomaly detection from data creation to analysis.
<img width="921" height="871" alt="Screenshot 2026-01-22 080302" src="https://github.com/user-attachments/assets/d21cf60a-630d-4d4b-a708-e9f1fb99205a" />

# Task 8
In this task, I learnt the fundamental concepts of Generative Adversarial Networks (GANs) and how they are used in Generative AI to create realistic synthetic images. I understood the adversarial learning process involving two neural networks, namely the Generator and the Discriminator, where the generator learns to produce fake images from random noise while the discriminator learns to distinguish between real and generated images. I implemented a GAN model using the PyTorch framework and trained it on the MNIST handwritten digits dataset. Through this task, I learnt how to load and preprocess image datasets, define neural network architectures, apply loss functions, and use optimizers for training deep learning models. I also gained practical understanding of how the generator improves its output by learning from the discriminator’s feedback, a process often referred to as “fooling” the discriminator. During training, I observed how the generated images evolved from random noise to recognizable handwritten digits, demonstrating the effectiveness of adversarial training. This task helped me understand the challenges involved in training GANs and the importance of balanced training between the generator and discriminator. Overall, this task strengthened my understanding of generative models, deep learning workflows, and the real-world applications of GANs in image generation and data synthesis.
<img width="1224" height="793" alt="Screenshot 2026-01-22 170323" src="https://github.com/user-attachments/assets/837c1b7d-35b6-4106-a64c-ada165c97d74" />

<img width="1234" height="836" alt="Screenshot 2026-01-22 170307" src="https://github.com/user-attachments/assets/5da4940d-1a55-4e9a-a9c1-90a7574158ba" />

# Task 9




