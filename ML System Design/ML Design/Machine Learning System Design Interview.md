# Machine Learning System Design Interview

https://bytebytego.com/courses/machine-learning-system-design-interview/introduction-and-overview

# 01 Introduction & Overview

Components of a ML system for production.

![](../../../img/ML/ml design/1-1.png)

## ML system design framework

![Screenshot 2024-05-13 at 12.07.12 AM](/Users/yuwang/Library/Application Support/typora-user-images/Screenshot 2024-05-13 at 12.07.12 AM.png)

### Clarifying Requirements

Ask questions to understand the exact requirements:

- **Business objective**. If we are asked to create a system to recommend vacation rentals, two possible motivations are to increase the number of bookings and increase the revenue.
- **Features the system needs to support**. What are some of the features that the system is expected to support which could affect our ML system design? For example, let’s assume we’re asked to design a video recommendation system. We might want to know if users can “like” or “dislike” recommended videos, as those interactions could be used to label training data.
- **Data**. What are the data sources? How large is the dataset? Is the data labeled?
- **Constraints**. How much computing power is available? Is it a cloud-based system, or should the system work on a device? Is the model expected to improve automatically over time?
- **Scale of the system**. How many users do we have? How many items, such as videos, are we dealing with? What’s the rate of growth of these metrics?
- **Performance**. How fast must prediction be? Is a real-time solution expected? Does accuracy have more priority or latency?

It’s generally a good idea to write down the list of requirements and constraints we gather. By doing so, we ensure everyone is on the same page.

### Frame the Problem as an ML Task

Convert a problem to a real ML task

- Defining the ML objective
- Specifying the system’s input and output
- Choosing the right ML category

##### Defining the ML objective

Translate the business objective into a well-defined ML objective.

| Application                                          | Business objective                                  | ML objective                                     |
| :--------------------------------------------------- | :-------------------------------------------------- | :----------------------------------------------- |
| Event ticket selling app                             | Increase ticket sales                               | Maximize the number of event registrations       |
| Video streaming app                                  | Increase user engagement                            | Maximize the time users spend watching videos    |
| Ad click prediction system                           | Increase user clicks                                | Maximize click-through rate                      |
| Harmful content detection in a social media platform | Improve the platform's safety                       | Accurately predict if a given content is harmful |
| Friend recommendation system                         | Increase the rate at which users grow their network | Maximize the number of formed connections        |

##### Specifying the system’s input and output

In some cases, the system may consist of more than one ML model. If so, we need to specify the input and output of each ML model. There might also be multiple ways to specify each model’s input-output.

##### Choosing the right ML category

Supervised / Unsupervised

Classification / Regression

##### Talking points

- What is a good ML objective? How do different ML objectives compare? What are the pros and cons?
- What are the inputs and outputs of the system, given the ML objective?
- If more than one model is involved in the ML system, what are the inputs and outputs of each model?
- Does the task need to be learned in a supervised or unsupervised way?
- Is it better to solve the problem using a regression or classification model? In the case of classification, is it binary or multiclass? In the case of regression, what is the output range?

### Data Preparation

![](../../../img/ML/ml design/1-2.png)

##### Data Engineering

- Data source?
- New data coming in?
- Data storage?

- Data schema?

  - Structured

  - Unstructured

##### Feature Engineering

- Handling missing values
  - Deletion - Reduce data quantity
  - Imputation: Filling in missing values with their defaults, mean, median, or mode - Introduce noise
- Feature scaling
  - Normalization: $z=\frac{x-x_{min}}{x_{max}-x_{min}}$ - values are scaled into [0, 1]
  - Standardization: $z=\frac{x-\mu}{\sigma}$ - normal distribution

- Discretization

- Encoding categorical features
  - Integer encoding
  - One-hot encoding

##### Talking points

- **Data availability and data collection:** What are the data sources? What data is available to us, and how do we collect it? How large is the data size? How often do new data come in?
- **Data storage:** Where is the data currently stored? Is it on the cloud or on user devices? Which data format is appropriate for storing the data? How do we store multimodal data, e.g., a data point that might contain both images and texts?
- **Feature engineering:** How do we process raw data into a form that’s useful for the models? What should we do about missing data? Is feature engineering required for this task? Which operations do we use to transform the raw data into a format usable by the ML model? Do we need to normalize the features? Which features should we construct from the raw data? How do we plan to combine data of different types, such as texts, numbers, and images?
- **Privacy:** How sensitive are the available data? Are users concerned about the privacy of their data? Is anonymization of user data necessary? Is it possible to store users’ data on our servers, or is it only possible to access their data on their devices?
- **Biases:** Are there any biases in the data? If yes, what kinds of biases are present, and how do we correct them?

### Model Development

##### Model Selection

In practice, a typical process for selecting a model is to:

- **Establish a simple baseline.** For example, in a video recommendation system, the baseline can be obtained by recommending the most popular videos.
- **Experiment with simple models.** After we have a baseline, a good practice is to explore ML algorithms that are quick to train, such as logistic regression.
- **Switch to more complex models.** If simple models cannot deliver satisfactory results, we can then consider more complex models, such as deep neural networks.
- **Use an ensemble of models if we want more accurate predictions.** Using an ensemble of multiple models instead of only one may improve the quality of predictions. Creating an ensemble can be accomplished in three ways: bagging [3], boosting [4], and stacking [5], which will be discussed in later chapters.

Some typical model options include:

- Logistic regression
- Linear regression
- Decision trees
- Gradient boosted decision trees and random forests
- Support vector machines
- Naive Bayes
- Factorization Machines (FM)
- Neural networks

When choosing an ML algorithm, it’s important to consider different aspects of a model. For example:

- The amount of data the model needs to train on
- Training speed
- Hyperparameters to choose and hyperparameter tuning techniques
- Possibility of continuous learning
- Compute requirements. A more complex model might deliver higher accuracy, but might require more computing power, such as a GPU instead of a CPU
- Model’s interpretability [6]. A more complex model can give better performance, but its results may be less interpretable

##### Model Training

- Constructing dataset

  ![](../../../img/ML/ml design/1-3.png)

  - Address any class imbalances
    - Upsampling or downsampling
    - Adjust weight in loss

- Choosing loss function

- Training from scratch or fine-tuning

- Distributed training

##### Talking points

- Model selection: Which ML models are suitable for the task, and what are their pros and cons. Here’s a list of topics to consider during model selection:
  - The time it takes to train
  - The amount of training data the model expects
  - The computing resources the model may need
  - Latency of the model at inference time
  - Can the model be deployed on a user’s device?
  - Model’s interpretability. Making a model more complex may increase its performance, but the results might be harder to interpret
  - Can we leverage continual training, or should we train from scratch?
  - How many parameters does the model have? How much memory is needed?
  - For neural networks, you might want to discuss typical architectures/blocks, such as ResNet or Transformer-based architectures. You can also discuss the choice of hyperparameters, such as the number of hidden layers, the number of neurons, activation functions, etc.
- Dataset labels: How should we obtain the labels? Is the data annotated, and if so, how good are the annotations? If natural labels are available, how do we get them? How do we receive user feedback on the system? How long does it take to get natural labels?
- Model training.
  - What loss function should we choose? (e.g., Cross-entropy [15], MSE [16], MAE [17], Huber loss [18], etc.)
  - What regularization should we use? (e.g., L1 [19], L2 [19], Entropy Regularization [20], K-fold CV [21], or dropout [22])
  - What is backpropagation?
  - You may need to describe common optimization methods [23] such as SGD [24], AdaGrad [25], Momentum [26], and RMSProp [27].
  - What activation functions do we want to use and why? (e.g., ELU [28], ReLU [29], Tanh [30], Sigmoid [31]).
  - How to handle an imbalanced dataset?
  - What is the bias/variance trade-off?
  - What are the possible causes of overfitting and underfitting? How to address them?

### Evaluation

##### Offline

| Task                        | Offline metrics                                              |
| :-------------------------- | :----------------------------------------------------------- |
| Classification              | Precision, recall, F1 score, accuracy, ROC-AUC, PR-AUC, confusion matrix |
| Regression                  | MSE, MAE, RMSE                                               |
| Ranking                     | Precision@k, recall@k, MRR, mAP, nDCG                        |
| Image generation            | FID [32], Inception score [33]                               |
| Natural language processing | BLEU [34], METEOR [35], ROUGE [36], CIDEr [37], SPICE [38]   |

##### Online

- Connected with A/B testing

| Problem                   | Online metrics                                               |
| :------------------------ | :----------------------------------------------------------- |
| Ad click prediction       | Click-through rate, revenue lift, etc.                       |
| Harmful content detection | Prevalence, valid appeals, etc.                              |
| Video recommendation      | Click-through rate, total watch time, number of completed videos, etc. |
| Friend recommendation     | Number of requests sent per day, number of requests accepted per day, etc. |

##### Talking points

Here are some talking points for the evaluation step:

- **Online metrics:** Which metrics are important for measuring the effectiveness of the ML system online? How do these metrics relate to the business objective?
- **Offline metrics:** Which offline metrics are good at evaluating the model’s predictions during the development phase?
- **Fairness and bias:** Does the model have the potential for bias across different attributes such as age, gender, race, etc.? How would you fix this? What happens if someone with malicious intent gets access to your system?

### Deployment

|                                   | **Cloud**                                                    | **On-device**                                                |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Simplicity                        | ✓ Simple to deploy and manage using cloud-based services     | ✘ Deploying models on a device is not straightforward        |
| Cost                              | ✘ Cloud costs might be high                                  | ✓ No cloud cost when computations are performed on-device    |
| Network latency                   | ✘ Network latency is present                                 | ✓ No network latency                                         |
| Inference latency                 | ✓ Usually faster inference due to more powerful machines     | ✘ ML models run slower                                       |
| Hardware constraints              | ✓ Fewer constraints                                          | ✘ More constraints, such as limited memory, battery consumption, etc. |
| Privacy                           | ✘ Less privacy as user data is transferred to the cloud      | ✓ More privacy since data never leaves the device            |
| Dependency on internet connection | ✘ Internet connection needed to send and receive data to the cloud | ✓ No internet connection needed                              |

##### Shadow Deployment

In this method, we deploy the new model in parallel with the existing model. Each incoming request is routed to both models, but only the existing model's prediction is served to the user.

By shadow deploying the model, we minimize the risk of unreliable predictions until the newly developed model has been thoroughly tested. However, this is a costly approach that doubles the number of predictions.

##### A/B

With this method, we deploy the new model in parallel with the existing model. A portion of the traffic is routed to the newly developed model, while the remaining requests are routed to the existing model.

In order to execute A/B testing correctly, there are two important factors to consider. First, the traffic routed to each model has to be random. Second, A/B tests should be run on a sufficient number of data points in order for the results to be legitimate.

##### Prediction Pipeline

- **Batch prediction.** With batch prediction, the model makes predictions periodically. Because predictions are pre-computed, we don’t need to worry about how long it takes the model to generate predictions once they are pre-computed.

However, the batch prediction has two major drawbacks. First, the model becomes less responsive to the changing preferences of users. Secondly, batch prediction is only possible if we know in advance what needs to be pre-computed. For example, in a language translation system, we are not able to make translations in advance as it entirely depends on the user’s input.

- **Online prediction.** In online prediction, predictions are generated and returned as soon as requests arrive. The main problem with online prediction is that the model might take too long to generate predictions.

This choice of batch prediction or online prediction is mainly driven by product requirements. Online prediction is generally preferred in situations where we do not know what needs to be computed in advance. Batch prediction is ideal when the system processes a high volume of data, and the results are not needed in real time.

##### Talking points

- Is model compression needed? What are some commonly used compression techniques?
- Is online prediction or batch prediction more suitable? What are the trade-offs?
- Is real-time access to features possible? What are the challenges?
- How should we test the deployed model in production?
- An ML system consists of various components working together to serve requests. What are the responsibilities of each component in the proposed design?
- What technologies should we use to ensure that serving is fast and scalable?

## Summary

In this chapter, we proposed a framework for an ML system design interview. While many topics discussed in this chapter are task-specific, some are generic and applicable to a wide range of tasks. Throughout this book, we only focus on unique talking points specific to the problem at hand, in order to avoid repetition. For example, topics related to deployment, monitoring, and infrastructure are often similar, regardless of the task. Therefore, we do not repeat generic topics in later chapters, but you are usually expected to talk about them during an interview.

Finally, no engineer can be an expert in every aspect of the ML lifecycle. Some engineers specialize in deployment and production, while others specialize in model development. Some companies may not care about infrastructure, while others may focus heavily on monitoring and infrastructure. Data science roles generally require more data engineering, while applied ML roles focus more on model development and productionization. Depending on the role and the interviewer's preference, some steps may be discussed in more detail, while others may be discussed briefly or even skipped. In general, a candidate should seek to drive the conversation, while being ready to go with the interviewer’s flow, if they raise a question.

Now you understand these fundamentals, we’re ready to tackle some of the most common ML system design interview questions.

## References

1. Data warehouse. https://cloud.google.com/learn/what-is-a-data-warehouse.
2. Structured vs. unstructured data. https://signal.onepointltd.com/post/102gjab/machine-learning-libraries-for-tabular-data-problems.
3. Bagging technique in ensemble learning. https://en.wikipedia.org/wiki/Bootstrap_aggregating.
4. Boosting technique in ensemble learning. https://aws.amazon.com/what-is/boosting/.
5. Stacking technique in ensemble learning. https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/.
6. Interpretability in Machine Mearning. https://blog.ml.cmu.edu/2020/08/31/6-interpretability/.
7. Traditional machine learning algorithms. https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/.
8. Sampling strategies. https://www.scribbr.com/methodology/sampling-methods/.
9. Data splitting techniques. https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/.
10. Class-balanced loss. https://arxiv.org/pdf/1901.05555.pdf.
11. Focal loss paper. https://arxiv.org/pdf/1708.02002.pdf.
12. Focal loss. https://medium.com/swlh/focal-loss-an-efficient-way-of-handling-class-imbalance-4855ae1db4cb.
13. Data parallelism. https://www.telesens.co/2017/12/25/understanding-data-parallelism-in-machine-learning/.
14. Model parallelism. https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-intro.html.
15. Cross entropy loss. https://en.wikipedia.org/wiki/Cross_entropy.
16. Mean squared error loss. https://en.wikipedia.org/wiki/Mean_squared_error.
17. Mean absolute error loss. https://en.wikipedia.org/wiki/Mean_absolute_error.
18. Huber loss. https://en.wikipedia.org/wiki/Huber_loss.
19. L1 and l2 regularization. https://www.analyticssteps.com/blogs/l2-and-l1-regularization-machine-learning.
20. Entropy regularization. https://paperswithcode.com/method/entropy-regularization.
21. K-fold cross validation. https://en.wikipedia.org/wiki/Cross-validation_(statistics).
22. Dropout paper. https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf.
23. Overview of optimization algorithm. https://ruder.io/optimizing-gradient-descent/.
24. Stochastic gradient descent. https://en.wikipedia.org/wiki/Stochastic_gradient_descent.
25. AdaGrad optimization algorithm. https://optimization.cbe.cornell.edu/index.php?title=AdaGrad.
26. Momentum optimization algorithm. https://optimization.cbe.cornell.edu/index.php?title=Momentum.
27. RMSProp optimization algorithm. https://optimization.cbe.cornell.edu/index.php?title=RMSProp.
28. ELU activation function. https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu.
29. ReLU activation function. https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu.
30. Tanh activation function. https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh.
31. Sigmoid activation function. https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#softmax.
32. FID score. [https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance](https://en.wikipedia.org/wiki/Fréchet_inception_distance).
33. Inception score. https://en.wikipedia.org/wiki/Inception_score.
34. BLEU metrics. https://en.wikipedia.org/wiki/BLEU.
35. METEOR metrics. https://en.wikipedia.org/wiki/METEOR.
36. ROUGE score. https://en.wikipedia.org/wiki/ROUGE_(metric).
37. CIDEr score. https://arxiv.org/pdf/1411.5726.pdf.
38. SPICE score. https://arxiv.org/pdf/1607.08822.pdf.
39. Quantization-aware training. https://pytorch.org/docs/stable/quantization.html.
40. Model compression survey. https://arxiv.org/pdf/1710.09282.pdf.
41. Shadow deployment. [https://christophergs.com/machine%20learning/2019/03/30/deploying-machine-learning-applications-in-shadow-mode/](https://christophergs.com/machine learning/2019/03/30/deploying-machine-learning-applications-in-shadow-mode/).
42. A/B testing. https://en.wikipedia.org/wiki/A/B_testing.
43. Canary release. https://blog.getambassador.io/cloud-native-patterns-canary-release-1cb8f82d371a.
44. Interleaving experiment. https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55.
45. Multi-armed bandit. https://vwo.com/blog/multi-armed-bandit-algorithm/.
46. ML infrastructure. https://www.run.ai/guides/machine-learning-engineering/machine-learning-infrastructure.
47. Interpretability in ML. https://fullstackdeeplearning.com/spring2021/lecture-6/.
48. Chip Huyen. *Designing Machine Learning Systems: An Iterative Process for Production-Ready Application*. ” O’Reilly Media, Inc.”, 2022.

# 02 Visual Search System

![](../../../img/ML/ml design/2-1.png)

## Clarifying Requirements

**Candidate**: Should we rank the results from most similar to least similar?
**Interviewer**: Images that appear first in the list should be more similar to the query image.

**Candidate**: Should the system support videos, too?
**Interviewer**: Let’s focus only on images.

**Candidate**: A platform like Pinterest allows users to select an image crop and retrieve similar images. Should we support that functionality?
**Interviewer**: Yes.

**Candidate**: Are the displayed images personalized to the user?
**Interviewer**: For simplicity, let’s not focus on personalization. A query image yields the same results, regardless of who searches for it.

**Candidate**: Can the model use the metadata of the query image, such as image tags?
**Interviewer**: In practice, the model uses image metadata. But for simplicity, let’s assume we don’t rely on the metadata, but only on the image pixels.

**Candidate**: Can users perform other actions such as save, share, or like? These actions can help label training data.
**Interviewer**: Great point. For simplicity, let’s assume the only supported action is image clicks.

**Candidate**: Should we moderate the images?
**Interviewer**: It’s important to keep the platform safe, but content moderation is out of scope.

**Candidate**: We can construct training data online and label them based on user interactions. Is this the expected way to construct training data?
**Interviewer**: Yes, that sounds reasonable.

**Candidate**: How fast should the search be? Assuming we have 100-200 billion images on the platform, the system should be able to retrieve similar images quickly. Is that a reasonable assumption?
**Interviewer**: Yes, that is a reasonable assumption.

Sumamrize:

- Design a visual search system that retrieves images similar to the query image provided by the user
- Ranks them based on their similarities to the query image, and then displays them to the user
- Only supports images, with no video or text queries allowed. For simplicity, no personalization is required.

## Frame the Problem as an ML Task

### Defining the ML objective

Accurately retrieve images that are visually similar to the image the user is searching for.

### Specifying input & output

Input: query image from user

Output: similar images ranked by similarities

### Choosing the righe ML category

Ranking + Representation Learning

## Data Preparation

### Date engineering

##### Images

Creators upload images, and the system stores the images and their metadata, such as owner id, contextual information (e.g., upload time), tags, etc. Table 2.1 shows a simplified example of image metadata.

| ID   | Owner ID | Upload time | Manual tags             |
| :--- | :------- | :---------- | :---------------------- |
| 1    | 8        | 1658451341  | Zebra                   |
| 2    | 5        | 1658451841  | Pasta, Food, Kitchen    |
| 3    | 19       | 1658821820  | Children, Family, Party |

##### Users

User data contains demographic attributes associated with users, such as age, gender, etc. Table 2.2 shows an example of user data.

| **ID** | **Username** | **Age** | **Gender** | **City** | **Country** | **Email**                                 |
| ------ | ------------ | ------- | ---------- | -------- | ----------- | ----------------------------------------- |
| 1      | johnduo      | 26      | M          | San Jose | USA         | [john@gmail.com](mailto:john@gmail.com)   |
| 2      | hs2008       | 49      | M          | Paris    | France      | [hsieh@gmail.com](mailto:hsieh@gmail.com) |
| 3      | alexish      | 16      | F          | Rio      | Brazil      | [alexh@yahoo.com](mailto:alexh@yahoo.com) |

##### User-image interactions

Interaction data contains different types of user interactions. Based on the requirements gathered, the primary types of interactions are impressions and clicks. Table 2.3 shows an overview of interaction data.

| User ID | Query image ID | Displayed image ID | Position in the displayed list | Interaction type | Location (lat, long) | Timestamp  |
| :------ | :------------- | :----------------- | :----------------------------- | :--------------- | :------------------- | :--------- |
| 8       | 2              | 6                  | 1                              | Click            | 38.8951 -77.0364     | 1658450539 |
| 6       | 3              | 9                  | 2                              | Click            | 38.8951 -77.0364     | 1658451341 |
| 91      | 5              | 1                  | 2                              | Impression       | 41.9241 -89.0389     | 1658451365 |

### Feature Engineering

Common image preprocessing operations:

- **Resizing:** Models usually require fixed image sizes (e.g., 224×224)
- **Scaling:** Scale pixel values of the image to the range of 0 and 1
- **Z-score normalization:** Scale pixel values to have a mean of 0 and variance of 1
- **Consistent :** Ensuring images have a consistent color mode (e.g., RGB or CMYK)

## Model Development

### Model Selection

CNN-based or Transformer-based

### Model Training

A common technique for learning image representations is contrastive training.

#### Constructing Dataset

Each data sample: (query image, 1 similar image and n-1 dissimilar images, idx of the similar image)

![](../../../img/ML/ml design/2-2.png)

Sources of the positive image:

- Human Annotation
- Data Augmentation: rotating / noise - less human work

==In an interview setting, it’s critical you propose various options and discuss their tradeoffs. Discussing different options and trade-offs with the interviewer is critical to make good design decisions.==

#### Choosing loss

Contrastive loss [9]

![](../../../img/ML/ml design/2-3.png)

1. Compute similarities between embeddings of query image and other images.
2. Softmax
3. Cross-entropy

## Evaluation

### Offline metrics

==Search, information retrieval, and recommendation systems usually share the same offline metrics.==

- Mean reciprocal rank (MRR)
- Recall@k
- Precision@k
- Mean average precision (mAP)
- Normalized discounted cumulative gain (nDCG)

Testing data sample: 

![](../../../img/ML/ml design/2-4.png)

#### MRR

$$
MRR=\frac{1}{m}\sum^m_{i=1}\frac{1}{rank_i}
$$

m is the total number of output lists and rank_i refers to the rank of the first relevant item in the *i*th output list.

This does not measure the ranking.

#### Recall@K

**The ratio between the number of relevant items in the output list and the total number of relevant items available in the entire dataset.** Can be negatively affected when the number of relevant items is very large.

Does not measure the ranking.

#### Precision@K

**The proportion of relevant items among the top k items in the output list.**

Does not measure the ranking.

#### mAP

![](../../../img/ML/ml design/2-5.png)

Measures the ranking, but only works for binary relevances, i.e., when items is either relevant or irrelevant.

#### nDCG (Normalized discounted cumulative gain)

Measures the ranking quality of an output list and shows how good the ranking is, compared to the ideal ranking.

DCG: 
$$
DCG_p=\sum^p_{i=1}\frac{rel_i}{\log_2(i+1)}
$$
$rel_i$ is the ground truth relevance score of the image ranked at location i

nDCG = DCG / IDCG, IDCG is ideal DCG (DCG for ideal ranking)

![](../../../img/ML/ml design/2-6.png)

### Online metrics

#### CTR (Click-through rate)

CTR = Number of clicked images / Total number of suggested images

## Serving

![](../../../img/ML/ml design/2-7.png)

Embedding generation service: generate embeddings for input query image

Nearest neighbor service: find similair images from from embedding space.

## Other Talking Points

- Moderate content in the system by identifying and blocking inappropriate images [24].
- Different biases present in the system, such as positional bias [25][26].
- How to use image metadata such as tags to improve search results. This is covered in Chapter 3 Google Street View Blurring System.
- Smart crop using object detection [27].
- How to use graph neural networks to learn better representations [28].
- Support the ability to search images by a textual query. We examine this in Chapter 4.
- How to use active learning [29] or human-in-the-loop [30] ML to annotate data more efficiently.

 # 03 Google Street View Blurring System

## Clarification

**Candidate:** Is it fair to say the business objective of the system is to protect user privacy?
**Interviewer:** Yes.

**Candidate:** We want to design a system that detects all human faces and license plates in Street View images and blurs them before displaying them to users. Is that correct? Can I assume users can report images that are not correctly blurred?
**Interviewer:** Yes, those are fair assumptions.

**Candidate:** Do we have an annotated dataset for this task?
**Interviewer:** Let's assume we have sampled 1 million images. Human faces and license plates are manually annotated in those images.

**Candidate:** The dataset may not contain faces from certain racial profiles, which may cause a bias towards certain human attributes such as race, age, gender, etc. Is that a fair assumption?
**Interviewer:** Great point. For simplicity, let's not focus on fairness and bias today.

**Candidate:** My understanding is that latency is not a big concern, as the system can detect objects and blur them offline. Is that correct?
**Interviewer:** Yes. We can display existing images to users while new ones are being processed offline.

Let's summarize the problem statement. We want to design a Street View blurring system that automatically blurs license plates and human faces. We are given a training dataset of 1 million images with annotated human faces and license plates. The business objective of the system is to protect user privacy.

## Frame the Problem as an ML Task

### Defining the ML objective

One possible ML objective is to accurately detect objects of interest in an image. If an ML system can detect those objects accurately, then we can blur the objects before displaying the images to users.

### Specifying input and output

The input of an object detection model is an image with zero or multiple objects at different locations within it. The model detects those objects and outputs their locations.

### Choosing the right ML category

Two-stage solution:

1. **Region proposal network:** scans an image and proposes candidate regions that are likely to be objects.
2. **Classifier:** processes each proposed region and classifies it into an object class.

## Data Preparation

### Data engineering

we have the following data available:

- Annotated dataset

  | **Image path**     | **Objects**                         | **Bounding boxes**                          |
  | ------------------ | ----------------------------------- | ------------------------------------------- |
  | dataset/image1.jpg | human face human face license plate | [10,10,25,50] [120,180,40,70] [80,95,35,10] |
  | dataset/image2.jpg | human face                          | [170,190,30,80]                             |
  | dataset/image3.jpg | license plate human face            | [25,30,210,220] [30,40,30,60]               |

- Street View images

### Feature engineering

1. Apply standard methods, such as resizing and normalization

2. Data augmentation

- Random crop
- Random saturation
- Vertical or horizontal flip
- Rotation
- Changing brightness, saturation, or contrast

​	Remember to change the ground truth bounding boxes along with the pictures.

**Online vs. offline:** In offline data augmentation, training is faster since no additional augmentation is needed. However, it requires additional storage to store all the augmented images. While online data augmentation slows down training, it does not consume additional storage.

The choice between online and offline data augmentation depends upon the storage and computing power constraints. ==What is more important in an interview is that you talk about different options and discuss trade-offs==

## 

## Model Development

### Model Selection

For both RPN and Classifier, we go with NN.

### Model Training

For RPN, we use regression loss, such as Mean Squared Error.

For Classifier, we use cross-entropy.

## Evaluation

An object detection model usually needs to detect N different objects in an image. To measure the overall performance of the model, we evaluate each object separately and then average the results.

**Intersection Over Union (IOU):** IOU measures the overlap between two bounding boxes. Figure 3.93.9 shows a visual representation of IOU.

IOU determines whether a detected bounding box is correct. An IOU of 1 is ideal, indicating the detected bounding box and the ground truth bounding box are fully aligned. In practice, it's rare to see an IOU of 1 . A higher IOU means the predicted bounding box is more accurate. An IOU threshold is usually used to determine whether a detected bounding box is correct (true positive) or incorrect (false positive). For example, an IOU threshold of 0.7 means any detection that has an overlap of 0.7 or higher with a ground truth bounding box, is a correct detection.

### Offline Metrics

- Precision
- Average precision: average of precision along all the choices of IOU threshold
- Mean average precision: average of AP along all the classes of objects

### Online Metrics

- User reports and complaints.

## Serving

**NMS** (Non-maximum suppression) is a post-processing algorithm designed to select the most appropriate bounding boxes. It keeps highly confident bounding boxes and removes overlapping bounding boxes. 

![](../../../img/ML/ml design/3-1.png)

##### Batch prediction pipeline

Based on the requirements gathered, latency is not a big concern because we can display existing images to users while new ones are being processed. Since instant results are not required, we can utilize batch prediction and precompute the object detection results.

**Preprocessing** Raw images are preprocessed by this component. This section does not discuss the preprocess operations as we have already discussed them in the feature engineering section.

**Blurring service** This performs the following operations on a Street View image:

1. Provides a list of objects detected in the image.
2. Refines the list of detected objects using the NMS component.
3. Blurs detected objects.
4. Stores the blurred image in object storage (Blurred Street View images).

Note that the preprocessing and blurring services are separate in the design. The reason is preprocessing images tends to be a CPU-bound process, whereas blurring service relies on GPU. Separating these services has two benefits:

- Scale the services independently based on the workload each receives.
- Better utilization of CPU and GPU resources.

##### Data pipeline

This pipeline is responsible for processing users' reports, generating new training data, and preparing training data to be used by the model. Data pipeline components are mostly self-explanatory. Hard negative mining is the only component that needs more explanation.

# 04 Youtube Video Search

## Clarifying Requirements

**Candidate:** Is the input query text-only, or can users search with an image or video?
**Interviewer:** Text queries only.

**Candidate:** Is the content on the platform only in video form? How about images or audio files?
**Interviewer:** The platform only serves videos.

**Candidate:** The YouTube search system is very complex. Can I assume the relevancy of a video is determined solely by its visual content and the textual data associated with the video, such as the title and description?
**Interviewer:** Yes, that's a fair assumption.

**Candidate:** Is there any training data available?
**Interviewer:** Yes, let's assume we have ten million pairs of ⟨⟨ video, text query ⟩⟩.

**Candidate:** Do we need to support other languages in the search system?
**Interviewer:** For simplicity, let's assume only English is supported.

**Candidate:** How many videos are available on the platform?
**Interviewer:** One billion videos.

**Candidate:** Do we need to personalize the results? Should we rank the results differently for different users, based on their past interactions?
**Interviewer:** As opposed to recommendation systems where personalization is essential, we do not necessarily have to personalize results in search systems. To simplify the problem, let's assume no personalization is required.

## Frame the Problem as an ML Task

### Defining ML objective

One way to translate this into an ML objective is to rank videos based on their relevance to the text query.

### Specifying input and output

Takes a text query as input and outputs a ranked list of videos sorted by their relevance to the text query.

### Choosing the right ML category

![](../../../img/ML/ml design/4-1.png)

Visual Search: Get embeddings for videos and text queries, and compute the similarity score by dot product. Then we rank the videos based on their similarity scores.

Test search: Commonly uses "inverted index" to search video titles and descriptions. **No need to use ML models.** (Can also discuss trade-off here)

## Data Preparation

### Data engineering

Training data ready: <video, query>

### Feature engineering

Preprocess video: decode frames, sample frames, resizing, and normalizing

## Model Development

### Model Selection

Text encoder: Bert/LLM

Video encoder

- video model: slower but better performative
- frame model: faster, computational efficient but less performative

### Model Development

![](../../../img/ML/ml design/4-2.png)

## Evaluation

- MRR (Mean Reciprocal Rank)
  $$
  MRR=\frac{1}{m}\sum^m_{i=1} \frac{1}{rank_i}
  $$
  
- Online metrics
  - **CTR**
  - **Video completion rate**
  - **Total watch time of search results.**

## Serving

![](../../../img/ML/ml design/4-3.png)

**Fusing layer.** This component takes two different lists of relevant videos from the previous step, and combines them into a new list of videos.

The fusing layer can be implemented in two ways, the easiest of which is to re-rank videos based on the weighted sum of their predicted relevance scores. A more complex approach is to adopt an additional model to re-rank the videos, which is more expensive because it requires model training. Additionally, it's slower at serving. As a result, we use the former approach.

## Other talking points

Before concluding this chapter, it's important to note we have simplified the system design of the video search system. In practice, it is much more complex. Some improvements may include:

- Use a multi-stage design (candidate generation + ranking).
- Use more video features such as video length, video popularity, etc.
- Instead of relying on annotated data, use interactions (e.g., clicks, likes, etc.) to construct and label data. This allows us to continuously train the model.
- Use an ML model to find titles and tags which are semantically similar to the text query. This model can be combined with Elasticsearch to improve search quality.

If there's time left at the end of the interview, here are some additional talking points:

- An important topic in search systems is query understanding, such as spelling correction, query category identification, and entity recognition. How to build a query understanding component? 
- How to build a multi-modal system that processes speech and audio to improve search results.
- How to extend this work to support other languages.
- Near-duplicate videos in the final output may negatively impact user experience. How to detect near-duplicate videos so we can remove them before displaying the results.
- Text queries can be divided into head, torso, and tail queries. What are the different approaches commonly used in each case.
- How to consider popularity and freshness when producing the output list.

# 05 Harmful Content Detection

## Clarification

**Candidate:** Does the system detect both harmful content and bad actors?
**Interviewer:** Both are equally important. For simplicity, let's focus on detecting harmful content only.

**Candidate:** Should a post only contain text, or are images and videos allowed?
**Interviewer:** The content of a post can be text, image, video, or any combination of these.

**Candidate:** What languages are supported? Is it English only?
**Interviewer:** The system should detect harmful content in various languages. For simplicity, assume we can use a pre-trained multilingual model to embed the textual content.

**Candidate**: Which specific categories of harmful content are we looking to identify? I can think of violence, nudity, hate speech, misinformation, etc. Are there other harmful categories to consider?
**Interviewer:** Great, you brought up the major ones. Misinformation is more complex and controversial. For simplicity, let's not focus on misinformation.

**Candidate**: Are there any human annotators available to label posts manually?
**Interviewer:** The platform receives more than 500 million posts each day. Asking humans to label all of them would be very expensive and time-consuming. However, you can assume human annotation is available to label a limited number of posts, say 10,000 per day.

**Candidate:** Allowing users to report harmful content is beneficial for understanding where the system is failing. Can I assume the system has that feature?
**Interviewer:** Good point. Yes, users can report harmful posts.

**Candidate:** Should we explain why a post is deemed harmful and removed?
**Interviewer:** Yes. Explaining to users why we remove a post is essential. It helps users to ensure they align their future posts with the guidelines.

**Candidate:** What is the system's latency requirement? Do we need a real-time prediction, i.e., the system detects harmful content immediately and blocks it, or can we rely on batch prediction, i.e., detecting harmful content offline hourly or daily?
**Interviewer:** This is a very important question. What are your thoughts?

**Candidate:** In my opinion, the requirements for different harmful content might vary. For example, violent content may require real-time solutions, while for others, late detection may work.
**Interviewer:** Those are fair assumptions.

## Frame the Problem as an ML Task

### ML Objective

We define our ML objective as accurately predicting harmful posts.

### Input and Output

- Input: post (multi-modal)
- Output: probability that the post is harmful

### Choosing the right ML category

In this section, we examine the following ML category options:

- Single binary classifier
- One binary classifier per harmful class
  - Pros: well explained; models can be developed and replaced separately
  - Cons: multiple models to maintain
- Multi-label classifier
- Multi-task classifier: one head for each type of harmfulness
  - Pros: First, it is not expensive to train or maintain since we use a single model. Second, the shared layers transform the features in a way that is beneficial for each task. This prevents redundant computations and makes multi-task classification efficient. Lastly, the training data for each task contributes to the learning of other tasks. This is especially helpful when limited data is available for a particular task.

## Data Preparation

### Data engineering

- Users
- Posts
  - Including text, pictures, audio, etc
- User-Post interactions
  - Like, comment, dislike, report, ...

### Feature engineering

- **The number of likes, shares, comments, and reports:** We usually scale these numerical values to speed up convergence during model training.

- ##### Author features

  The author's past interactions can be used to determine if the post is harmful or not. Let's engineer features related to the post author.

  **Author's violation history**

  - **Number of violations:** This is a numerical value representing the number of times the author violated the guidelines in the past.
  - **Total user reports:** A numerical value representing the number of times users reported the author's posts.
  - **Profane words rate:** This is a numerical value representing the rate of profane words used in the author's previous posts and comments. A predefined list of profane words is used to determine whether a word is profane.

  **Author's demographics**

  - **Age:** A user's age is one of the most important predictive features.
  - **Gender:** This categorical feature represents the user's gender. We use one-hot encoding to represent gender.
  - **City and country:** Both the city and country take many distinct values. To represent the features, we use an embedding layer to convert city and country into feature vectors. Note that one-hot encoding is not an efficient method to represent the city and country because their representations would be long and sparse.

  **Account information**

  - **Number of followers and followings**
  - **Account age:** This is a numerical value representing the age of the author's account. This is a predictive feature as accounts with a lower age are more likely to be spam or to violate integrity.

  ##### Contextual information

  - **Time of day:** This is the time of day when the author made a post. We bucketize this into multiple categories, such as morning, noon, afternoon, evening or night. We use one-hot encoding to represent this feature.

## Model Development

### Model Selection

NN

### Model Training

#### Constructing Dataset

To train the multi-task classification model, we first need to construct the dataset. The dataset comprises model inputs (features) and outputs (labels) that the model is expected to predict. To construct inputs, we process posts offline in batches and compute fused features as described earlier. These features can be stored in a feature store for future training. In order to create labels for each input, we have two options:

- Hand labeling
- Natural labeling

With hand labeling, human contractors label posts manually. This option produces accurate labels, but it is expensive and time-consuming. With natural labeling, we rely on user reports to label posts automatically. While this option results in noisier labels, labels are produced more quickly. For the evaluation dataset, we use hand labeling to prioritize the accuracy of labels, and for the training dataset, we use natural labeling to prioritize labeling speed.

#### Choosing loss function

Binary CR for each category. L = L1 + L2 + ...

## Evaluation

Offline: Precision, Recall

Online: **Harmful impressions.** **Valid appeals.** 

## Serving

![6-1](../../../img/ML/ml design/6-1.png)

# 06 Video Recommendation System

## Clarification

**Candidate:** Can I assume the business objective of building a video recommendation system is to increase user engagement?
**Interview**: That’s correct.

**Candidate:** Does the system recommend similar videos to a video a user is watching right now? Or does it show a personalized list of videos on the user’s homepage?
**Interviewer:** This is a homepage video recommendation system, which recommends personalized videos to users when they load the homepage.

**Candidate:** Since YouTube is a global service, can I assume users are located worldwide and videos are in different languages?
**Interviewer:** That’s a fair assumption.

**Candidate:** Can I assume we can construct the dataset based on user interactions with video content?
**Interviewer:** Yes, that sounds good.

**Candidate:** Can a user group videos together by creating playlists? Playlists can be informative for the ML model during the learning phase.
**Interviewer:** For the sake of simplicity, let’s assume the playlist feature does not exist.

**Candidate:** How many videos are available on the platform?
**Interviewer:** We have about 10 billion videos.

**Candidate:** How fast should the system recommend videos to a user? Can I assume the recommendation should not take more than 200 milliseconds?
**Interviewer:** That sounds good.

## Frame the Problem as an ML Task

### Defining the ML objective

**Maximize the number of relevant videos.** This objective produces recommendations that are relevant to users. Engineers or product managers can define relevance based on some rules. Such rules can be based on implicit and explicit user reactions. For example, one definition could state a video is relevant if a user explicitly presses the "like" button or watches at least half of it. Once we define relevance, we can construct a dataset and train a model to predict the relevance score between a user and a video.

### Specifying input and output

- Input: user/user profile
- Output: Ranked list of videos

### Chossing the right ML category

#### Content-based filtering

1. User A engaged with videos X and Y in the past
2. Video Z is similar to video X and video Y
3. The system recommends video Z to user A

Content-based filtering has pros and cons.

**Pros:**

- **Ability to recommend new videos.** With this method, we don't need to wait for interaction data from users to build video profiles for new videos. The video profile depends entirely upon its features.
- **Ability to capture the unique interests of users.** This is because we recommend videos based on users' previous engagements.

**Cons:**

- **Difficult to discover a user's new interests.**
- The method requires **domain knowledge**. We often need to engineer video features manually.

#### Collaborative filtering (CF)

The goal is to recommend a new video to user A.

1. Find a similar user to A based on their previous interactions; say user B
2. Find a video that user B engaged with but which user A has not seen yet; say video Z
3. Recommend video Z to user A

**Pros:**

- **No domain knowledge needed.** CF does not rely on video features, which means no domain knowledge is needed to engineer features from videos.
- **Easy to discover users' new areas of interest.** The system can recommend videos about new topics that other similar users engaged with in the past.
- **Efficient.** Models based on CF are usually faster and less compute-intensive than content-based filtering, as they do not rely on video features.

**Cons:**

- **Cold-start problem.** This refers to a situation when limited data is available for a new video or user, meaning the system cannot make accurate recommendations. CF suffers from a cold-start problem due to the lack of historical interaction data for new users or videos. This lack of interactions prevents CF from finding similar users or videos. We will discuss later in the serving section how our system handles the cold-start problem.
- **Cannot handle niche interests.** It's difficult for CF to handle users with specialized or niche interests. CF relies upon similar users to make recommendations, and it might be difficult to find similar users with niche interests.

#### Hybrid Filtering (Sequential)

Input -> CF Based filtering -> Content based filtering -> Output

## Data Preparation

### Data engineering

VIdeos

| **Video ID** | **Length** | **Manual tags** | **Manual title**            | **Likes** | **Views** | **Language** |
| :----------- | :--------- | :-------------- | :-------------------------- | :-------- | :-------- | :----------- |
| 1            | 28         | Dog, Family     | Our lovely dog playing!     | 138       | 5300      | English      |
| 2            | 300        | Car, Oil        | How to change your car oil? | 5         | 250       | Spanish      |
| 3            | 3600       | Ouli, Vlog      | Ooneymoon to Bali           | 2200      | 255K      | Arabic       |

User

| **ID** | **Username** | **Age** | **Gender** | **City** | **Country** | **Language** |
| :----- | :----------- | :------ | :--------- | :------- | :---------- | :----------- |
|        |              |         |            |          |             |              |

User-video interactions

| **User ID** | **Video ID** | **Interaction type** | **Interaction value** | **Location (lat, long)** | **Timestamp** |
| :---------- | :----------- | :------------------- | :-------------------- | :----------------------- | :------------ |
| 4           | 18           | Like                 | -                     | 38.8951 -77.0364         | 1658451361    |
| 2           | 18           | Impression           | 8 seconds             | 38.8951 -77.0364         | 1658451841    |
| 2           | 6            | Watch                | 46 minutes            | 41.9241 -89.0389         | 1658822820    |
| 6           | 9            | Click                | -                     | 22.7531 47.9642          | 1658832118    |
| 9           | -            | Search               | Basics of clustering  | 22.7531 47.9642          | 1659259402    |
| 8           | 6            | Comment              | Amazing video. Thanks | 37.5189 122.6405         | 1659244197    |

### Feature engineering

Discrete tags/numeric features: keep the number

Text features: embedding

Liked videos/Impressions/Watched videos: vision emebdding

## Model Development

We examine two embedding-based models that are typically employed in CF-based or content-based recommenders:

- Matrix factorization
- Two-tower neural network

### Matrix factorization

##### Feedback matrix

Also called a utility matrix, this is a matrix that represents users' opinions about videos. Figure 6.11 shows a binary user-video feedback matrix where each row represents a user, and each column represents a video.

##### Matrix factorization model

Matrix factorization is a simple embedding model. The algorithm decomposes the user-video feedback matrix into the product of two lower-dimensional matrices. One lower-dimensional matrix represents user embeddings, and the other represents video embeddings. In other words, the model learns to map each user into an embedding vector and each video into an embedding vector, such that their distance represents their relevance.

![](../../../img/ML/ml design/5-1.png)

##### Matrix factorization training

As part of training, we aim to produce user and video embedding matrices so that their product is a good approximation of the feedback matrix.

To learn these embeddings, matrix factorization first randomly initializes two embedding matrices, then iteratively optimizes the embeddings to decrease the loss between the "Predicted scores matrix" and the "Feedback matrix". 

**A weighted combination of squared distance over observed and unobserved pairs**

The first summation in the loss formula calculates the loss on the observed pairs, and the second summation calculates the loss on unobserved pairs. W*W* is a hyperparameter that weighs the two summations. It ensures one does not dominate the other in the training phase. This loss function with a properly tuned W works well in practice. We choose this loss function for the system.

![](../../../img/ML/ml design/5-2.png)

Before wrapping up matrix factorization, let’s discuss the pros and cons of this model.

**Pros:**

- Training speed: Matrix factorization is efficient during the training phase. This is because there are only two embedding matrices to learn.
- Serving speed: Matrix factorization is fast at serving time. The learned embeddings are static, meaning that once we learn them, we can reuse them without having to transform the input at query time.

**Cons:**

- Matrix factorization only relies on user-video interactions. It does not use other features, such as the user's age or language. This limits the predictive capability of the model because features like language are useful to improve the quality of recommendations.
- Handling new users is difficult. For new users, there are not enough interactions for the model to produce meaningful embeddings. Therefore, matrix factorization cannot determine whether a video is relevant to a user by computing the dot product between their embeddings.

### Two-tower neural network

A two-tower neural network comprises two encoder towers: the user tower and the video tower. The user encoder takes user features as input and maps them to an embedding vector (user embedding). The video encoder takes video features as input and maps them into an embedding vector (video embedding). The distance between their embeddings in the shared embedding space represents their relevance.

Figure 6.19 shows the two-tower architecture. In contrast to matrix factorization, twotower architectures are flexible enough to incorporate all kinds of features to better capture the user's specific interests.

![](../../../img/ML/ml design/5-3.png)

##### Choosing the loss function

Since the two-tower neural network is trained to predict binary labels, the problem can be categorized as a classification task. We use a typical classification loss function, such as cross-entropy, to optimize the encoders during training. 

Let’s see the pros and cons of a two-tower neural network model.

**Pros:**

- **Utilizes user features.** The model accepts user features, such as age and gender, as input. These predictive features help the model make better recommendations.
- **Handles new users.** The model easily handles new users as it relies on user features (e.g., age, gender, etc.).

**Cons:**

- **Slower serving.** The model needs to compute the user embedding at query time. This makes the model slower to serve requests. In addition, if we use the model for content-based filtering, the model needs to transform video features into video embedding, which increases the inference time.
- **Training is more expensive.** Two-tower neural networks have more learning parameters than matrix factorization. Therefore, the training is more computeintensive.

## Evaluation

The system’s performance can be evaluated with offline and online metrics.

### Offline metrics

We evaluate the following offline metrics commonly used in recommendation systems.

**Precision@k.** This metric measures the proportion of relevant videos among the top kk recommended videos. Multiple kk values (e.g., 1,5,101,5,10 ) can be used.

**mAP.** This metric measures the ranking quality of recommended videos. It is a good fit because the relevance scores are binary in our system.

**Diversity.** This metric measures how dissimilar recommended videos are to each other. This metric is important to track, as users are more interested in diversified videos. To measure diversity, we calculate the average pairwise similarity (e.g., cosine similarity or dot product) between videos in the list. A low average pairwise similarity score indicates the list is diverse.

Note that using diversity as the sole measure of quality can result in misleading interpretations. For example, if the recommended videos are diverse but irrelevant to the user, they may not find the recommendations helpful. Therefore, we should use diversity with other offline metrics to ensure both relevance and diversity.

### Online metrics

In practice, companies track many metrics during online evaluation. Let's examine some of the most important ones:

- Click-through rate (CTR)
- The number of completed videos
- Total watch time
- Explicit user feedback

## Serving

![](../../../img/ML/ml design/5-4.png)

#### Candidate generation

The goal of candidate generation is to narrow down the videos from potentially billions, to thousands. We prioritize efficiency over accuracy at this stage and are not concerned about false positives.

To keep candidate generation fast, we choose a model which doesn't rely on video features. In addition, this model should be able to handle new users. A two-tower neural network is a good fit for this stage. Once the computation is complete, it retrieves the most similar videos from the approximate nearest neighbor service. These videos are ranked based on similarity in the embedding space and are returned as the output.

Users may be interested in videos for many reasons. For example, a user may choose to watch a video because it's popular, trending, or relevant to their location. To include those videos in the recommendations, it is common to use more than one candidate generation.

#### Scoring

Also known as ranking, scoring takes the user and candidate videos as input, scores each video, and outputs a ranked list of videos.

At this stage, we prioritize accuracy over efficiency. To do so, we choose content-based filtering filtering and pick a model which relies on video features. A two-tower neural network is a common choice for this stage. Since there are only a handful of videos to rank in the scoring stage, we can employ a heavier model with more parameters. 

#### Re-ranking

This component re-ranks the videos by adding additional criteria or constraints. For example, we may use standalone ML models to determine if a video is clickbait. Here are a few important things to consider when building the re-ranking component:

- Region-restricted videos
- Video freshness
- Videos spreading misinformation
- Duplicate or near-duplicate videos
- Fairness and bias

## Other Talking Points

If there is time left at the end of the interview, here are some additional talking points:

- The exploration-exploitation trade-off in recommendation systems [9].
- Different types of biases may be present in recommendation systems [10].
- Important considerations related to ethics when building recommendation systems [11].
- Consider the effect of seasonality - changes in users' behaviors during different seasons - in a recommendation system [12].
- Optimize the system for multiple objectives, instead of a single objective [13].
- How to benefit from negative feedback such as dislikes [14].
- Leverage the sequence of videos in a user's search history or watch history [2].

# 07 Event Recommendation System

## Clarification

**Candidate:** What is the business objective? Can I assume the main business objective is to increase ticket sales?
**Interviewer:** Yes, that sounds good.

**Candidate:** Besides attending an event, can users book hotels or restaurants on the platform?

**Interviewer**: For simplicity, let's assume only events are supported.

**Candidate:** An event is considered an ephemeral one-time occurrence item that only happens once, and then expires. Is this assumption correct?
**Interviewer:** That's an excellent observation.

**Candidate:** What event attributes are available? Can I assume we have access to the textual description of the event, price range, location, date and time, etc.?
**Interviewer:** Sure, those are fair assumptions.

**Candidate:** Do we have any annotated data?
**Interviewer:** We don't have a hand-labeled dataset. You can use event and user interaction data to construct the training dataset.

**Candidate:** Do we have access to the user's current location?
**Interviewer:** Yes. Since this problem focuses on a location-based recommendation system, let's assume users agree to share their location data.

**Candidate:** Can users become friends on the platform? Friendship information is valuable for building a personalized event recommendation system.
**Interviewer:** Good question. Yes, let's assume users can form friendships on our platform. A friendship is bidirectional, meaning if A is a friend of B, then B is also a friend of A.

**Candidate:** Can users invite others to events?
**Interviewer:** Yes.

**Candidate:** Can a user RSVP to an event?
**Interviewer:** For simplicity, let's assume only a registration option is available for an event.

**Candidate:** Are the events free or paid?
**Interviewer:** We need to support both.

**Candidate:** How many users and events are available?
**Interviewer:** We host around 1 million total events every month.

**Candidate:** How many daily active users visit the website/app?
**Interviewer:** Assume we have one million unique users per day.

**Candidate:** Since we are building a location-based event recommendation system, it's important to calculate the distance and travel time between two locations efficiently. Can we assume external APIs such as Google Maps API or other map services can be used to obtain such data?
**Interviewer:** Good point. Assume we can use third-party services to obtain location data.

## Frame the Problem as an ML task

### Defining the ML objective

The business objective is to increase ticket sales. One way to translate this into a well-defined ML objective is to maximize the number of event registrations.

### Specifying input and output

Input: user

Output: topk ranked events by relevance to the user

### Choosing the right ML category

We reformulate the task into a ranking problem and use Learning to Rank (LTR) to solve it.

LTR is a class of algorithmic techniques that apply supervised machine learning to solve ranking problems. The ranking problem can be formally defined as: "having a query and a list of items, what is the optimal ordering of the items from most relevant to least relevant to the query?" There are generally three LTR approaches: pointwise, pairwise, and listwise. 

- Pointwise LTR: Item, Query -> [Pointwise ranking model] -> Relevance Score
- Pairwise LTR: Item a, Item b, Query -> [Pointwise ranking model] -> Item a > Item b
- Listwise LTR

For simplicity, we use the pointwise approach for this problem. In particular, we employ a binary classification model which takes a single event at a time and predicts the probability that the user will register for it. 

## Data Preparation

### Data engineering

Since an event management platform is mainly centered around users and events, we assume the following data are available:

- Users

  The user data schema is shown below.

  | ID   | Username | Age  | Gender | City | Country | Language | Time zone |
  | :--- | :------- | :--- | :----- | :--- | :------ | :------- | :-------- |
  |      |          |      |        |      |         |          |           |

- Events

  Table 7.2 shows what the event data might look like.

  | ID   | Host User ID | Category/ Subcategory | Description                               | Price    | Location                          | Date/Time                |
  | :--- | :----------- | :-------------------- | :---------------------------------------- | :------- | :-------------------------------- | :----------------------- |
  | 1    | 5            | Music Concert         | Dua Lipa Tour in Miami                    | 200-900  | American Airlines Arena Miami, FL | 09//18//2022 19:00-24:00 |
  | 2    | 11           | Sports Basketball     | Golden State Warriors vs. Milwaukee Bucks | 140-2500 | Chase Center SF, CA               | 09//22//2022 17:00-19:00 |
  | 3    | 7            | Art Theater           | The Comedy and Magic of Robert Hall       | Free     | San Jose Improv San Jose, CA      | 09//06//2022 18:00-19:30 |

- Friendship

  Each row represents a friendship formed between two users, along with the timestamp of when it was formed

  | User ID 1 | User ID 2 | Timestamp when friendship was formed |
  | :-------- | :-------- | :----------------------------------- |
  | 28        | 3         | 1658451341                           |
  | 7         | 39        | 1659281720                           |
  | 11        | 25        | 1659312942                           |

- Interactions

User interaction data, such as event registrations, invitations, and impressions.

| User ID | Event ID | Interaction type | Interaction value   | Location (lat, long) | Timestamp  |
| :------ | :------- | :--------------- | :------------------ | :------------------- | :--------- |
| 4       | 18       | Impression       | -                   | 38.8951 -77.0364     | 1658450539 |
| 4       | 18       | Register         | Confirmation number | 38.8951 -77.0364     | 1658451341 |
| 4       | 18       | Invite           | User 9              | 41.9241 -89.0389     | 1658451365 |

### Feature engineering

Event-based recommendations are more challenging than traditional recommendations. An event is fundamentally different from a movie or a book, as there is no consumption after the event ends. Events are typically short-lived, meaning the time is short between event creation and when it finishes. As a result, there are not many historical interactions available for a given event. For this reason, event-based recommendations are intrinsically cold-start and suffer from a constant new-item problem. To overcome those issues, we put more effort into feature engineering to create as many meaningful features as possible. 

#### Location-related features

**How accessible is the event's location?**

The accessibility of an event's location is an important factor. For example, if an event is high up in hills far from public transportation, the commute may discourage users from attending. Let's create the following features to capture accessibility:

- Walk score: Walk score is a number between 0 and 100, which measures how walkable an address is, based on the distance to nearby amenities. It is computed by analyzing various factors such as distance to amenities, pedestrian friendliness, population density, etc. We assume walk scores can be obtained from external data sources such as Google Maps, Open Street Map, etc. Table 7.57.5 shows walk scores bucketized into 5 categories.

| Category | Walk score | Description       |
| :------- | :--------- | :---------------- |
| 1        | 90-100     | No car needed     |
| 2        | 70-89      | Very walkable     |
| 3        | 50-69      | Somewhat walkable |
| 4        | 25-49      | Car-dependent     |
| 5        | 0-24       | Requires a car    |

Table 7.5: Walk score categories

- Walk score similarity: The difference between the event's walk score and the user's average walk score of previous events registered by the user.
- Transit score, transit score similarity, bike score, bike score similarity.

**Is the event in the same country and city as the user?**
A very important deciding factor for a user is whether the event is in the same country and city where they are located. The following two features can be created:

- If the user's country is the same as the event's country, this feature is 1, otherwise 0
- If the user's city is the same as the event's city, this feature is 1, otherwise 0

**Is the user comfortable with the distance?**
Some users may prefer events that are very close to their location, while others prefer events that are further away. We use the following features to capture this:

- The distance between the user's location and the event's location. This value can be obtained from external APIs and bucketized into a few categories. For example:
  - 0: less than a mile
  - 1: 1-5 miles
  - 2: 5-20 miles
  - 3: 20-50 miles
  - 4: 50-100 miles
  - 5: +100 miles
- Distance similarity: Difference between the distance to an event and the average distance (in reality, the median or percentile range can be used) to events previously registered by the user.

#### Time-related features

**How convenient is the time remaining until an event?**
Some users may plan events a few days in advance, while others don't. Let's create the following features to capture this:

- The remaining time until the event begins. This feature can be bucketized into different categories and one-hot encoded. For example:
  - 0: less than 1 hour left until the event starts
  - 1: 1-2 hours
  - 2: 2-4 hours
  - 3: 4-6 hours
  - 4: 6-12 hours
  - 5: 12-24 hours
  - 6: 1-3 days
  - 7: 3-7 days
  - 8: +7 days
- Remaining time similarity: Difference between "remaining time" and average "remaining time" of events previously registered by the user.
- The estimated travel time from the user's location to the event's location. This value will be obtained from external services and bucketized into categories.
- Estimated travel time similarity: The difference between the estimated travel time to the event in question, and the average estimated travel time of events previously registered by the user.

**Are the date and time convenient for the user?**
Some users may prefer events that occur at weekends, while others prefer weekdays. Some users prefer events in the morning, while others may prefer evening events. To capture a user's historical preferences for days of the week, we create a user profile. This user profile is a vector of size 7 , and each value counts the number of events the user attended on a particular day. By dividing these values by the total number of attended events, we get the historical rate of event attendance for each day of the week.

![7-1](../../../img/ML/ml design/7-1.png)

#### Social-related features

**How many people are attending this event?**
In general, users are more likely to register for an event if there are a lot of other attendees. Let's extract the following features to capture this:

- Number of users registered for this event
- The ratio of the total number of registered users to the number of impressions
- Registered user similarity: The difference between the number of registered users for the event in question and previously registered events

**Features related to attendance by friends**
A user is more likely to register for an event if their friends are attending it. Here are some of the features we can use:

- Number of the user's friends who registered for this event
- The ratio of the number of registered friends to the total number of friends
- Registered friend similarity: Difference between the number of registered friends for the event in question and previously registered events

**Is the user invited to this event by others?**
Users are more likely to attend events to which they are invited. Some features that might be helpful are:

- The number of friends who invited this user to the event
- The number of fellow users who invited this person to the event

**Is the event's host a friend of the user?**
Users tend to attend events created by their friends. We create a binary feature to reflect this: if the event's host is the user's friend, this value is 1, otherwise, 0.

**How often has the user attended previous events created by this host?**
Some users are interested in following a particular host's events.

#### User-related features

**Age and gender**
Some events are geared toward specific ages and genders. For example, "Women in Tech" and "Life lessons to excel in your 30 s" are examples of events that may be specific to certain demographic groups. We create two features to capture this:

- User's gender, encoded with one-hot encoding
- User's age, bucketized into multiple categories and encoded with one-hot encoding

#### Event-related features

**Price of event:**
The price of an event might affect the user's decision to register for it. Some features to use are:

- Event's price, bucketized into a few categories. For example:
  - 0: Free
  - 1: $1-$99
  - 2: $100-$499
  - 3: $500-$1,999
  - 4:+$2,000
- Price similarity: Difference between the price of the event in question and the average price of events previously registered for by the user.

**How similar is this event's description to previously registered descriptions?**
This indicates the user's interests, based on previously registered events. For example, if the word "concert" repeatedly appears in the descriptions of previous events, it may indicate the user is interested in concert events. To capture this, we create a feature that represents the similarity between the event's description and the descriptions of previously registered events by the user. To compute the similarity, the description is converted into a numerical vector using TF-ID, and similarity is calculated using cosine distance.

Note, this feature might be noisy as descriptions are manually provided by hosts. We can experiment by training our model with and without this feature, to measure its importance.

#### Other points

- **Batch vs. streaming features:** Batch (static) features refer to features that change less frequently, such as age, gender, and event description. These features can be computed periodically using batch processing and stored in a feature store. In contrast, streaming (aka dynamic) features change quickly. For example, the number of users registered for an event and the remaining time until an event, are dynamic features. The interviewer may want you to dive deeper into this topic and discuss batch vs. online processing in ML. If you're interested to learn more, refer to [8].
- **Feature computation efficiency.** Computing features in real-time is not efficient. You may want to discuss this issue and possible ways to avoid it. For example, instead of computing the distance between the user's current location and the event's location as a feature, we can pass both locations to the model as two separate features, and rely on the model to implicitly compute useful information from the two locations. To learn more about how to prepare location data for ML models, refer to [9]

## Model Development

### Model selection

**Binary classification problem**

- Logistic regression

  - **Pros:**
    - **Fast inference speed.** Computing a weighted combination of input features is fast.
    - **Efficient training.** Given the simple architecture, it's easy to implement, interpret, and train quickly.
    - Works well when the data is linearly separable (Figure 7.12).
    - **Interpretable and easy to understand.** The weights assigned to each feature indicate the importance of different features, which gives us insight into why a decision was made.
  - **Non-linear problems can't be solved** with LR, since it uses a linear combination of input features.
  - **Multicollinearity** occurs when two or more features are highly correlated. One of the known limitations of LR is that it cannot learn the task well when multicollinearity is present in the input features.

- Decision tree

  - **Pros:**

    - **Fast training:** Decision trees are quick to train.
    - **Fast inference:** Decision trees make predictions quickly at inference time.
    - **Little to no data preparation:** Decision tree models don't require data normalization or scaling, since the algorithm does not depend on the distribution of the input features.
    - **Interpretable and easy to understand.** Visualizing the tree provides good insights into why a decision was made and what the important decision factors are.

  - **Cons:**

    - **Non-optimal decision boundary:** decision tree models produce decision boundaries that are parallel to the axes in the feature space (Figure 7.13). This may not be the optimal way to find a decision boundary for certain data distributions.

    - **Overfitting:** Decision trees are very sensitive to small variations in data. A small change in input data may lead to different outcomes at serving time. Similarly, a small change in training data can lead to a totally different tree structure. This is a major issue and makes predictions less reliable.

  - In practice, naive decision trees are rarely used. The reason is that they are too sensitive to variations of input data. To reduce the sensitivity of decision trees, two techniques are commonly used. These two techniques are widely used across the tech industry. It's essential to understand how they work. Let's take a closer look.

    - Bootstrap aggregation (Bagging)

      - Bagging is the ensemble learning method that trains a set of ML models in parallel, on multiple subsets of the training data. In bagging, the predictions of all these trained models are combined to make a final prediction. This significantly reduces the model's sensitivity to the change in data (variance).

        One example of bagging is the commonly used "random forest" model [12]. Random forest builds multiple decision trees in parallel during training, to reduce the model's sensitivity. To make a prediction, each decision tree independently predicts the output class (positive or negative) of the given input, and then a voting mechanism is used to combine these predictions to make a final prediction.

      - The bagging technique has the following advantages:

        - Reduces the effect of overfitting (high variance).
        - Does not significantly increase training time because the decision trees can be trained in parallel.
        - Does not add much latency at the inference time because decision trees can process the input in parallel.

      - Despite its advantages, bagging is not helpful when the model faces underfitting (high bias). To overcome bagging’s drawbacks, let’s discuss another technique called boosting.

    - Boosting

      - In ML, boosting involves training several weak classifiers sequentially to reduce prediction errors. The phrase "weak classifier" refers to a simple classifier that performs slightly better than random guesses. In boosting, multiple weak classifiers are converted into a single strong learning model. 
      - **Pros:**
        - **Boosting reduces bias and variance.** Combining weak classifiers leads to a strong model less sensitive to the change in data. To learn more about bias/variance tradeoffs, refer to [13]. Cons:
        - **Slower training and inference.** Given the classifiers are trained based on the mistakes of the previous classifiers, they work sequentially. This adds to the serving time due to the sequential nature of boosting.
      - 

- Gradient-boosted decision tree (GBDT)

  - GBDT is a commonly used tree-based model, utilizing GradientBoost to improve decision trees. Some variants of GBDT, such as XGBoost [15], have demonstrated strong performance in various ML competitions
  - **Pros:**
    - **Easy data preparation:** Similar to decision trees, it does not require data preparation.
    - **Reduces variance:** GBDT reduces variance as it uses the boosting technique.
    - **Reduces bias:** GBDT reduces the prediction error by leveraging several weak classifiers, iteratively improving upon the misclassified data points from the previous classifiers.
    - Works well with structured data.

  - **Cons:**

    - **Lots of hyperparameters to tune**, such as the number of iterations, tree depth, regularization parameters, etc.

    - GBDT does not work well on unstructured data such as images, videos, audio, etc.

    - **Unsuitable for continual learning** from streaming data.

- Neural network

  - In an event recommendation system, we have many features that might not correlate linearly with the outcome. Learning these complex relationships is difficult. In addition, continual learning is necessary for adapting the model to new data.

    NNs are great at solving those challenges. They are capable of learning complex tasks with non-linear decision boundaries. Additionally, NN models can be fine-tuned on new data very easily, making them ideal for continual learning.

  - **Pros**

    - **Continual learning:** NNs are designed to learn from data and improve themselves continually.
    - Works well with unstructured data such as text, image, video, or audio.
    - **Expressiveness:** NNs have expressive power due to their high number of learning parameters. They can learn very complex tasks and non-linear decision boundaries.

  - **Cons**

    - **Computationally expensive** to train.

    - **The quality of input data strongly influences the outcome:** NNs are sensitive to input data. For example, if input features are in very different ranges, the model may converge slowly during the training phase. An important step for NNs is data preparation, such as normalization, log-scaling, one-hot encoding, etc.

    - **Large training data** is required to train NNs.

    - **Black-box nature:** NNs are not interpretable, meaning it's not easy to understand the influence of each feature upon the outcome, as the input features go through multiple layers of non-linear transformations.

**Which model should we select?**

We can choose the right model based on various factors:

- Complexity of the task
- Data distribution and data type
- Product requirements or constraints, such as training cost, speed, model size, etc In this problem, both GBDTs and NNs are good candidates for experimentation. We start with the GBDT variant, XGBoost, since it is fast to implement and train. The result can be used as an initial baseline.

Once we have a baseline, we explore the possibility of building a better model with NNs. Neural networks are expected to work well here for the following reasons:

- Massive training data is available in our system. Users continuously interact with the system by registering for events, inviting friends, publishing new events, etc. Given the number of users, this creates a massive amount of data available for training.
- Data may not be linearly separable, and neural networks can learn non-linear data.

When designing a NN architecture, several hyperparameters must be considered, including the number of hidden layers, neurons in each layer, activation function, etc. These can be determined by employing hyperparameter tuning techniques.

### Model training

#### Constructing dataset

To construct a single data point, we extract a ⟨⟨ user, event ⟩⟩ pair from the interaction data and compute the input features from the pair. We then label the data point with 1 if the user has registered for the event, and 0 if not.

One issue we may face after constructing the dataset is class imbalance. The reason is that users may explore tens or hundreds of events before registering for one. Therefore, the number of negative ⟨⟨ user, event ⟩⟩ pairs is significantly higher than positive data points. We can use one of the following techniques to address the class imbalance issue:

- Use focal loss or class-balanced loss to train the classifier
- Undersample the majority class

#### Choosing the loss function

- Cross Entropy

## Evaluation

### Offline metrics

**nDCG, or mAP**: nDCG works well when the relevance score between a user and an item is non-binary. In contrast, mAP works only when the relevance scores are binary. Since events are either relevant (a user registered for it) or irrelevant (a user saw the event but did not register), mAP is a better fit.

### Online metrics

- Click-through rate (CTR): A ratio showing how often users who see recommended events go on to click on an event.
- Conversion rate: A ratio showing how often users who see recommended events go on to register for them.
- Bookmark rate
- Revenue lift

## Deployment

![7-2](../../../img/ML/ml design/7-2.png)

##### Event filtering

The event filtering component takes the query user as input and narrows down the events from 1 million to a small subset of events. This is based upon simple rules, such as event locations, or other types of user filters. For example, if a user adds a “concerts only” filter, the component quickly narrows down the list to a subset of candidate events. Since these types of filters are common in event recommendation systems, they can be used to significantly reduce our search space from potentially millions of events, to hundreds of candidate events.

##### Ranking service

This service takes the user and candidate events produced by the filtering component as input, computes features for each ⟨⟨ user, event ⟩⟩ pair, sorts the events based on the probabilities predicted by the model, and outputs a ranked list of top k*k* most relevant events to the user.

Ranking service interacts with the feature computation component responsible for computing features that the model expects. Static features are obtained from a feature store, while dynamic features are computed in real-time from the raw data.

## Other talking points

- What are the different types of bias we may observe in this system [21].
- How to utilize feature crossing to achieve more expressiveness [22].
- Some users like to see a diverse list of events. How to ensure the recommended events are diverse and fresh [23]?
- We utilize the user's attributes to train a model. We also rely on users' live locations. What are additional considerations related to privacy and security [24]?
- Event management platforms are usually two-sided marketplaces, where event hosts are the suppliers and users fulfill the demand side. How to ensure the system is not optimized for one side only? Additionally, how to keep the platform fair for different hosts? To learn more about unique challenges in two-sided marketplaces, refer to [25].
- How to avoid data leakage when constructing the dataset [26].
- How to determine the right frequency to update the models [27].

# 8 Ad Click prediction on social platforms

## Clarification

Here is a typical interaction between a candidate and an interviewer.

**Candidate**: Can I assume the business objective of building an ad prediction system is to maximize revenue?
**Interviewer:** Yes, that’s correct.

**Candidate:** There are different types of ads, such as video and image ads. In addition, ads can be displayed in different sizes and formats, like users’ timelines, pop-up ads, etc. For simplicity, can I assume ads are placed on users’ timelines only, and every click generates the same revenue?
**Interviewer:** That sounds good.

**Candidate:** Can the system show the same ad to the same user more than once?
**Interviewer:** Yes, we can show an ad more than once. Sometimes, an ad turns into a click after multiple impressions. In reality, companies have a “fatigue period”, that is, they don’t show the same ad to the same user for X days if the user repeatedly ignores it. For simplicity, assume we have no fatigue period.

**Candidate:** Do we support the “hide this ad” feature? How about “block this advertiser”? These kinds of negative feedback help us to detect irrelevant ads.
**Interviewer:** Good question. Let’s assume users can hide an ad they don’t like. “Block this advertiser” is an interesting feature, but we don’t need to support it for now.

**Candidate:** Would it be okay to assume that the training dataset should be constructed using user and ad data, and the labels should be based on user-ad interactions?
**Interviewer:** Sure.

**Candidate:** We can construct positive training data points via user clicks, but how do we generate negative data points? Can we assume any impression that is not clicked is a negative data point? What if the user scrolls fast and doesn’t spend time seeing the ad? What if we count an impression as negative, but eventually, the user clicks on it?
**Interviewer:** These are excellent questions. What are your thoughts?

**Candidate:** If an ad is visible on a user’s screen for a certain duration but not clicked, we can count it as a negative data point. An alternative approach would be to assume impressions are negative until a click is observed. In addition, we can rely on negative feedback such as “hide this ad” to label negative data points.
**Interviewer:** Makes sense! In practice, we might use other complex techniques to label negative data points . For this interview, let’s proceed with your suggestions.

**Candidate:** In ad click prediction systems, it’s critical for the model to learn from new interactions continuously. Is it fair to assume continual learning is a necessity here?
**Interviewer:** Great point. Experiments have shown that even a 5-minute delay in updating models can damage performance [1].

## Frame the Problem as an ML task

### Defining ML objective

Goal: increase revenue by showing users ads they are more likely to click on. This can be converted into the following ML objective: predicting if an ad will be clicked. This is due to the fact that by correctly predicting click probabilities, the system can display relevant ads to users, which leads to an increase in revenue.

### Specifying input and output

- Input: user and ads
- Output: a ranked list of ads based on click probabilities.

### Choosing the right ML category

Employs a binary classification model.

## Data Preparation

### Data engineering

#### Ads

| Ad ID | Advertiser ID | Ad group ID | Campaign ID | Category  | Subcategory |       Images or Videos        |
| :---: | :-----------: | :---------: | :---------: | :-------: | :---------: | :---------------------------: |
|   1   |       1       |      4      |      7      |  travel   |    hotel    | http: //cdn.mysite.com/u1.jpg |
|   2   |       7       |      2      |      9      | insurance |     car     | http: //cdn.mysite.com/t3.mp4 |

#### Users

|  ID  | Username | Age  | Gender | City | Country | Language | Time zone |
| :--: | :------: | :--: | :----: | :--: | :-----: | :------: | :-------: |
|      |          |      |        |      |         |          |           |

#### User-Ad interation

| User ID | Ad ID | Interaction type | Dwell time | Location (lat, long) | Timestamp  |
| :-----: | :---: | :--------------: | :--------: | :------------------: | :--------: |
|   11    |   6   |    Impression    |    5sec    |   38.8951 -77.0364   | 165845053  |
|   11    |   7   |      Click       |     -      |   41.9241 -89.0389   | 1658451365 |

### Feature engineering

Our aim in this section is to engineer features that will assist us in predicting user clicks.

#### Ad features

- IDs
  - These are advertiser ID, campaign ID, ad group ID, ad ID,etc.
  - **Why is it important?** The IDs represent the advertiser, the campaign, the ad group, and the ad itself. These IDs are used as predictive features to capture the unique characteristics of different advertisers, campaigns, ad groups, and ads.
  - **How to prepare it?** The embedding layer converts sparse features, such as IDs, into dense feature vectors. Each ID type has its own embedding layer.

- Image/Video
  - **Why is it important?** A video or image in a post is another signal that can help us predict what the ad is about. For example, an image of an airplane may indicate the ad is related to travel.
  - **How to prepare it?** The images or videos are first preprocessed. After that, we use a pre-trained model such as SimCLR [3] to convert unstructured data into a feature vector.

- Ad category and subcategory
  - **Why is it important?** It helps the model to understand which category the ad belongs to.
  - **How to prepare it?** These are manually provided by the advertiser based on a predefined list of categories and subcategories. To learn more about preparing textual data, read Chapter 4, YouTube Video Search.

- Impressions and click numbers
  - Total impression/clicks on the ad
  - Total impressions/clicks on ads supplied by an advertiser
  - Total impressions of the campaign

#### User-Ad interaction features

- Clicked ads
  - Ads previously clicked by the user.
  - **Why is it important?** Previous clicks indicate a user's interests. For example, when a user clicks on lots of insurance-related ads, it suggests they are likely to click on a similar ad again.
  - **How to prepare it?** In the same way as described in "Ad features".

- User’s historical engagement statistics

  - These are the user’s historical engagement numbers, such as their total ad views and ad click rate.
  - **Why is it important?** An individual's historical engagement is a good predictor of future engagement. In general, users are more likely to click on ads in the future, if they clicked on ads frequently in the past.

  - **How to prepare it?** Engagement statistics are represented as numerical values. To prepare them, we scale their values into a similar range.

A challenge: many features, such as categories and advertiser ID, are high cardinality features - many features will be null

## Model development

### Model selection

- Logistic regression
- Feature crossing + logistic regression
- Gradient boosted decision trees
- Gradient boosted decision trees + logistic regression
- Neural networks
- Deep & Cross networks

#### Logistic regression

- Pros: easy to train, fast inference
- Cons: cannot solve non-linear problem; cannot capture feature dependence

#### Feature crossing + logistic regression

Feature crossing is a technique used in ML to create new features from existing features. It involves combining two or more existing features into one new feature by taking their product, sum, or another combination. It is possible to capture nonlinear interactions between the original features in this way, which can improve the performance of ML models.

- Manually add new features to the existing features based on prior knowledge.
- Use the original and the crossed features as input for the LR model.
- Cons: manual process, domain knowledge, not work for higher order interactions.

#### GBDT

- Pros: interpretable
- Cons: not suited for continual learning; too many features to consider

#### Neual Network

- Single NN: freatures -> NN -> probability
- Two-tower: User -> NN -> User embedding; Ad -> NN -> Ad embedding; User embedding * Ad embedding -> probability

- Cons: not enough data with features to train; difficult to capture all feature interactions

#### Deep & Cross Network

In 2017, Google proposed an architecture named DCN [6] to find feature interactions automatically. This addresses the challenges of the manual feature crossing method. The following two parallel networks are used in this method:

- **Deep network:** Learns complex and generalizable features using a Deep Neural Network (DNN) architecture.
- **Cross network:** Automatically captures feature interactions and learns good feature crosses.

The outputs of deep network and cross network are concatenated to make a final prediction.

### Model training

#### Constructing dataset

For every ad impression, we construct a new data point. The input features are computed from the user and the ad. A label is assigned to the data point, based on the following strategy:

- **Positive label:** if the user clicks the ad in less than t seconds after the ad is shown, we label the data point as "positive". Note that t*t* is a hyperparameter and can be tuned via experimentation.
- **Negative label:** if the user does not click the ad in less than t*t* seconds, we label the data point as "negative".

#### Choosing the loss function

Since we are training a binary classification model, we choose cross-entropy as a classification loss function.

## Evaluation

### Offline metrics

- Cross Entropy
  $$
  H(p, q)=-\sum^C_{c=1}p_c\log q_c
  $$

  - Better system -> closer to 0

### Online metrics

- CTR
- Hide rate
- Revenue lift

## Serving

![](../../../img/ML/ml design/8-1.png)

The prediction pipeline takes a query user as input and outputs a list of ads ranked by their click probabilities. Since some of the features which the model relies upon are dynamic, we cannot use batch prediction. Instead, requests are served as they arrive using online prediction.

As we've seen in previous chapters, a two-stage architecture is used in the prediction pipeline. First, we employ a candidate generation service to efficiently narrow down the available pool of ads to a small subset of ads. In this case, we use the ad targeting criteria often provided by advertisers, such as target age, gender, and country.

Next, we employ a ranking model which fetches the candidate ads from the candidate generation service, ranks them based on click probability, and outputs the top ads. This component interacts with the same feature store and online feature computation component. Once the static and dynamic features are obtained, the ranking service uses the model to get a predicted click probability for each candidate ad. These probabilities are used to rank the ads and to output those with the highest click probability.

Finally, a re-ranking service modifies the list of ads by incorporating additional logic and heuristics. For example, we can increase the diversity of ads by removing very similar ads from the list.

## Other Talking Points

If there is time left at the end of the interview, here are some potential talking points you might discuss with the interviewer:

- In ranking and recommendation systems, it's important to avoid data leakage [12][13][12][13]
- The model needs to be calibrated in ad click prediction systems. Discuss model calibration and techniques for calibrating a model [14].
- We've described why continuous learning is necessary for ad click prediction systems. However, continual learning on new data may lead to catastrophic forgetting. Discuss what catastrophic forgetting is and what common solutions are [16].

# 10 Personalized News Feed

## Clarification

Here is a typical interaction between a candidate and an interviewer.

**Candidate:** Can I assume the motivation for a personalized news feed is to keep users engaged with the platform?
**Interviewer:** Yes, we display sponsored ads between posts, and more engagement leads to increased revenue.

**Candidate:** When a user refreshes their timeline, we display posts with new activities to the user. Can I assume this activity consists of both unseen posts and posts with unseen comments?
**Interviewer:** That is a fair assumption.

**Candidate:** Can a post contain textual content, images, video, or any combination?
**Interviewer:** It can be any combination.

**Candidate:** To keep users engaged, the system should place the most engaging content at the top of timelines, as people are more likely to interact with the first few posts. Does that sound right?
**Interviewer:** Yes, that's correct.

**Candidate:** Is there a specific type of engagement we are optimizing for? I assume there are different types of engagement, such as clicks, likes, and shares.
**Interviewer:** Great question. Different reactions have different values on our platform. For example, liking a post is more valuable than only clicking it. Ideally, our system should consider major reactions when ranking posts. With that, I'll leave you to define "engagement" and choose what your model should optimize for.

**Candidate:** What are the major reactions available on the platform? I assume users can click, like, share, comment, hide, block another user, and send connection requests. Are there other reactions we should consider?
**Interviewer:** You mentioned the major ones. Let's keep our focus on those.

**Candidate:** How fast is the system supposed to work?
**Interviewer:** We expect the system to display the ranked posts quickly after users refresh their timelines or open the application. If it takes too long, users will get bored and leave. Let's assume the system should display the ranked posts in less than 200 milliseconds (ms).

**Candidate:** How many daily active users do we have? How many timeline updates do we expect each day?
**Interviewer:** We have almost 3 billion users in total. Around 2 billion are daily active users who check their feeds twice a day.

## Frame the problem as an ML task

### Defining the ML objective

We use both implicit (click, time spent) and explicit reactions (like, share, comment, dislike) to determine how engaged a user is with a post. In particular, we assign a weight to each reaction, based on how valuable the reaction is to us. We then optimize the ML system to maximize the weighted score of reactions.

Table 10.1 shows the mapping between different reactions and weights. As you can see, pressing the "like" button has more weight than a click, while a share is more valuable than a like. In addition, negative reactions such as hide and block have a negative weight. Note that these weights can be chosen based on business needs.

### Input and output

- Input: user
- Output: ranked unseen posts

### Chossing the right ML category

Ranking -> Binary classifiers -> One classifier for each reactions

Once these probabilities are predicted, we compute the engagement score. 

## Data Preparation

### Data Engineering

#### Users

The user data schema is shown below.

|  ID  | Username | Age  | Gender | City | Country | Language | Time zone |
| :--: | :------: | :--: | :----: | :--: | :-----: | :------: | :-------: |
|      |          |      |        |      |         |          |           |

#### Post

| Author ID |              Textual Content               |      Hashtags       |      Mentions      |       Images or videos        | Timestamp  |
| :-------: | :----------------------------------------: | :-----------------: | :----------------: | :---------------------------: | :--------: |
|     5     | Today at our fav place with my best friend | life_is_good, happy |       hs2008       |               -               | 1658450539 |
|     1     |      It was the best trip we ever had      |  Travel, Maldives   | Alexish, shan.tony | htcdn.mysite.com/maldives.jpg | 1658451341 |

#### User-post interactions

Table 10.4 shows user-post interaction data.

| User ID | Post ID | Interaction type | Interaction value | Location (lat, long) | Timestamp  |
| :-----: | :-----: | :--------------: | :---------------: | -------------------: | :--------: |
|    4    |   18    |       Like       |         -         |     38.8951 -77.0364 | 1658450539 |
|    4    |   18    |      Share       |      User 9       |     41.9241 -89.0389 | 1658451365 |
|    9    |   18    |     Comment      | You look amazing  |      22.7531 47.9642 | 1658435948 |
|    9    |   18    |      Block       |         -         |      22.7531 47.9642 | 1658451849 |
|    6    |    9    |    Impression    |                   |     37.5189 122.6405 | 1658821820 |

##### Friendship

The friendship table stores data of connections between users. We assume users can specify their close friends and family members. Table 10.5 shows examples of friendship data.

| User ID 1 | User ID 2 | Time when friendship was formed | Close friend | Family member |
| :-------: | :-------: | :-----------------------------: | :----------: | :-----------: |
|    28     |     3     |           1558451341            |     True     |     False     |
|     7     |    39     |           1559281720            |    False     |     True      |
|    11     |    25     |           1559312942            |    False     |     False     |

### Feature engineering

 In particular, we engineer features from each of the following categories:

- Post features
- User-post interaction features
- User-author affinities

#### Post features

- Textual content and hashtag
- Images or video
- Reactions: number of likes, shares, ... - Need to scale
- Post's age - quantize into buckets

#### User-post interaction features

- User-post historical interactions: All posts liked by a user are represented by a list of post IDs. The same logic applies to shares and comments.
  - **Why is it important?** Users' previous engagements are usually helpful in determining their future engagements.
  - **How to prepare it?** Extract features from each post that the user interacted with.

- Being mentioned in a post: This means whether or not the user is mentioned in a post. Why is it important? Users usually pay more attention to posts that mention them.
  - This feature is represented by a binary value. If a user is mentioned in the post, this feature is 1 , otherwise 0 .

#### User-author affinities

- Like/click/comment/share rate
- Friendship degree

## Model development

### Model selection

Neural network - work well with unstructured data, compatible with embeddings

Two options

- N indepdent NNs
  - 1 for each reaction
  - Cons: compute-intensive, data sparity for less frequent reactions
- 1 multitask NN

#### Improving for passive users

many users use the platform passively, meaning they do not interact much with the content on their timelines. For such users, the current DNN model will predict very low probabilities for all reactions, since they rarely react to posts. 

Solution: Add two implicit reactions

- Time spent: regression task
- Skip rate: classification task

### Model training

#### Constructing dataset

the number of negative data points is usually much higher than positive data points. To avoid having an imbalanced dataset, we create negative data points to equal the number of positive data points. 

#### Chooing the loss function

- Classification: CE
- Regression: MSE

## Evaluation

### Offline

- Evaluate a user-ad pair prediction - precision, recall
- Evaluate user-ranked ads list - mAP, nDCG

### Online

- CTR
- Reaction rate
- DAU
- Time spent

## Serving

Similar to Ad clieck prediction

## Other Talking Points

If there is time left at the end of the interview, here are some additional talking points:

- How to handle posts that are going viral [15].
- How to personalize the news feed for new users [16].
- How to mitigate the positional bias present in the system [17].
- How to determine a proper retraining frequency [18].

# 11 People you may know

## Clarification

**Candidate:** Can I assume the motivation for building the PYMK feature is to help users discover potential connections and grow their network?
**Interviewer:** Yes, that’s a good assumption.

**Candidate:** To recommend potential connections, a huge list of factors must be considered, such as location, educational background, work experience, existing connections, previous activities, etc. Should I focus on the most important factors, such as educational background, work experience, and the user’s social context?
**Interviewer:** That sounds good.

**Candidate:** On LinkedIn, two people are friends if – and only if – each is a friend of the other. Is that correct?
**Interviewer:** Yes, friendship is symmetrical. When someone sends a connection request to another user, the recipient needs to accept the request for the connection to be made.

**Candidate:** What’s the total number of users on the platform? How many of them are daily active users?
**Interviewer:** We have nearly 1 billion users and 300 million daily active users.

**Candidate:** How many connections does an average user have?
**Interviewer:** 1,000 connections.

**Candidate:** The social graph of most users is not very dynamic, meaning their connections don’t change significantly over a short period. Can I make this assumption when designing PYMK?
**Interviewer:** That’s an excellent point. Yes, it’s a reasonable assumption.

## Frame the problem as an ML task

### Defining the ML objective

A common ML objective in PYMK systems is to maximize the number of formed connections between users. This helps users to grow their networks quickly.

### Specifying the system’s input and output

The input to the PYMK system is a user, and the outputs are a list of connections ranked by relevance to the user. 

### Chossing the right ML category

#### Pointwise LTR

We employ a binary classification model which takes two users as input and outputs the probability of the given pair forming a connection. However, this approach has a major drawback; since the model's inputs are two distinct users, it doesn't consider the available social context. While this does simplify things, leaving out information about a user's connections might make predictions less accurate.

##### Edge prediction

In this approach, we supplement the model with graph information. This enables the model to rely on the additional knowledge extracted from the social graph, to predict whether an edge exists between two nodes.

More formally, we use a model that takes the entire social graph as input, and predicts the probability of an edge existing between two specific nodes. To rank potential connections for user A, we compute the edge probabilities between user A and other users, and use these probabilities as the ranking criteria.

In addition to the typical features that the model utilizes, the model also relies on additional knowledge extracted from the social graph to predict whether an edge exists between two nodes.

![11-1](../../../img/ML/ml design/11-1.png)

## Data Preparation

### Data engineering

In this section, we discuss the raw data available:

- Users

| **User ID** | **School** | **Degree** |       **Major**        | **Start date** | **End date** |
| :---------: | :--------: | :--------: | :--------------------: | :------------: | :----------: |
|     11      |  Waterloo  |    M.Sc    |    Computer Science    |  August 2015   |   May 2017   |
|     11      |  Harvard   |    M.Sc    |        Physics         |    May 2004    | August 2006  |
|     11      |    UCLA    | Bachelors  | Electrical Engineering |    Sep 2022    |      -       |

- Connections

A simplified example of connection data is shown in Table 11.2. Each row represents a connection between two users and when the connection was formed.

| User ID 1 | User ID 2 | Timestamp when the connection was formed |
| :-------: | :-------: | :--------------------------------------: |
|    28     |     3     |                1658451341                |
|     7     |    39     |                1659281720                |
|    11     |    25     |                1659312942                |

- Interactions

| User ID |  Interaction type   |      Interaction value       | Timestamp  |
| :-----: | :-----------------: | :--------------------------: | :--------: |
|   11    | Connection request  |          user_id_8           | 1658450539 |
|    8    | Accepted connection |          user_id_11          | 1658451341 |
|   11    |       Comment       | [user_id_4, Very insightful] | 1658451365 |
|    4    |       Search        |        "Scott Belsky"        | 1658435948 |
|   11    |    Profile view     |          user_id_21          | 1658451849 |

### Feature engineering

###### **Demographics: age, gender, city, country, etc.**

Demographic data helps determine if two users are likely to form a connection. Users tend to connect with others who have similar demographics.

It's common to have missing values in demographic data. To learn more about how to handle missing values, refer to the "Introduction and Overview" chapter.

###### **The numbers of connections, followers, following, and pending requests**

This information is important as users are more likely to connect with someone with lots of followers or connections, compared to a user with few connections.

###### **Account’s age**

Accounts created very recently are less reliable than those that have existed for longer. For example, if an account was created yesterday, it's more likely to be a spam account. So, it may not be a good idea to recommend it to users.

###### **The number of received reactions**

These are numerical values representing the total number of reactions received, such as likes, shares, and comments over a certain period, like one week. Users tend to connect with more active users on the platform, who receive more interactions from other users.

##### User-user affinities

The affinity between two users is a good signal to predict if they will connect. Let’s look at some important features which capture user-user affinities.

###### **Education and work affinity**

- **Schools in common:** Users tend to connect with others who attended the same school.
- **Contemporaries at school:** Overlapping years at school increases the likelihood of two users connecting. For example, users might want to connect with someone who attended school XX the same time they did.
- **Same major:** A binary feature representing whether two users had the same major in school.
- **Number of companies in common:** Users may connect with people who have worked at the same companies.
- **Same industry:** A binary feature representing whether the two users work in the same industry.

###### Social affinity

- **Profile visits:** The number of times a user looks at the profile of another user.
- **Number of connections in common, aka mutual connections:** If two users have many common connections, they are more likely to connect. This feature is one of the most important predictive features [2].
- **Time discounted mutual connections:** This feature weighs mutual connections by how long they have existed. Let's go through an example to understand the reasoning behind this feature.

## Model Development

### Model selection

GNN/GCN

### Model Training

To train a GNN model, we provide the model with a snapshot of the social graph at time t. The model predicts the connections which will form at time t+1. Let's examine how to construct the training data.

#### Constructing dataset

To construct the dataset, we do the following:

1. Create a snapshot of the graph at time t

   The first step in constructing training data is to create input for the model. Since a GNN model expects a social graph as input, we create a snapshot of the social graph at time t*t* using the available raw data. Figure 11.1111.11 shows an example of the graph at time t*t*.

2. Compute initial node features and edge features of the graph

   We extract the user's features, such as age, gender, account age, number of connections, etc. These are used as the nodes' initial feature vectors. Similarly, we extract user-user affinity features and employ them as the initial feature vectors of the edges. 

3. Create labels

   In this step, we create labels that the model is expected to predict. We use the graph snapshot at time t+1*t*+1 to determine positive or negative labels. Let's take a look at a concrete example.

   ![11-2](../../../img/ML/ml design/11-2.png)

As shown in Figure 11.14, positive and negative labels are created depending on whether a new edge forms at t+1*t*+1. In particular, we label a pair of nodes as positive when they connect at t+1*t*+1. Otherwise, they are labeled as negative.

![11-3](../../../img/ML/ml design/11-3.png)

## Evaluation

##### GNN model

Since the GNN model predicts the presence of edges, we can think of it as a binary classification model. ROC-AUC metric is used to measure the performance of the model.

##### PYMK system

We extensively discuss choosing the right offline metrics for ranking and recommendation systems in previous chapters, so don't go into detail here. In our system, a user will either connect with a recommended connection or discard it. Due to this binary nature (connect or not), mAPmAP is a good choice.

#### Online metrics

In practice, companies track lots of online metrics to measure the impact of PYMK systems. Let's explore two of the most important metrics:

- The total number of connection requests sent in the last X days
- The total number of connection requests accepted in the last X days

**The total number of connection requests sent in the last X days.** This metric helps us understand if the model increases or decreases the number of connection requests. For example, if a model leads to a 5%5% increase in the total number of sent connection requests, we can assume the model has a positive impact on the business objective.

However, this metric has a major drawback. A new connection forms between two users only when the recipient accepts a request to connect. For example, a user may send 1,000 connection requests, but recipients accept only a small percentage. This metric might not correctly reflect the actual growth of the users' network. Now, let's address this drawback with the next metric.

**The total number of connection requests accepted in the last X days.** As a new connection forms only when the recipient accepts the sender's request, this metric accurately reflects the real growth of the users' network.

## Serving

![11-4](../../../img/ML/ml design/11-4.png)

## Other Talking Points

If there's time left at the end of the interview, here are some additional talking points:

- Personalized random walk [8] is another method often used to make recommendations. Since it's efficient, it is a helpful way to establish a baseline.
- Bias issue. Frequent users tend to have greater representation in the training data than occasional users. The model can become biased towards some groups and against others due to uneven representation in the training data. For example, in the PYMK list, frequent users might be recommended to other users at a higher rate. Subsequently, these users can make even more connections, making them even more represented in the training data [9].
- When a user ignores recommended connections repeatedly, the question arises of how to take them into account in future re-ranks. Ideally, ignored recommendations should have a lower ranking [9].
- A user may not send a connection request immediately when we recommend it to them. It may take a few days or weeks. So, when should we label a recommended connection as negative? In general, how would we deal with delayed feedback in recommendation systems [10]?
