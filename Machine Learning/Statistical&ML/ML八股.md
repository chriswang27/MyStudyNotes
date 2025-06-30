# ML八股

## Resources

ML Interview Book

- https://huyenchip.com/ml-interviews-book/

ML八股

- https://bertmclee.medium.com/2024-%E7%BE%8E%E5%9C%8B%E5%9C%B0%E7%8D%84%E6%A8%A1%E5%BC%8F%E4%B8%8A%E5%B2%B8ds-mle%E7%B6%93%E9%A9%97%E5%88%86%E4%BA%AB-%E8%82%86-%E5%A6%82%E4%BD%95%E6%BA%96%E5%82%99machine-learning-statistics-interview-b6e4226363b8
- https://www.1point3acres.com/bbs/thread-713903-1-1.html

## Knowledge

### ML Algorithm

#### Variance-Bias

复杂模型有更多参数和更强的拟合能力，能够捕捉数据中的细微模式 -> 更低的bias

但也更容易"记住"训练数据中的噪声(Overfit) -> Poor generalization -> 高variance -> Bad on test set

### Regularization

- L1

  - Loss=Original Loss+λi∑∣wi∣
  - Promotes **sparsity** by driving some weights exactly to zero.
  - Use L1 if
    - You expect **many irrelevant features**.
    - You want to perform **automatic feature selection**.
    - You need a **sparse model** for interpretability.

- L2

  - Loss=Original Loss+λi∑wi^2

  - Penalizes large weights smoothly; encourages **small, but nonzero weights**.

  - Use L2 if 

    - All features are expected to be relevant.

      You want to **distribute the weight** more evenly.

      You care more about **numerical stability and generalization** than sparsity.

- Work with over-parameterization
  - A model is **over-parameterized** when it has **more parameters than training data** (e.g., deep neural networks with millions of weights and only thousands of examples). This means the model can **perfectly fit (memorize)** the training data — even noise or random labels.
  - Without constraints, it leads to: Overfitting, Poor generalization

### ML Workflow

#### Basics

##### ML -> DL

- Availability of Big Data
- Advancements in Hardware
- Algorithmic Improvements
  - **Better training techniques** like ReLU activations, batch normalization, dropout, and residual connections made it feasible to train deep networks.
  - **Optimization advances** such as Adam and RMSprop improved convergence speed and stability.
  - **Architectural innovations**:
    - CNNs for image data (e.g., AlexNet, ResNet)
    - RNNs, LSTMs, and later Transformers for sequential and textual data

- Infra support: Pytorch, Tenserflow

##### Deploying Large ML Models

- Latency and Inference Speed
  - Model compression (quantization, pruning, distillation)
  - Hardware acceleration (GPUs, TPUs)
  - Use smaller models or knowledge distillation for deployment

- Resource Consumption
  - Use model parallelism or sharding
  - Optimize model architecture

- Debugging and Observability - More unpredictable behaviors
  - More testings
  - Monitoring

##### Why model performs different between test and production

Your model performs really well on the test set but poorly in production. What are your hypotheses about the causes? How do you validate whether your hypotheses are correct? Imagine your hypotheses about the causes are correct. What would you do to address them?

| Hypotheses                        | How to Validate                                              | How to address                                               |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Data Distribution Shift           | Compare distribution of key features between train/test and prod | Log and replay production inputs through offline model to compare outputs |
| Training/Test Data Contamination  | Revisited training/test data                                 | Use stricter train/test split rules                          |
| Different Preprocessing Pipelines | Unit test both pipelines; compare outputs given same raw input | Consolidate preprocessing                                    |
| Evaluation Metrics Misalignment   | Correlate with offline metrics                               | Redefine evaluation metric to better reflect production objectives |
| Infrastructure or Serving Issues  | Log and replay production inputs through offline model to compare outputs | Add unit tests, version control for models, shadow deployment before full launch |

#### Sampling

##### **Markov Chain Monte Carlo (MCMC)** 

MCMS is a family of algorithms used to draw **samples from complex probability distributions**, especially when direct sampling is hard.

How MCMC Works

- **Construct a Markov chain** whose stationary distribution is the desired target distribution p(x)p(x)p(x).

- **Run the chain for a while** (called “burn-in”) so it converges.

- **Collect samples** from the chain, treating them as approximate samples from p(x)p(x)p(x).

Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Target: Mixture of two Gaussians (unnormalized)
def target_distribution(x):
    component1 = 0.3 * np.exp(-0.5 * ((x + 2) / 0.5)**2)
    component2 = 0.7 * np.exp(-0.5 * ((x - 2) / 1.0)**2)
    return component1 + component2

# Proposal: Gaussian centered at current x
def proposal(x, step_size=1.0):
    return np.random.normal(x, step_size)

def metropolis_hastings(num_samples=10000, burn_in=1000, step_size=1.0):
    samples = []
    x = 0.0  # initial value

    for _ in range(num_samples + burn_in):
        x_prop = proposal(x, step_size)  # <--- Propose new state (Markov chain)

        # Evaluate target density (unnormalized)
        p_current = target_distribution(x)
        p_proposal = target_distribution(x_prop)

        # Compute acceptance probability
        acceptance = min(1, p_proposal / p_current)

        # Accept/reject step (Markov chain transition)
        if np.random.rand() < acceptance:
            x = x_prop

        samples.append(x)  # <--- Collect sample (Monte Carlo)

    return np.array(samples[burn_in:])
```

##### Candidate Sampling

Candidate sampling approximates the full softmax by:

- Only computing scores for the **true class** and a small number of **negative (candidate) classes**.
- Making training scalable by **avoiding full normalization**.

Common Candidate Sampling Algorithms:

- Noise Contrastive Estimation (NCE)

  - **Idea**: Reframe classification as a **binary classification problem**:
    - Distinguish between true data and samples from a **noise distribution**.

  - Instead of modeling the softmax over all classes, learn to **differentiate real vs. fake (noise)**.

  - Learns the unnormalized logits and normalizes them implicitly.

- Sampled Softmax

  - **Direct approximation of the softmax loss**.

  - Instead of summing over all classes in the denominator, sample a subset.

  - Typically includes the true class and a few sampled negative classes.

##### How to make sure Train-Test distribution are same

- Train a binary classifier to distinguish between train and test samples.

- If the classifier achieves **> 50% accuracy**, the two sets are likely **from different distributions**.

##### How do you know you’ve collected enough samples to train your ML model?

- Monitor Model Performance Stability. You’ve collected enough data when:
  - Performance **plateaus** or **saturates** — adding more data doesn’t help much.
  - Variance (error bars) across runs stabilizes.
- **High variance / overfitting?** → More data can help reduce overfitting.
- **High bias / underfitting?** → More data might not help — you may need a better model or features.
- If model performance is **highly sensitive** to data split or random initialization, you don’t have enough data (or it's too noisy).

##### Sampling Duplication

- When should you remove duplicate training samples?

  - **Duplicates arise from data pipeline or collection bugs**

  - **They overweight certain patterns artificially**
    - Can bias model toward frequent duplicated examples (especially in imbalanced datasets).

  - **You're training a model sensitive to instance frequency**
    - E.g., logistic regression, SVMs, or any model where gradient descent is affected by repeated examples.

  - **You care about generalization**
    - Duplicates reduce training set diversity → poorer generalization to new data.

- Don’t remove duplicates when:

  - **Duplicates reflect real-world frequency or importance**
    - E.g., user logs or purchase history: if 10 people bought the same item, you want the model to see it 10 times. Especially in NLP or image classification, where repeated examples may appear naturally.

  - **Your model is frequency-aware**
    - Like Naive Bayes or language models trained on token frequency (e.g., Word2Vec).

- Duplication to test set:  Overestimates model robustness and generalization.

##### Hanling Missing Features

- Identify the reason for missing
- Then can
  - Drop
  - Replace it with mean
  - Add a "missing" flag colum
- When handling missing data techniques worsen selection bias? If **missingness is not random**

##### Training Data Leakage

- Check whether there is temporal causality

 ##### Case Study

1. Suppose you want to build a model to classify whether a Reddit comment violates the website’s rule. You have 10 million unlabeled comments from 10K users over the last 24 months and you want to label 100K of them.

   1. [M] How would you sample 100K comments to label?

      - Stratify by Key Axes

        - **User activity**: heavy vs. light posters

        - **Time**: comments over the 24 months (to avoid recency bias)

        - **Subreddit or topic category** (if known)

        - **Text length or metadata**: long vs. short comments

        - **Comment position**: top-level vs. reply

      - Add Uncertainty-Based or Diversity Sampling

      - Apply De-duplication / Relevance Filtering

   2. [M] Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?

      - How Many to Review? Typically: **0.5%–1%** of the total labeled set.

      - Sampling Strategy

        - Random Sample
        - Stratified by Annotator
        - Label-Based Stratification

        - Model Disagreement or Uncertainty-Based Sample*

#### Metrics & Eval

##### Overfitting

- Sympotoms

  - Very low training error

  - High validation/test error

- How to mitigate?

  - Simplify the model (e.g., shallower tree, fewer parameters)
  - Add regularization (L2, dropout, etc.)
  - Get more training data
  - Use cross-validation

##### F1 Score

- Why F1 is better than Accuracy?

  - Accuracy = (Correct Predictions) / (Total Predictions). A model predicting only the **majority class** can still achieve **high accuracy**, while completely **failing to detect the minority class**.

  - F1 is useful when **false positives and false negatives both matter**, and when **classes are imbalanced**.

- F1 for multi-class classification

  - Macro F1 - treat each classes equally
  - Micro F1 - treat each samples equally -> **Favors frequent classes**, good for imbalance.

##### Cross Entropy

- Cross-Entropy is Designed for Probabilities

  - In classification, you want to **predict a probability distribution** over classes.

  - Cross-entropy directly compares the predicted probability distribution to the true one-hot label
    $$
    CE=-\sum_{i=1}^Ky_i\log (\hat y_i)
    $$

  - It's minimized when the predicted probability for the true class is **maximized**.

- CE and negative log-likelihood are equivalent for binary classification. 
  $$
  L_{NLL}=-\log P(y|\hat y)=-[y\log y+(1-y)\log(1-\hat y)]
  $$
