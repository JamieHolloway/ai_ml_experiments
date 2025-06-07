# Logistic Regression 

## Introduction

Logistic regression is a fundamental algorithm for binary classification problems. Unlike linear regression (which predicts any continuous value), logistic regression outputs a probability between 0 and 1 for each class. By applying a **sigmoid function** to a linear combination of input features, logistic regression constrains predictions to the range \[0, 1], which can then be interpreted as probabilities. To make a final classification, we choose a threshold (often 0.5): if the predicted probability is above the threshold, we classify as one class (say “1”), otherwise as the other class (“0”). This threshold at 0.5 corresponds to the point where the sigmoid output is 0.5 (and the linear input to the sigmoid is 0). In this tutorial, we will build a logistic regression model from scratch using only **NumPy**, and in the process explain key concepts such as the sigmoid function, loss function, gradient descent, and decision boundary. We will also visualize the model's behavior and draw connections to a simple neural network.

## The Sigmoid (Logistic) Function

In logistic regression, after computing a linear combination of the input features with weights, we squash that value through the **sigmoid function** (also known as the logistic function) to obtain a number between 0 and 1. The sigmoid is defined as:

$\sigma(z) = \frac{1}{1 + e^{-z}}$

where $z$ is the linear combination (sometimes called the logit). This function has an S-shaped curve as shown below, approaching 1 for large positive $z$ and 0 for large negative $z$.

&#x20;*Figure: The sigmoid function $\sigma(z) = \frac{1}{1+e^{-z}}$. It outputs values between 0 and 1. Notably, $\sigma(0) = 0.5$, which is often used as the classification threshold (dashed line).*

The sigmoid function ensures our model's output can be interpreted as a probability. For example, $\sigma(z)=0.8$ means the model predicts an 80% probability for the positive class. A key property is that $\sigma(0) = 0.5$. Typically, we use 0.5 as the cutoff: outputs >= 0.5 are labeled as class 1 and outputs < 0.5 as class 0. This cutoff defines the **decision boundary** in the input space where the model is indifferent between classes.

## Loss Function for Logistic Regression

To train the model (i.e. find the best weights), we need to quantify how well the model’s predictions match the true labels. This is done with a **loss function** (also called a cost function). In linear regression, we often use Mean Squared Error, but that doesn't work well for logistic regression because it leads to a non-convex optimization problem. Instead, logistic regression uses the **log loss** (also known as binary cross-entropy) as the cost function, which is convex and suitable for optimization.

The **log loss** for a single training example is:

$\ell(\hat{y}, y) = -\Big[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\Big],$

where $y \in \{0,1\}$ is the true label and $\hat{y} \in [0,1]$ is the predicted probability for the label being 1. This loss is **zero** if the prediction is perfect (e.g. $\hat{y}=1$ for a true label 1 gives $-\log(1)=0$), and grows larger as the prediction deviates from the truth (for instance, predicting a probability close to 0 for a true label 1 incurs a large penalty due to the $-\log(\hat{y})$ term). The overall cost function $J(w,b)$ for the dataset is the average loss over all training examples. Using log loss (cross-entropy) ensures the cost function is convex for logistic regression, so gradient-based optimization can find a global minimum.

## Gradient Descent

**Gradient descent** is an iterative optimization algorithm used to minimize the cost function. The idea is to start with some initial weights and **iteratively update** them in the direction that reduces the cost. The “gradient” of the cost function with respect to the parameters indicates the direction of steepest increase; so we subtract the gradient (scaled by a learning rate) to move the parameters in the direction of steepest decrease. Repeating this process leads the model to a set of parameters that (hopefully) minimize the cost.

In logistic regression, the parameters are the weight vector $w$ and bias $b$. The gradient of the cost with respect to $w$ can be derived from the loss function. Without going into full calculus detail, the result is:

$\nabla_w J(w,b) = \frac{1}{m} X^T (\hat{y} - y),$

where $X$ is the matrix of features for all $m$ training examples, $y$ is the vector of true labels, and $\hat{y} = \sigma(Xw + b)$ is the vector of predictions. Similarly, the gradient with respect to the bias is $\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$. These formulas come from differentiating the log-loss; they essentially average the prediction error ($\hat{y}-y$) weighted by the input values. Using these gradients, the **gradient descent update rule** for each iteration is:

$w := w - \alpha \, \nabla_w J, \quad b := b - \alpha \, \frac{\partial J}{\partial b},$

where $\alpha$ is the learning rate (a small positive number). We repeat this update until the cost function converges (changes very little) or until we’ve done a predetermined number of iterations.

**Summary of the training procedure (Gradient Descent Algorithm):**

1. **Initialize** weights $w$ and bias $b$ (e.g. to zeros).
2. **Loop** over training iterations:
   a. Compute the linear combination $z^{(i)} = w^T x^{(i)} + b$ for each example.
   b. Apply the sigmoid to get predictions $\hat{y}^{(i)} = \sigma(z^{(i)})$.
   c. Compute the cost (average log-loss over examples).
   d. Compute gradients $\nabla_w$ and $\partial J/\partial b$.
   e. Update parameters: $w := w - \alpha \nabla_w, \; b := b - \alpha \frac{\partial J}{\partial b}$.
3. **Repeat** steps 2a–2e until convergence (or for a fixed number of iterations).

Gradient descent will adjust the model parameters to reduce the cost. Over many iterations, the algorithm should find weights that make the predictions as close as possible to the true labels.

## Creating a Synthetic Dataset

To demonstrate logistic regression, let's create a simple synthetic dataset. We will generate two sets of 2D points for two classes (class 0 and class 1) that are roughly linearly separable. This allows us to visualize them and see how a linear decision boundary can separate the classes.

For example, we can sample 50 points around $(2,2)$ for class 0 and 50 points around $(4.5,4.5)$ for class 1, with a bit of random noise. The scatter plot below shows our synthetic dataset, with each class in a different color.

&#x20;*Figure: A synthetic binary classification dataset. We have two features (plotted on x and y axes) and two classes: class 0 (cyan points) and class 1 (magenta points). Our goal is for logistic regression to learn a line that separates these two classes.*

Looking at the data, you can imagine a diagonal line roughly dividing the cyan and magenta points. Points on one side of this line would belong to class 0 and on the other side to class 1. Logistic regression will try to learn exactly such a **decision boundary**. In a 2D feature space, the decision boundary is a line that separates the regions where the model predicts class 0 versus class 1.

## Implementation: Logistic Regression with NumPy

Now, let's walk through a step-by-step implementation of logistic regression using NumPy. We will implement the model training from scratch, following the gradient descent approach outlined above. No machine learning libraries (scikit-learn, TensorFlow, etc.) will be used—just basic NumPy for calculations.

### 1. Sigmoid function in code

First, we define the sigmoid function in Python. This function will take a scalar or NumPy array $z$ and return $\sigma(z) = 1/(1+e^{-z})$. We will use this to convert linear outputs into probabilities.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

This uses NumPy's ability to handle array computations, so it works for both individual values and arrays of values.

### 2. Initializing parameters

We need to initialize the model parameters. For simplicity, we can start with all weights = 0 and bias = 0. (In practice, small random values can also be used, but zero initialization is fine for this example.)

```python
# Assume X_train is our feature matrix of shape (m, n)
m, n = X_train.shape  # m = number of examples, n = number of features
w = np.zeros(n)       # initialize weight vector of length n
b = 0.0               # initialize bias as 0.0
```

At this point, $w = 0, b = 0$. Next, we'll perform the iterative training.

### 3. Training loop with gradient descent

We'll run a loop for a fixed number of iterations (epochs) to update $w$ and $b$. In each iteration, we will:

* Compute $z = X \cdot w + b$ for all training examples (this is a vectorized form of computing $w^T x^{(i)} + b$ for each example).
* Apply sigmoid to get predictions $\hat{y}$.
* Compute the cost (average log loss).
* Compute gradients for $w$ and $b$.
* Update $w$ and $b$ using the gradients and the learning rate.

Let's choose a learning rate (for example, 0.1) and a number of iterations (for example, 1000) for training. Below is the code for the training loop, with comments explaining each part:

```python
alpha = 0.1        # learning rate
num_iter = 1000    # number of iterations for gradient descent

for it in range(num_iter):
    # Forward pass: compute linear combination and prediction
    z = np.dot(X_train, w) + b             # shape (m,), linear combination for each example
    preds = sigmoid(z)                     # shape (m,), predicted probabilities for each example
    
    # Compute the cost (log loss)
    # Add a small epsilon inside log to avoid log(0)
    cost = -np.mean(y_train * np.log(preds + 1e-9) + (1 - y_train) * np.log(1 - preds + 1e-9))
    
    # Compute gradients
    error = preds - y_train                # shape (m,), difference between prediction and true label
    grad_w = (1/m) * np.dot(X_train.T, error)   # shape (n,), gradient for weights
    grad_b = (1/m) * np.sum(error)              # scalar, gradient for bias
    
    # Update parameters
    w -= alpha * grad_w
    b -= alpha * grad_b
    
    # (Optional) you could print or store cost here to monitor convergence
```

During training, the cost should gradually decrease, indicating that the model is fitting the data better over time. The gradients `grad_w` and `grad_b` are computed as described earlier. We subtract `alpha * grad` from the parameters to move in the direction of decreasing cost.

After running this loop, we obtain a trained weight vector `w` and bias `b`. These define our model: given a new input $x$, we compute $\hat{y} = \sigma(w^T x + b)$. We can use $\hat{y} \ge 0.5$ to predict class 1 or else class 0.

### 4. Model results and decision boundary

Now that we have trained the model, let's see what it learned. The weights $w$ and bias $b$ define a **decision boundary** in our feature space. For two features, the decision boundary is the line where $ w_1 x_1 + w_2 x_2 + b = 0$. On this line, $z = 0$ and thus $\sigma(z)=0.5$. Points on one side of the line will have $z>0$ (and thus $\hat{y}>0.5$, class 1), and points on the other side will have $z<0$ ($\hat{y}<0.5$, class 0).

The plot below shows the training data again, with the learned decision boundary superimposed:

&#x20;*Figure: The learned decision boundary (yellow line) from our logistic regression model on the synthetic dataset. Points on one side of the line are classified as class 0 and on the other side as class 1. The model has adjusted $w_1, w_2, b$ so that $w_1 x_1 + w_2 x_2 + b = 0$ (sigmoid output 0.5) along this line. This line indeed separates most cyan and magenta points correctly, illustrating the concept of a decision boundary.*

As we can see, the logistic regression has found a linear boundary that largely distinguishes the two classes. The exact equation for the line can be written from our learned weights. For example, if $w = [0.98, 0.93]$ and $b = -6.03$ (these are the values from our training run), the decision boundary equation is $0.98 \cdot x_1 + 0.93 \cdot x_2 - 6.03 = 0$. Any point satisfying this equation yields a predicted probability of 0.5. If we plug in a point like $(x_1=4.5, x_2=4.5)$ for class 1, we get $0.98(4.5) + 0.93(4.5) - 6.03 \approx 2.58$, which is > 0 (above the line, classified as class 1). For a point around $(2,2)$ for class 0, $0.98(2) + 0.93(2) - 6.03 \approx -2.2$, which is < 0 (below the line, class 0). This matches our expectations.

## Logistic Regression as a Single-Layer Neural Network

It’s insightful to realize that logistic regression is essentially a very simple **neural network** – one with no hidden layers (just an input layer directly connected to an output neuron). The diagram below illustrates this: we have input features $x_1, x_2, \dots, x_n$ feeding into a single output unit. Each input is multiplied by a weight, all summed together along with a bias term, and then a sigmoid activation is applied.

&#x20;*Figure: Logistic regression can be viewed as a one-layer neural network. Each input feature $x_j$ is multiplied by a weight $w_j$, the weighted inputs are summed (plus a bias term $b$), and then a sigmoid activation produces the output $\hat{y}$. If $\hat{y} \ge 0.5$, the output is classified as 1; otherwise 0. This is equivalent to a simple neural network with one output neuron using a sigmoid activation.*

In the figure, the bias input is shown as a constant 1 feeding into the neuron with weight $b$. Logistic regression has no hidden units; it's just a direct mapping from inputs to output through a non-linear activation (sigmoid). This perspective helps connect logistic regression to broader machine learning: it is basically a one-neuron neural network. In fact, many neural network libraries implement logistic regression as a special case of a one-layer network. The thresholding (at 0.5, or another value depending on context) at the output can be seen as part of the decision process to yield a final class prediction.

## Execution Flow of the Logistic Regression Algorithm

To recap the entire process, let's visualize the execution flow of training logistic regression using gradient descent:

&#x20;*Figure: Flowchart of logistic regression training. Step 1: Initialize weights and bias. Step 2: Compute the weighted sum $z = w^T x + b$. Step 3: Apply sigmoid to get predicted probability $\hat{y}$. Step 4: Compute the loss (log loss) comparing $\hat{y}$ to true $y$. Step 5: Compute gradients of the loss w\.r.t. parameters. Step 6: Update $w, b$ using the gradients (gradient descent step). Steps 2–6 are repeated until convergence. This forms the computation graph of logistic regression, from inputs through the sigmoid to the loss, and backpropagating gradients for learning.*

Each iteration of training goes through these computations. Initially, the model might predict poorly (high loss), but as steps 2–6 repeat, the weights are adjusted and the loss typically decreases. The **computation graph** above shows how data and gradients flow: inputs go through a linear combination then sigmoid to produce an output, which is compared to the true label to compute loss; then gradients flow backward to update parameters.

## Conclusion

In this tutorial, we built a logistic regression classifier from scratch using NumPy and a synthetic dataset. We covered the key concepts: the sigmoid function that squashes outputs to \[0,1], the log-loss cost function that guides learning, and gradient descent which optimizes the parameters. We also visualized how the model finds a linear decision boundary to separate classes, and saw that logistic regression is essentially a one-layer neural network. By working through the math and code step by step, you should now have a solid understanding of how logistic regression works under the hood and how to implement it yourself.&#x20;
