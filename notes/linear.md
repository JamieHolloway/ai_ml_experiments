
This equation is a **linear combination** used in machine learning models, especially in **logistic regression** and **neural networks**.

$$
z^{(i)} = w^T x^{(i)} + b
$$

* $z^{(i)}$: The output of the linear function for the $i$-th example (sometimes called the "logit").
* $w$: The **weight vector** (parameters learned by the model).
* $x^{(i)}$: The **feature vector** (input data for the $i$-th example).
* $w^T$: The **transpose** of the weight vector.
* $b$: The **bias** term (a scalar, shifts the output).

---

### Why is $w$ transposed?

* **Purpose:** The transpose ensures that the matrix multiplication $w^T x^{(i)}$ gives a **scalar** (a single number) instead of a vector.
* **Shape explanation:**

  * If $w$ is a **column vector** of shape $(n, 1)$, then $w^T$ is a **row vector** of shape $(1, n)$.
  * $x^{(i)}$ is a column vector of shape $(n, 1)$.
  * So: $w^T x^{(i)}$ is $(1, n) \times (n, 1) = (1, 1) $ (a scalar).

#### Example (with numbers):

Suppose
$w = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}$,
$x^{(i)} = \begin{bmatrix} x_1^{(i)} \\ x_2^{(i)} \end{bmatrix}$.

Then,

$$
w^T x^{(i)} = [w_1\;\;w_2] \begin{bmatrix} x_1^{(i)} \\ x_2^{(i)} \end{bmatrix}
= w_1 x_1^{(i)} + w_2 x_2^{(i)}
$$

which is a **scalar**.

---

### Why is this important in AI/ML?

* This linear combination is the **core calculation** for a neuron in a neural network or for logistic/linear regression.
* After calculating $z^{(i)}$, an **activation function** (like sigmoid, ReLU, etc.) is usually applied to get the final prediction.

---

**Summary:**

* The transpose ($w^T$) aligns the shapes for correct multiplication.
* It turns a vector product into a scalar value—the fundamental step in most machine learning models.
