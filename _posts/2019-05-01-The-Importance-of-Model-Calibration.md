---
layout: post
---

When working on classification problems, it is very common to use the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) to transform the output's logits into a probability for each class. It is also important that the probability given to the predicted class reflects how certain we can be of that result. That is not the case when a model is **uncalibrated**.

For instance, imagine you have just started learning about machine learning and you are experimenting with the [_Iris_ flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set), which has 3 classes, _Iris setosa_, _Iris virginica_, and _Iris versicolor_. 

The following set of logits would be transformed into probabilities using softmax as follows:

```python
import numpy as np

def softmax(x):
    return np.exp(x)/np.exp(x).sum()

x = np.array([2.048, 0.418, 0.651])

probs = softmax(x)

print(probs)

>>> [0.69287228 0.13575417 0.17137355]
```

This means that for this given input, assuming the classes are ordered as `Iris setosa, Iris virginica, Iris versicolor`, the model has given a probability of 69.29% to the class `Iris setosa`, 13.58% to `Iris virginica`, and 17.14% to `Iris versicolor`. Thus, the input would be classified as `Iris setosa`.

But what if the model is very confident on its predictions, i.e., the softmax probabilities are high, but the overall performance of the model on those predictions is not good? When this happens we say the model is **uncalibrated**. It is easy to understand why it is desirable to have calibrated models. As machine learning models become more commonly used in critical applications (such as the detection of obstacles by autonomous vehicles, detection of fraud in online transactions, etc.), it is important to be aware of how reliable the probability given to a certain prediction is. In particular, it is possible to define alternative approaches to be used when the predictions of our model are not as reliable as we would like them to be.

---

One way to represent the calibration of the model is through **reliability diagrams**, which plot the accuracy of the predictions of the model as function of the confidence (probability given to those predictions). To obtain this plot we:

1. Define a number N of interval bins, equally-spaced, with regard to confidence values.
2. Check which samples are part of each of those intervals.
3. Obtain the accuracy for each of the intervals.
4. Plot the accuracy of the predictions according to the intervals of confidence.

Figure 1 shows an example of a reliability diagram (from Guo et al. [[1]](#ref1)). For now focus only on the blue bars, which correspond to the accuracy of the predictions within a given range of confidence values. Notice how a perfectly calibrated model corresponds to the identity function in this type of diagram. The reasoning behind this is the following: _If a model outputs a probability of 60% for a given class for a given input, then, over the course of several predictions with different inputs, all with this level of confidence, the model should have guessed 60% of them correctly_. Extending this reasoning to all values, we obtain the aforementioned identity function.

<p align="center">
    <img src="/static/fig01.png" width="40%">
    <font size="2">
        <figcaption class="text-center">Figure 1: Example of reliability diagram (modified from [1]).</figcaption>
    </font>
</p>

---

A shortcoming of the reliability diagram is the fact that it does not represent the number of samples in each bin. This could lead to erroneous conclusions, as bins with more samples end up being more important to the overall calibration of the model.

It is therefore useful to use a different metric when talking about the calibration of the model. A common alternative/complement is the **expected calibration error (ECE)**. This corresponds to the _weighted average of the differences between the accuracies and confidences of the model predictions_. To obtain this value:

1. Define a number N of interval bins, equally-spaced, with regard to confidence values.
2. Check which samples are part of each of those intervals.
3. Obtain the accuracy for each of those intervals.
4. Obtain the average confidence for each of those intervals.
5. Calculate the weighted (by the number of samples in each bin) average of the absolute differences between the accuracy and average confidence values.

The red bars in Figure 1 correspond to the gap between the accuracy and the average confidence, for each of the intervals. It is also possible to think in terms of **maximum calibration error (MCE)**. This is particularly important for applications in which it is necessary to be concerned about the worst-case scenario. In plots like the one shown in Figure 1 this corresponds to the largest red bar.

---

Guo et al. [[1]](#ref1) discusses several approaches to tackle the problem of uncalibrated models, starting with solutions that can be applied to binary models, and then extending them to multiclass problems. Here, I will focus on an extension to the [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling), namely by making use of **temperature scaling**.

The temperature value might be interpreted as the inverse of the [thermodynamics beta](https://en.wikipedia.org/wiki/Thermodynamic_beta). In particular, we can think about it in the following way:

* If we **increase** the temperature value, we will "increase the output entropy" and thus, obtain more uniform probability distributions over our classes.

* If we **decrease** the temperature value, we will "decrease the output entropy" and thus, obtain more peaked probability distributions.


The temperature, as used in Guo et al. [[1]](#ref1), and Tangseng and Okatani [[2]](#ref2), is tuned on the validation set. Notice that the temperature value only scales the outputs, and thus, the most probable class remains the same. This means that changing the temperature of the softmax won't affect the model accuracy.

Let's see how adding a temperature scaling factor would affect the result we've seen before:

```python
import numpy as np

def softmax(x, t=1):
    # Notice how the inputs `x` are divided by `t`,
    # corresponding to the temperature scaling
    return np.exp(x/t)/np.exp(x/t).sum()

x = np.array([2.048, 0.418, 0.651])

probs_t1 = softmax(x)
probs_t5 = softmax(x, 5)
probs_0pt5 = softmax(x, 0.5)

print("Temperature 1: ", probs_t1)
print("Temperature 5: ", probs_t5)
print("Temperature 0.5: ", probs_0pt5)

>>> Temperature 1:  [0.69287228 0.13575417 0.17137355]
>>> Temperature 5:  [0.40354432 0.29128039 0.30517529]
>>> Temperature 0.5:  [0.90945104 0.03491237 0.05563659]
```

As expected, increasing the temperature resulted in a more _uniform_ probability distribution, whereas decreasing the temperature resulted in a more _peaked_ probability distribution. Notice as well that, as mentioned before, the ordering of the classes from most probable to least probable, remains the same, regardless of the temperature factor.

Figure 2 shows an example resulting from applying temperature scaling to the CIFAR-100 dataset. It is noticeable how the temperature scaling method, despite simple, is very effective in terms of reducing the ECE error of the model's predictions. In this particular case, it is able to reduce the ECE from 12.67 to 0.96. This means that in the case of the calibrated model (on the right), we can now use the probability value given to the chosen class as a measure of how confident we can be in that prediction, as it reflects the accuracy of the model in that circumstance.

<p align="center">
    <img src="/static/fig02.png" width="65%">
    <font size="2">
        <figcaption class="text-center">Figure 2: Example of the application of temperature scaling (extracted from [1]).</figcaption>
    </font>
</p>

---

With this blog post I hope I sparked the interest in you to explore a little bit more about the topic of model calibration. In particular, I hope that the importance of having calibrated models, and to be able to understand when we should, or should not, rely on the output of a given model, is now clear to you.

For more details on the possible causes of miscalibration, and further theoretical details of the mentioned topics, please refer to Guo et al. [[1]](#ref1). Furthermore, one of the authors of the paper made their implementation publicly available [[3]](#ref3)·

---

#### References

<a name="ref1">[1] - Guo, Chuan, et al. "On calibration of modern neural networks." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.</a>

<a name="ref2">[2] - Pongsate Tangseng and Takayuki Okatani. "Toward explainable fashion recommendation." arXiv preprint arXiv:1901.04870, 2019.</a>

<a name="ref3">[3] - [https://github.com/gpleiss/temperature_scaling](https://github.com/gpleiss/temperature_scaling)</a>
