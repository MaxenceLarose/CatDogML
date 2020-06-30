# CatDogML

### [Tutorial by Dipanjan Sarkar](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)

### Prepare the data

The dataset was downloaded from the famous [cats and dogs kaggle competition](https://www.kaggle.com/c/dogs-vs-cats/data).

It contains 25000 images of cats and dogs (all labelled by their filename).

We ran a small script `prepData.py` to split the dataset into training, validation and test. For the purpose of this tutorial we only kept 300, 1000 and 1000 images for train, val and test sets respectively (equal amount of cats and dogs in each).

Our dataset folder looks like this

```
trainingData/
    |-- cat.2.png
    |-- cat.21.png
    |-- dog.50.png
    |-- ...
validationData/
    |-- cat.17.png
    |-- ...
testData/
    |-- dog.11.png
    |-- ...
```