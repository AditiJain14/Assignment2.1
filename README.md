# Assignment2.1
# COL 341: Assignment 2

```
Notes: (See PDF)
```
- This assignment has two parts - Neural Network and Convolutional Neural Networks.
- You are advised to use vector operations (wherever possible) for best performance as the evaluation will be
    timed.
- Include a report of maximum 5 pages which should be a brief description explaining what you did. Include
    any observations and/or plots required by the question in the report.
- You should use Python for all your programming solutions.
- Your assignments will be auto-graded, make sure you test your programs before submitting. We will use
    your code to train the model on training data and predict on test set.
- Input/output format, submission format and other details are included. Your programs should be modular
    enough to accept specified parameters.
- You should submit work of your own. You should cite the source, if you choose to use any external resource.
    You will be awarded F grade or DISCO in case of plagiarism.

1.Neural Networks ( 100 points, Due date: 9:00 PM Monday, 4thOctober, 2021)
In this problem, we’ll train neural networks to classify a binary class(Toy) and multi-class(Devanagri
handwritten characters) dataset.

```
(a) (25 points) Write a program to implement a general neural network architecture. Implement the back-
propagation algorithm from first principles to train the network. You should train the network using
Mini-Batch Gradient Descent. Your implementation should be generic enough so that it can work
with different architectures(any specified learning rate, activation function, batch size etc.). Assume
a fully connected architecture i.e., each unit in a hidden layer is connected to every unit in the next
layer. Have an option for an adaptive learning rateηt=√η^0 t. Use Cross Entropy Loss(CE) as the loss
function. Usesigmoidas the activation function for the units inintermediate layers. Usesoftmax
as the activation function for the units inoutput layer. Your implementation should also work for
tanhandreluactivation functions.
```
## CE=−

## 1

```
n
```
```
∑n
```
```
i=
```
```
∑k
```
```
j=
```
```
yijlog(ˆyij) (1)
```
```
yˆij=
```
```
expzij
∑k
j′=1expzij′
```
## (2)

```
Hereiindexes samples whilejindexes class labels. Hereyiis a one-hot vector where only one value
(corresponding to true class index) is non-zero for samplei.nis the number of samples in the batch.
kis the number of labels. ˆyij∈(0,1) :
```
## ∑

```
jyˆij= 1∀i,jis the prediction of a sample. zijis the value
being input intojthperceptron of softmax layer forithsample.
```
```
Use Your Implementation to train neural network on Toy training dataset(first column correspond to
classes) with predefined parameters(corresponding to fixed learning rate) and predict on the publicly
available Toy testing dataset(first column filled with -1, labels included separately). Before starting
to implement, have a look at the evaluation criteria, submission instructions and coding guidelines for
this part.
```

- Fixed Learning Rate

```
w(t) =w(t−1)−η 0 ∇wL(w;Xb(t−1):bt,yb(t−1):bt) (3)
```
```
Heretindicates epoch number whilewdenotes model weights. bindicates batch size whileX,y
denotes data and labels respectively.η 0 ,Lrepresents fixed learning rate and loss function respec-
tively.∇wLsignifies gradient of the loss function with respect to model weights.
```
- The way to initialise weights
    We will use a standard way to initialise weights in all parts of this problem. Letwijl be the weight
    input to the layerlwherei∈ 0 ...mandj∈ 1 ...n. mandnhere are the number of neurons
    in layers numberedl−1 andlrespectively. Note that{w 0 j:j∈ 1 ...n}corresponds to the bias
    vector input to layerl. Eachwlij∼ N(0,1)∗

## √

```
2
(m+1+n)whereN(0,1) is the standard normal
distribution. This is also known as xavier initialisation. You can read about it. You must use
numpy.random.normal to initialise the weights andwlijfor anylmust be initialised using one and
only one call to numpy.random.normal. You must use numpy.random.seed to set the seed. The
value of seed must be taken as input from the user. Also the code to set the seed should only
run once for entire training phase. Weights when initialised must have the data type float32(use
numpy.float32() to do this). The datatype can turn to a higher one while running but initially it
must be float32.
```
(b) (25 points) Modify neural network architecture made in part a) to cater for multi-class dataset(Da-
vanagri handwritten characters). You have been provided image files data corresponding to train and
test sets respectively. The training dataset has 8-bit 32x32 greyscale images corresponding to to 46
Devanagari characters (last column in both training and test data is for labels). Use Mean Squared
Loss(MSE) as the loss function. Usetanhas the activation function for the units inintermediate
layersandoutput layer.

## MSE=−

## 1

```
2 n
```
```
∑n
```
```
i=
```
```
∑k
```
```
j=
```
```
(yij−yˆij) (4)
```
```
Hereiindexes samples whilejindexes class labels. Hereyiis a one-hot vector where only one value
(corresponding to true class index) is non-zero for samplei.nis the number of samples in the batch.
kis the number of labels. The value of ˆyijdepends on the activation function used in the output layer.
```
```
Use Your Implementation to train a neural network on the given training dataset with predefined
parameters(corresponding to adaptive learning rate) and predict on the publicly testing dataset.
```
- Adaptive Learning Rate

```
w(t) =w(t−1)−
```
```
η 0
√
t
```
```
∇wL(w;Xb(t−1):bt,yb(t−1):bt) (5)
```
```
Heretindicates epoch number whilewdenotes model weights. bindicates batch size whileX,y
denotes data and labels respectively. η 0 ,Lrepresents seed value and loss function respectively.
∇wLsignifies gradient of the loss function with respect to model weights.
(c) (25 points) Use Your Implementation in part a to train a neural network on the given multiclass
dataset with predefined parameters. The algorithm that you used for gradient updates in part a and
part b(mini-batch gradient descent) can have many variants, each giving an edge over the conventional
SGD algotrithm. For this part you are required to go through these variants here. You are required
to implement 5 of these which are:
```
- Momentum
- Nesterov
- RMSprop
- Adam
- Nadam


```
There are a lot of additional hyper parameters you will come across while you are implementing these.
These hyperparameters are mostly kept constant(standard values are available on the web). You may
want to do the same to prevent an exponential blow-up of hyperparameters. After implementation,
use the different forms of gradient descent and experiment which one converges faster. Try to tweak
different parameters keeping the following parameters same.
```
- softmax in the output layer
- Cross-entropy loss
Report the best architecture you get(the best is the one that reduces the loss to a minimum in least
number of iterations). Also mention in the report how did you arrive at these parameters.

```
(d) (25 points) Use Your Implementation to train a neural network on the given Devanagri handwritten
characters dataset with predefined parameters. Experiment with different types of architectures. Vary
the number of hidden layers. Try different number of units in each layer, different loss and activa-
tion functions, gradient descent variants etc. What happens as we increase the number of units or the
number of hidden layers? Comment on your observation and submit your best performing architecture.
```
```
Note: Use holdout method to find the best architecture. Split the data set into two: a set of examples
to train with, and a validation set. Keep the size of the validation set as 30% of the total training
set. Train the model on the training set. Use the prediction on the validation set to determine which
architecture to use.
```
- A note of caution:your report must clearly specify how optimal parameters were reached in
    part (c) and (d). Any cutting corners will be treated as disciplinary misconduct and will be dealt
    with accordingly.

Evaluation:

- For part-a and part-b, you can get 0 (error), partial marks (code runs fine but weights are incorrect
    within some predefined threshold) and full (works as expected). We will check obtained weights and
    predictions for grading these parts. Make sure to initialise the weights in the same way as mentioned.
    Any deviations would lead to incorrect results.
- For part-c, marks will be given based on loss value after predefined number of epochs. The lesser the
    value, higher the marks you get.
- For part-d, marks will be given based on accuracy on a variant of test data-set.
- For part-c and part-d, there will be relative marking
- For part-c and part-d marking will be done for code as well as report.

Submission Instructions:

Neural Network

Submit your code in 4 executable python files called neurala.py, neuralb.py, neuralc.py, neurald.py

The file name should corresponds with part [a,b,c,d] of the assignment.
The parameters are dependant on the mode:

- python neurala.py inputpath outputpath param.txt
    Here your code must read the input files for the toy dataset(filenames will remain same as provided)
    from theinputpath, initialise the parameters of the network to what is provided in theparam.txt
    file(present in theinputpath) and write weights and predictions tooutputpath.
    param.txtwill contain 8 lines specifying epochs, batch size, a list specifying the architecture([100,50,10]
    implies 2 hidden layers with 100 and 50 neurons and 10 neurons in the output layer), learning rate
    type(0 for fixed and 1 for adaptive), learning rate value, activation function(0 for log sigmoid, 1 for
    tanh, 2 for relu), loss function(0 for CE and 1 for MSE), seed value for the numpy.random.normal
    used(some whole number). The order will be the same as here.
    weights must be written to the outputpath for each layer in form of numpy arrays inwl.npyfile where


```
l is the index(starting from 1) of the layers 1 to output layer(example: for architecture[100,50,10], the
weight files will bew1.npy,w2.npy,w3.npy, with the last one containing weights in the last layer).
the predictions(for test data) must be a 1-D numpy array written to theoutputpathaspredictions.npy
file.
```
- python neuralb.py inputpath outputpath param.txt
    Same as for part a.
- python neuralc.py inputpath outputpath
    Here you need to read dataset frominputpath, run your best model and writepredictions.npyfile for
    thetraining datato theoutputpath.
- python neurald.py inputpath outputpath
    Same as for part except that the predictions should be of thetest data.
- Part (e)
    Here you have to submit a pdf file with all details of best architectures and features for parts c) and d).
    Report results and observations of all variations experimented with irrespective of whether they lead
    to increase in accuracy. Report must also contain how the optimal parameters were reached. There
    will be no demos or presentations for this part.

Coding Guidelines:

```
(a) For parts a) and b)
```
- Don’t shuffle/change order of the data files.
- The data should be scaled to fit 0 to 1 using 1/255 as the scaling factor. No other transformation
    must be done to the data before feeding to the network.
- Don’t apply early stopping criterion. Run for full specified number of iterations/epochs.
- Be careful to code Mini-batch Gradient Descent to be closely consistent with the theoretical
    framework taught in class.
- Make sure you do not use any random function other than numpy.random.normal. Also make
    sure the usage is as mentioned in the question.
- Make sure there are no other libraries used other than numpy, math, time, pandas and sys. Use
    of any other library is prohibited.
- In case of cross entropy loss, last layer activation function is assumed to be softmax unless otherwise
    stated.
- For you to check your implementation, we have provided the weight files and checker script on the
    given link. The weight files namelyacwl.npyare weights after 5 epochs of toy and multiclass
    datasets. The ones namedacwliter.npyare after 5 iterations of the same datasets. You can use
    the checker code to make sure the weights agree with the evaluation weights.
- Your code must not print anything on the screen.
(b) For parts c) and d)
- The validation set must be the bottom 30% of the training set and must not be selected at random.
- Your code must not print anything on the screen.
- Design code with emphasis on Vector Operations and minimal use of loops to ensure run-time
efficiency. Your code must finish executing within15 minutes.
(c) Keep checking piazza for announcements regarding the assignment.

Extra Readings (highly recommended for part (a,b)):

```
(a) Backpropogation for different loss functions
(b) Gradient descent algorithms
```

2.Convolutional Neural Networks (Score: 100 Points, Due date: 11thOct. 2021, 9PM)This
part of the assignment is to get you started with Convolutional Neural Networks (CNN) and deep learning.
You are going to experiment and get your hands dirty with deep learning frameworkPyTorch. A2.2 has
been divided into three parts, in which you will perform image classification task. (a) Implement a simple
CNN architecture and evaluate performance on Devanagari dataset, same as that of A2.1 and compare the
results of both the models. (b) A slightly more complex architecture to be evaluated on CIFAR-10 dataset.
(c) Competitive part on CIFAR-10. All the parts have to be implemented inPyTorch.
You are allowed to use from the following modules. Make sure your scripts do not include
any other modules. Python3.7, glob, os, collections, PyTorch, torchvision, numpy, pandas, skimage,
scipy, scikit-learn, matplotlib, OpenCV (4.2 or higher)

```
(a)(30 Points) Devanagari Character Classification: You are going to use the same dataset as
the Neural Network Part with46 classesof Devanagari characters. Implement the following CNN
architecture to perform the classification.
```
```
Dataset:
```
- Train Data: Download. Each data row contains 1025 columns. First 1024 values (index: 0-
    1023) correspond to the pixel value of (32×32) grayscale image (only 1 channel) of a Devanagari
    character. 1025thvalue is the class label (0-45). Train dataset has been shuffled already. Make
    sure that youdo not shuffleit.
- Public Test Data: Download. Description same as Train Data.
- Private Test Data: TBD

```
Model Architecture:
i.CONV1(2D Convolution Layer) inchannels = 1, outchannels = 32, kernel=3×3, stride = 1.
ii.BN12D Batch Normalization Layer
iii.RELUReLU Non-Linearity
iv.POOL1(MaxPool Layer) kernelsize=2×2, stride=2.
v.CONV2(2D Convolution Layer) inchannels = 32, outchannels = 64, kernel=3×3, stride = 1.
vi.BN12D Batch Normalization Layer
vii.RELUReLU Non-Linearity
viii.POOL2(MaxPool Layer) kernelsize=2×2, stride=2.
ix.CONV3(2D Convolution Layer) inchannels = 64, outchannels = 256, kernel=3×3, stride =
2.
x.BN12D Batch Normalization Layer
xi.RELUReLU Non-Linearity
xii.POOL3(MaxPool Layer) kernelsize=2×2, stride=1.
xiii.CONV4(2D Convolution Layer) inchannels = 256, outchannels = 512, kernel=3×3, stride =
2.
xiv.RELUReLU Non-Linearity
xv.FC1(Fully Connected Layer) output = 256
xvi.RELUReLU Non-Linearity
xvii.DROPOUTDropout layer withp= 0. 2
xviii.FC2(Fully Connected Layer) output = 46
```
```
Instructions:
CNN Specifications:
```
- Your model has been designed to be compatible with the input size of [C= 1,H= 32,W= 32].
- Read more about ReLU, Batch Normalization and Dropouts online in order to understand their
    use and benefits.
- UseCross Entropy Loss(torch.nn.CrossEntropyLoss())function andAdam Optimizer
    torch.optim.Adam()with fixed learning ratelr=1e-4.


- Train your model for8 epochs.
- keep a track of average training loss for each epoch and test accuracy after each epoch. You will
    be asked to export these 8 loss and 8 accuracy values in a separate file during submission for the
    evaluation purpose.

```
Data Loader:You have been provided with a custom dataloader boiler-plate code for Devanagari to
get you started quickly. Link TBD.
```
```
Report:
```
- Report public test accuracy and other observations.
- Line plot to show training loss on the y-axis and the epoch on the x-axis.
- Calculate the accuracy of public test data at the end of each epoch and plot a line graph with
    accuracy on y-axis and number of epochs on the x-axis.
- Compare the accuracy with the Neural Network and argue which one performs better in terms of
    accuracy, and training time. Comment on these observations.
Extra Readings:
- Getting started with the PyTorch
- PyTorch tutorial on YouTube
- Dropout
- Batch Normalization

```
(b) (20 Points) CIFAR-10 ClassificationPerforming a10 classimage classification by using the
below defined architecture.
Dataset:
```
- Train Data: TBD.
- Public Test Data: TBD.
- Private Test Data: TBD
Model Architecture:
i. To be provided. It will be similar to that of part (a).
CNN Specifications:
- Your model has been designed to be compatible with the input size of [C= 1,H= 32,W= 32].
- UseCross Entropy Loss(torch.nn.CrossEntropyLoss())function andAdam Optimizer
torch.optim.Adam()with fixed learning ratelr=TBD.
- Train your model forN epochs.
- keep a track of average training loss for each epoch and test accuracy after each epoch. You will
be asked to export these N loss and N accuracy values in a separate file during submission for the
evaluation purpose. To be provided. (N=TBD)
Data Loader:Link to custom data loader. TBD
Report:Same as Part(a)

```
(c) (50 Points) CIFAR-10 Competition:Here you have to come up with your own architecture for
CIFAR-10 classification. You are free to experiment with all type of loss functions, optimizers and
learning rates. This is a competitive part in which you will be evaluated over aprivate test dataset.
Your model should have less than5 millionparameters (including both trainable and non-trainable).
```
Submission Guidelines: TBD

Coding GuidelinesSame as Q1.

Grading Scheme: TBD


