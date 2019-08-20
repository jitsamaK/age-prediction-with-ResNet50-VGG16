# <center> <b>Age Prediction

## <b>Introduction
Age prediction is a popular problem in computer vision under the umbrella of pattern recognition. As a technical perspective, this problem is widely used to test new computer vision algorithms. There are plenty of new models developed in this recent years according to the increase in high computation power, and two main benchmarks used to evaluate how much they can defeat the previous models are accuracy and time consumption.<br>

For application, age prediction is also beneficial to many areas when forgetting about privacy. An example is a business such as a supermarket. The manager will be able to know how customers in each age group walk through a shelf-stacker differently. The information can be used to rearrange goods for more appropriate with the target group and better income.<br>

Two deep learning models, ResNet (Residual Networks) and VGG (Visual Geometry Group), will be used in this study for age prediction. ImageNet and VGGface are used as the model weight, respectively. These model performances will be compared together in the condition that all layers are frozen. The model which provides the best result will be tested further by releasing weight for more layers. Finally, this process will end when the model performance drops along with adding more trainable layers.<br>


## <b> Methodology

To do an age prediction, there are 6 main processes as below; <br>
1. Prepare working environment
2. Load data (in this case, the data is images)
3. Preprocess data
4. Construct and test prediction models
5. Evaluate the model performance
6. Conclude and discuss the result


## <b> The datasorce
Please download the data from: https://talhassner.github.io/home/projects/Adience/Adience-data.html

## <b> Required Libraries
- Python 3.6.8
- Keras 2.2.4
- OpenCV 4.1.0
- Sklearn 0.21.3
- joblib 0.13.2

## Deep learning architecture
- Resnet50
- VGG16
    
## Pre-trained weight 
- VGGFace
- Imagenet

## Conclusion and dicussion
The best model performace is on VGG16 whcih adopts pre-train-weight from VGGFace. All model evalution matrixes reveal the same conclusion. However, the model is better when the weights at the top 2 layers are realesed. When realeasing more weight, the model performance declines dramatically. A possible reason is that the number of training instances is not enough to improve the model.<br>

This conclusion that VGG16 is better than ResNet50 makes sense as it is a lot shallow than ResNet50. Although ResNet50 uses residual block, there is no garuntee how many blocks are skipped. Additionally, VGG16 adopts weight from VGGFace which the problem is closer to this study problem, realted to human face, than ImageNet in ResNet50.<br>

Considering each prediction class, the model predicts 0-6 years old at the most accurate. Human face around the ages is also easily to differentiate from other age groups evthough it is the human task. Bad prediction is on 55-100 years old and this result cases by the lowest number of instances.


## Suggestion
The suggestion for future analysis is to compare different model structure with different pre-train-weight and vice versa. Changing learning rate to see better accuarcy and loss across epcoh is also suggested. This experiment adopts default library learning rate and it is quite too much for using with pre-train-weight. Besides, it hardly see the differences in accuracy and loss across epcoh. Finally, changing input size to see how it affects the model performance is the possible experiment.


