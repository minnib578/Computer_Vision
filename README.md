# Computer_Vision: 
Computer vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs — and take actions or make recommendations based on that information

Application:
* Object detection
* Object classification
* Scene understanding
* Semantic scene
* segmentation
* 3D reconstruction
* object tracking
* human pose estimation
* activity recognition


#  Image classification:
* what is image classification?

Image classification is the process of categorizing and labeling groups of pixels or vectors within an image based on specific rule

- K-Nearest Neighbor
- Linear classifiers: SVM, Softmax
- Two-layer neural network
- Image features

* Challenges for image classification?
   * viewpoint variation: all pixels change when the camera moves
   * background cluster: objects are similar to the background (color or texture)
   * illumination: (too dark or too light)-->data driven x
   * occlusion: hiden by other objects
   * deformation: different shape/pose of the same objects
   * intraclass variation:differnt types of the same object
   * scale variation:Visual classes often exhibit variation in their siz
   
* Data-driven methods: k and distance choised, evaluation methods
   * Nearest Neighbor classifier:The nearest neighbor classifier will take a test image, compare it to every single one of the training images, and predict the label of the closest training image.One of the simplest possibilities is to compare the images pixel by pixel and add up all the differences. In other words, given two images and representing them as vectors I1,I2 , a reasonable choice for comparing them might be the L1 distance or L2 distance:  using vectoriation calculation
    
      * L1 (Manhattan) distance

          ![image](https://user-images.githubusercontent.com/63558665/120114755-a7b1f680-c14e-11eb-9122-f4c75d58a0b4.png)

      * L2 distance:(Euclidean) distance

          ![image](https://user-images.githubusercontent.com/63558665/120114899-4f2f2900-c14f-11eb-8e70-00fdf0e263e5.png)

         * ad: simple to implement and understand, no need to training
         * disad:computation cost, slow for training is ok, but we want fast prediction-->real time 

   * k-nearest neighbor classifier:instead of finding the single closest image in the training set, we will find the top k closest images, and have them vote on the label of the test image. when k==1, it is nearest neghbor classifier. knn with pixel distance never used because it is sensitive to lighting, and distance metrics on pixels are not informative.
        * disad: The classifier must remember all of the training data and store it for future comparisons with the test data. This is space inefficient because datasets may easily be gigabytes in size. Classifying a test image is expensive since it requires a comparison to all training images.
        * how to choose K and how to choose distance metric?--->choose hyperparameters working well,when K = 1 always works perfectly on training data-->Split data into train, val; choose hyperparameters on val and evaluate on test
        * k-Nearest Neighbor with pixel distance never used because Distance metrics on pixels are not informative and slow<-- light change affect performance
       
           ![image](https://user-images.githubusercontent.com/63558665/120115147-63bff100-c150-11eb-8425-4d92a49cdcff.png)
       
   * Tuning hyperparameters:
        * validation: split the training dataset into training and validation and evaluate the model on test set (don't touch the test set) 70%-90%
        * cross_validation:split data into folds,try each fols as validation and average the results (useful for small datasets),computation expensive-->validation approach.
        * Choose hyperparameters using the validation set;

# linear classification: y=f(x,w)=wx+b0
   
   ![image](https://user-images.githubusercontent.com/63558665/120711764-c20d0c80-c48d-11eb-8bcc-d2c360303188.png)

* Loss: Define a loss function that quantifies our unhappiness with the scores across the training data. Come up with a way of
efficiently finding the parameters that minimize the loss function. (optimization)

   ![image](https://user-images.githubusercontent.com/63558665/120116601-d6cc6600-c156-11eb-8826-46c3bdeca9d0.png)
      
* SVM loss: correct classification is 0, l1 and l2 SVM loss:difference in score between correct and incorrect class

     ![image](https://user-images.githubusercontent.com/63558665/120117034-257aff80-c159-11eb-9ced-793c3a178ba3.png)

     ![image](https://user-images.githubusercontent.com/63558665/120116958-d6cd6580-c158-11eb-9dde-6916a3e0287c.png)
       
     * questions:
         * if the correct score decrease maybe not affect the loss--> loss may not change
         * what is min/max loss-->min:0 and max:infinite
         * at inilization, weight is small so all s about to 0
         * sum of loss or mean of loss or square of loss?
         * whether w is unique? it is no unique w=2w and L=0---> which is better w or 2w?--->reguralization
         * The same output with different weight and how to determine which is better?<--reguralization 
           
          ![image](https://user-images.githubusercontent.com/63558665/120713457-ebc73300-c48f-11eb-851f-4e1b608317fd.png)

* Regularization:Prevent the model from doing too well on training data
           
    ![image](https://user-images.githubusercontent.com/63558665/120117062-522f1700-c159-11eb-9c99-b09383b1de6b.png)

    * methods:
            ![image](https://user-images.githubusercontent.com/63558665/120117110-9de1c080-c159-11eb-8795-d6ed8875a0ef.png)
    * why?
        * Express preferneces over weights
        * make the model simple so it works test data
        * improve optimization by adding curvature
                                 
* Softmax: interpret classifier score into probability, probability sum to 1,Choose weights to maximize the
likelihood of the observed data
            
   ![image](https://user-images.githubusercontent.com/63558665/120117288-763f2800-c15a-11eb-9c20-b11ab39c2090.png)
     
   ![image](https://user-images.githubusercontent.com/63558665/120117410-0f6e3e80-c15b-11eb-92cc-1a67f803002a.png)
   
   * Question:
        * What is the min/max possible softmax loss Li?
        * At initialization all sj will be approximately equal; what is the softmax loss Li, assuming C classes?
   
   ![image](https://user-images.githubusercontent.com/63558665/120714559-5cbb1a80-c491-11eb-93b9-ddc8fbed5a4b.png)

* Optimization: how to find the best w?
     * strategy 1: random search
     * startegy 2: follow the slope-->gradient descent

Three loss function: linear loss, SVM loss, softmax and data loss_reguralization     
        
# Neural network-multiple layers neural network
* Linear classifier is not useful and can only draw  linear decision boundaries-->featuere transformation: f(x, y) = (r(x, y), θ(x, y))

   ![image](https://user-images.githubusercontent.com/63558665/120117682-54df3b80-c15c-11eb-9cbc-26906f99b548.png)

   ![image](https://user-images.githubusercontent.com/63558665/120117694-6294c100-c15c-11eb-8fc0-13958b764a2d.png)

* why activation?--> W2*W1=W3  end up with linear classifier again!
   
   ![image](https://user-images.githubusercontent.com/63558665/120117745-ab4c7a00-c15c-11eb-8a22-2ff3352efbe1.png)
   
   ![image](https://user-images.githubusercontent.com/63558665/120117758-b6070f00-c15c-11eb-8bf9-efc7eb33bcd4.png)
   
* Multiple layer Neural network:
   
   ![image](https://user-images.githubusercontent.com/63558665/120117774-ca4b0c00-c15c-11eb-8930-57187382e583.png)

* Derive delta_w L on paper?
    * Very tedious: Lots of matrix calculus, need lots of paper
    * What if we want to change loss? E.g. use softmax instead of SVM? Need to re-derive from scratch
    * Not feasible for very complex models!
--> Backpropagation+ computational graph-->chain rule

   ![image](https://user-images.githubusercontent.com/63558665/120716932-b8d36e00-c494-11eb-8a9d-c4a1c07826cd.png)

* Summary
    * (Fully-connected) Neural Networks are stacks of linear functions and nonlinear activation functions; they have much more representational
    power than linear classifiers
    * backpropagation = recursive application of the chain rule along a computational graph to compute the gradients of all inputs/parameters/intermediates
    * implementations maintain a graph structure, where the nodes implement the forward() / backward() API
    * forward: compute result of an operation and save any intermediates needed for gradient computation in memory
    * backward: apply the chain rule to compute he gradient of the loss function with respect to the inputs


# linear classifier--> multiple layers neural network-->covolution network
* Architecture:
   
    ![image](https://user-images.githubusercontent.com/63558665/120119457-cc659880-c165-11eb-8f05-5f72a60b3440.png)
  
    ![image](https://user-images.githubusercontent.com/63558665/120119468-d4253d00-c165-11eb-9a37-22f349719691.png)
   
    ![image](https://user-images.githubusercontent.com/63558665/120119656-e05dca00-c166-11eb-9929-155a02c62103.png)
   
* Low-level features-->High level features
convolution architecture: convolution layer-->ReLU-->pooling layer-->fully connected layer
   
   ![image](https://user-images.githubusercontent.com/63558665/120119722-53ffd700-c167-11eb-8176-f08cdaa0f499.png)

   ![image](https://user-images.githubusercontent.com/63558665/120119757-87426600-c167-11eb-9981-ebc72605b8b7.png)

   In general, common to see CONV layers with stride 1, filters of size FxF, and zero-padding with (F-1)/2. (will preserve size spatially)

   ![image](https://user-images.githubusercontent.com/63558665/120119883-1cddf580-c168-11eb-9e3f-3c14d6c66890.png)
   
* Pooling:Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting.
    * perform a sub‐sampling to reduce the size of the feature map
    * merge the local semantically similar features into a more concise representation
    * Max pooling – major method
    * Average pooling
    * The effect of overlapping pooling in AlexNet is not significant
    ![image](https://user-images.githubusercontent.com/63558665/120119992-bf967400-c168-11eb-8adb-b38fb349403b.png)
   
* Padding:control the spatial size of the output volumes,most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same
   
# Convolutional Neural Networks
Detection and segmentation，classification,image caption, style transfer learning

* Fully connect layer:

 ![image](https://user-images.githubusercontent.com/63558665/120720682-83318380-c49a-11eb-908e-b765b866451d.png)

* convolution layer: 
       * preserve spatial structure,convolve (slide) over all spatial locations
       * different filter for different features extraction
       * low level features--> high level features-->linear separable classifier
       * 1x1 convolution reduce diemsnion with few parameters
* convolution Layer change:
 output size: (N - F) / stride + 1     (N + 2P - F) / stride + 1
 parameters: F2CK+k
 
* Pooling layer: makes the representations smaller and more manageable, downsample,operates over each activation map independently
        * Maxpooling
        * global pooling
        * average pooling
             
number of parameters is 0
* Tips:
    * Trend towards smaller filters and deeper architectures
    * Trend towards getting rid of POOL/FC layers (just CONV)
    * conv-->ReLu-->pool-->softmax
        
# Training
* one time setup: activation functions, preprocessing, weight initialization, regularization, gradient checking
      
   1. activation function
            
    * sigmoid:
         * Saturated neurons “kill” the gradients (-inf,0 (delta(x)=1),inf)--> the gradients flowing back will be zero and weights will never change
         * Sigmoid outputs are not zero-centered (output>0)--> local gradient of sigmoid is always positive or negative-->zigzag
         * exp() computation expensive
         * Always all positive or all negative-->zig zag path
     * Tanh:
         * Saturated neurons “kill” the gradients (-inf,0,inf)--> the gradients flowing back will be zero and weights will never change
         * Sigmoid outputs are zero-centered (nice)
         * Squashes numbers to range [-1,1]
     * ReLU:
         * not saturate (in +region)
         * computation efficiently
         * converge mush faster than sigmoid/Tanh in pratice (eg.6x)
         * not zero-centered output
         * an annoyance
         * Dead ReLU will never activate and no update weight
      * LeakyReLU: f(x)=max(0.001x,x)
         * not saturate (in +region)
         * computation efficiently
         * converge mush faster than sigmoid/Tanh in pratice (eg.6x)
         * not zero-centered output
         * will not "die"
         * need to manually setup a
       * exponential Linear Units (ELU):
         * all benefits of ReLU
         * closer to zero mean output
         * negative saturation regime compared with Leakly ReLU add some robutness to noise
         * computation require exp()
       * Parametric Rectifier（PReLU)
          f(x)=max(ax,x)
       * Scaled Exponential Linear Units (SELU): scale the output
           * Scaled versionof ELU that works better for deep networks
           * “Self-normalizing” property
           * Can train deep SELU networks without BatchNorm
           * computation require exp()
       * Maxout “Neuron”
           * Does not have the basic form of dot product ->nonlinearity
           * Generalizes ReLU and Leaky ReLU
           * Linear Regime! Does not saturate! Does not die!
           * Problem: doubles the number of parameters/neuron
       * Summary:
           * Using ReLu be careful with learning rate
           * Don’t use sigmoid or tanh
           * Try out Leaky ReLU / Maxout / ELU / SELU--To squeeze out some marginal gains

      
   2. data preprocessing:consider what happends when the input to a neura is always positive/negative--->zigzag path
     
     * zero-mean data--> visualize data with PCA and Whitening
          * substract the mean image(AlexNet)
          * substract per-channel mean (VGGNet)
          * substract per-channel mean and divide by per-channel std (ResNet)
     
     * normalization:Before normalization: classification loss very sensitive to changes in weight matrix; hard to optimize. After normalization: less sensitive to small changes in weights; easier to optimize
      
   3. weight initialization: different initialization point with different model performance with the same parameters
     
     * small random numbers:(gaussian with zero mean and 1e-2 standard deviation)-->work with small network, but no deep network---> vanishing gradient-->no learning
     
     * all activation tend to zero for deeper network layers--> all zero no lerning-->local gradient tend to zero
     
     * “Xavier” Initialization: Activations are nicely scaled for all layers-->ReLU Activations collapse to zero again-->
                     
          ![image](https://user-images.githubusercontent.com/63558665/120121487-c4135a80-c171-11eb-8561-d06a4b12ef89.png)
             
     * Kaiming/MSRA initilization
         
          ![image](https://user-images.githubusercontent.com/63558665/120121622-70554100-c172-11eb-992e-930fa4462046.png)
              
     * depending on differnt activation function using differnt weight inilization
      
   4. Regularization: zero-mean unit-variance activationa and improve single model performance
     
     * batchnormalization: (zero-mean unit-variance) During testing batchnorm becomes a linear operator! Can be fused with the previous fully-connected or conv layer
          
          ![image](https://user-images.githubusercontent.com/63558665/120121738-1dc85480-c173-11eb-814f-616134640c4c.png)
          
          * Makes deep networks much easier to train!
          * Improves gradient flow
          * Allows higher learning rates, faster convergence
          * Networks become more robust to initialization
          * Acts as regularization during training
          * Zero overhead at test-time: can be fused with conv!
          * Behaves differently during training and testing: this is a very common source of bugs!
          * layer batchnormalization used in recurrent networks
          * instance normalization/group normalization
          * Usually inserted after Fully Connected or Convolutional layers, and before nonlinearity
            
              ![image](https://user-images.githubusercontent.com/63558665/120122407-882ec400-c176-11eb-9a3a-d2856af4d60a.png)
           
     * dropout:In each forward pass, randomly set some neurons to zero Probability of dropping is a hyperparameter; 0.5 is common
          * Dropout is training a large ensemble of models (that share parameters).
          * Each binary mask is one model
          * drop in train and scale in test
          * Dropout is an effective method to suppress overfitting
          * Dropout layer randomly deletes some neurons from the dense layers.
          * It can reduce complex co‐adaptations of neurons and force the neural network to learn more robust features
      * data augmentation:
          * Random crops and scales
          * Color Jitter
          * translation/rotation/stretching/shearing/lens distortions/Cutout/random crop/mixup
          * add random noise
      * Dropconnect: Drop connections between neurons (set weights to 0)
      * Fractional Pooling:Use randomized pooling regions
      * stochastic depth: Skip some layers in the network
      
5. Improve training error:
   1. Optimizer:
   * Gradient Descent:
     * Batch gradient descent
         * Advantages:
                 Easy computation.
                 Easy to implement.
                 Easy to understand.
         * Disadvantages:
                 May trap at local minima.
                 Weights are changed after calculating gradient on the whole dataset. So, if the dataset is too large than this may take years to converge to the minima.
                 Requires large memory to calculate gradient on the whole dataset.
     * Stochastic gradient descent
         * Advantages:
                 Frequent updates of model parameters hence, converges in less time.
                 Requires less memory as no need to store values of loss functions.
                 May get new minima’s.
         * Disadvantages:
                 High variance in model parameters.
                 May shoot even after achieving global minima.
                 To get the same convergence as gradient descent needs to slowly reduce the value of learning rate.
     * Mini-batch gradient descent
         * Advantages:
                 Frequently updates the model parameters and also has less variance.
                 Requires medium amount of memory.

      All types of Gradient Descent have some challenges:
      Choosing an optimum value of the learning rate. If the learning rate is too small than gradient descent may take ages to converge.
      Have a constant learning rate for all the parameters. There may be some parameters which we may not want to change at the same rate.
      May get trapped at local minima.

  * Adaptive:
     * Momentum: Momentum was invented for reducing high variance in SGD and softens the convergence. It accelerates the convergence towards the relevant direction and reduces the fluctuation to the irrelevant direction.
         * Advantages:
                 Reduces the oscillations and high variance of the parameters.
                 Converges faster than gradient descent.
          * Disadvantages:
                 One more hyper-parameter is added which needs to be selected manually and accurately.
  * Adagrad: One of the disadvantages of all the optimizers explained is that the learning rate is constant for all parameters and for each cycle. This optimizer changes the learning rate. It changes the learning rate ‘η’ for each parameter and at every time step ‘t’. learning rate which is modified for given parameter θ(i) at a given time based on previous gradients calculated for given parameter θ(i).
      * Advantages:
             Learning rate changes for each training parameter.
             Don’t need to manually tune the learning rate.
             Able to train on sparse data.
       * Disadvantages:
             Computationally expensive as a need to calculate the second order derivative.
             The learning rate is always decreasing results in slow training.
  * Adadelta:It is an extension of AdaGrad which tends to remove the decaying learning Rate problem of it. Instead of accumulating all previously squared gradients, Adadelta limits the window of accumulated past gradients to some fixed size w. In this exponentially moving average is used rather than the sum of all the gradients.
      * Advantages:
             Now the learning rate does not decay and the training does not stop.
      * Disadvantages:
             Computationally expensive.
  * Adam: Adam (Adaptive Moment Estimation) works with momentums of first and second order. The intuition behind the Adam is that we don’t want to roll so fast just because we can jump over the minimum, we want to decrease the velocity a little bit for a careful search. In addition to storing an exponentially decaying average of past squared gradients like AdaDelta, Adam also keeps an exponentially decaying average of past gradients M(t).
      * Advantages:
             The method is too fast and converges rapidly.
             Rectifies vanishing learning rate, high variance.
      * Disadvantages:
             Computationally costly.
how to choose a optimizer?
Adam is the best optimizers. If one wants to train the neural network in less time and more efficiently than Adam is the optimizer.
For sparse data use the optimizers with dynamic learning rate.If, want to use gradient descent algorithm than min-batch gradient descent is the best option. 
    
   2. Learning rate schedule
   
* Improve test error: 
    * Regularization to improve single-model performance. Data augmentation to improve test time performance
    * Choosing hyperparameters:
    * Network architectures
    * Learning rate, its decay schedule, update type
    * Regularization (L2/Dropout strength)
 

    
        
6. Evaluation：
     * Model ensembles: Train multiple independent models,At test time average their results
     * Test-time augmentation
     * Transfer learning:Free part of pretained model and reinitialized fine tune part
         * Lower learning rate when finetuning; 1/10 of original LR is good starting point
         * With bigger dataset, train more layers, with small dataset train less layer(fcl)

              ![image](https://user-images.githubusercontent.com/63558665/120121953-13f32100-c174-11eb-9ed4-ad345d1f13b6.png)
                 
              ![image](https://user-images.githubusercontent.com/63558665/120122009-3f760b80-c174-11eb-8ced-bc8ab1c32ab5.png)
             
          * They also find that collecting more data is better than finetuning on a related task

* Summary
    * Activation Functions (use ReLU)
    * Data Preprocessing (images: subtract mean)
    * Weight Initialization (use Xavier/He init)
    * Batch Normalization (use this!)
    * Transfer learning (use this if you can!)
    * Consider dropout for large-->fully-connected layers
    * Batch normalization and data augmentation almost always a good idea
    * Try cutout and mixup especially for small classification datasets
    * Training Dynamics  
    * babysitting the learning process
    * Learning rate decays over time/cosine/linear
                
         ![image](https://user-images.githubusercontent.com/63558665/120122292-e1e2be80-c175-11eb-8ddc-0cca78a8a650.png)
                
     * Adam is a good default choice in many cases; it often works ok even with constant learning rate.SGD+Momentum can outperform Adam but mayrequire more tuning of LR and schedule-->Try cosine schedule, very few hyperparameters! If you can afford to do full batch updates then try out-->L-BFGS (and don’t forget to disable all sources of noise)
    * parameters update, hyperparameter optimization: check initial loss-->overfit a small sample--> find LR that makes loss go down--> cooarse grid, train for ~1-5 epochs-->refine grid, train longer-->look at loss and accuracy curves
    * early stop
    * Train multiple independent models, At test time average their results
    * Instead of using actual parameter vector, keep a moving average of the parameter vector and use that at test time
    * Huge train / val gap means overfitting! Increase regularization to get more data
    * No gap between train / val means underfitting: train longer, use a bigger model

# CNN architecture
* LeNet-5: Recognizing simple digit images

  ![image](https://user-images.githubusercontent.com/63558665/120122674-2ec79480-c178-11eb-9ddf-e0fea9dcef24.png)

    * ad:
        * Possesses the basic units of convolutional neural network,such as convolutional layer, pooling layer and full connection layer
        * Every convolutional layer includes three parts: convolution (5x5+2 padding), pooling(2x2+2 strid), and nonlinear activation functions(Tanh)
        * Using convolution to extract spatial features and weight share to reduce the computation complexity
        * Downsampling average pooling layer
        * Tanh activation function
        * Using MLP as the last classifier
        * Sparse connection between layers to reduce the complexity of computation
        * 对于分类（Classification）问题，利用全局平均池化（Global Average Pooling, GAP）操作代替特征图的拉伸，这样 CNN 便可以处理各种尺度的图像了
    * disad:
        * Tanh--> vanishing gradient
        
* AlexNet:

    * five conv layers and 5 layers (11x11,5x5,3x3)
    * Max pooling is applied between every two conv layers
    * After the tensors are flattened, two fully‐connected (dense) layers are used
    * The output layer is a softmax layerto compute the softmax loss function for learning
    * AlexNet uses both data augmentation and dropout layers
    * Training parameters:
        * Batch size:128
        * momentum:0.9
        * learning rate:0.001
    
    ![image](https://user-images.githubusercontent.com/63558665/120122712-6df5e580-c178-11eb-8ce1-61df9485e55f.png)

    ![image](https://user-images.githubusercontent.com/63558665/120122711-68000480-c178-11eb-8454-f9cfe86285db.png)
    
    * ad:
        * Replace Tanh with ReLU-->gradient vanishing and faster converge
        * Local Response Normalization-->overfitting
        * Pooling with smaller stride than convolution stride--> overlapping pooling-->overfitting
        * Data augmentation with random crop and reflection
        * Dropout--> avoiding overfitting. 
        * The unique advantage of AlexNet is the directly image input to the classification model
        * double GPU network structure
    * disad:
        * AlexNet is NOT deep enough compared to the later model such asVGG Net, GoogLENet, and ResNet
        * The use of large convolution filters (5*5 and 11*11) is not encouraged shortly after that
        * Use normal distribution to initiate the weights in the neural networks cannot effective solve the problem of gradient vanishing, replaced bythe Xavier method late
        * max-pooling layers result in loss of accurate spatial information
        * Smaller compute, still memory heavy, lower accuracy
    
* ZFNet: Improved hyperparameters over AlexNet
   Alexnet: CONV1: change from (11x11 stride 4) to (7x7 stride 2)
   CONV3,4,5: instead of 384, 384, 256 filters use 512, 1024, 512

* VGGNet:Deeper network--> complex network will overfit training data

    ![image](https://user-images.githubusercontent.com/63558665/120122818-46534d00-c179-11eb-89eb-de3a5eff18ab.png)
    
    * ad:
        * Multiple conv layers to form conv layer group then pooling
        * Instead of using large receptive fields like AlexNet (11x11 with a stride of 4), VGG uses very small receptive fields (3x3 with a stride of 1). Only 3x3 CONV stride 1, pad 1 and 2x2 MAX POOL stride 2 compared with AlexNet
        * VGG incorporates 1x1 convolutional layers to make the decision function more non-linear without changing the receptive fields.
        * The small-size convolution filters allows VGG to have a large number of weight layers
        * more layers leads to improved performance. Deeper network--> 8 layers AlexNet-->16 layers-19 layers(VGG16Net)
        * No LRN
        * Why use smaller filters? (3x3 conv) ?
           
           Stack of three 3x3 conv (stride 1) layers has same effective receptive field as one 7x7 conv layer.But deeper, more non-linearities.fewer parameters
    
    * disad: 
        * There are only a few exceptions when multi-scale training images are involved
        * Most memory is in early conv
        * Most params are in late FC
        * Computation expensive
        * most parameters, most operations
        
* GoogleNet: deeper but more computational

    22 layers--Efficient “Inception” module--No FC layers
    1x1 convolution-->dimension reduction +preserves spatial dimensions, reduces depth!
    The larger the model and the more the network parameters, the more likely it is to produce over-fitting. Therefore, a larger data set is required. However, the construction cost of large data sets is very high; the larger the model, the greater the cost of computing resources. The greater the demand, this is unacceptable in actual tasks. 
    
    * Inception Net V1:
        * ad
            * fewer parameters
            * Increase width not only depth
            * Replace the fully connected layers by the sparse ones
            * feature extract from 3x3 and 5x5 conv more thatn others--> more parameters-->“bottleneck” layers that use 1x1 convolutions to reducefeature channel size
            * Multiple receptive field sizes for convolution (1x1, 3x3, 5x5)
            * a global average pooling layer is used that spatially averages across each feature map, before final FC layer.
            * No FC layers
 
              ![image](https://user-images.githubusercontent.com/63558665/120852679-84b98500-c548-11eb-82af-391a7231fe99.png)
              
              ![image](https://user-images.githubusercontent.com/63558665/120123059-c4fcba00-c17a-11eb-8796-d1cfe67033b6.png)
    
        * disad:
            * Computational complexity
            * Pooling layer also preserves feature depth, which means total depth after concatenation can only grow at every layer!
            
    * Inception Net V2: batch normalization
        * 5x5 replace with 2 3x3 conv
        * Average pooling +max pooling
        * 两个 Inception Module 之间不再进行池化操作
        * 将网络第一层的卷积层替换为深度乘子为 8 的可分离卷积 Separable Convolution
        * 增大学习率，移除 Dropout，减小 L2 正则化项，加速学习率衰减，移除 LRN，更彻底的打乱训练数据，减少光学畸变

          ![image](https://user-images.githubusercontent.com/63558665/120853157-3c4e9700-c549-11eb-918e-51540262889a.png)
          
    * Inception Net V3:
        * Avoid representational bottlenecks, especially early in the network--> n × 1 和 1 × n to replace  nxn conv
        * Higher dimensional representations are easier to process locally within a network.
        * Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.
        * Balance the width and depth of the network.
        * Auxiliary Classifier--> setting additional loss in differnt inception (3c +4e)--> avoid vanishing gradient+improve training speed--reguralization
        * Auxiliary classification outputs to inject additional gradient at lower layers
          
          ![image](https://user-images.githubusercontent.com/63558665/120854015-81bf9400-c54a-11eb-926f-4abf6957cc68.png)
    
    * Inception Net V4: combine Inception Module, Residual Connection, Depthwise Seperable Convolution
    
        ![image](https://user-images.githubusercontent.com/63558665/120123142-2fadf580-c17b-11eb-99fb-3cba5243e6db.png)
    
    * Xception: 利用 Depthwise Separable Convolution 对 Inception V3 进行了改进，并结合 Residual Connection
            
         ![image](https://user-images.githubusercontent.com/63558665/120854561-56897480-c54b-11eb-9e75-bd4eb4436410.png)

    * ResNet:
         * No FC layers besides FC 1000 to output classes
         * Global average pooling layer after last conv layer
         * Additional conv layer at the beginning (stem)
         * Stack residual blocks and Every residual block has two 3x3 conv layers
         * double # of filters and downsample spatially using stride 2 (/2 in each dimension)
         * For deeper networks,use “bottleneck” layer to improve efficiency (similar to GoogLeNet) (50,101,152)
         * Moderate efficiency depending on model, highest accuracy
         
            ![image](https://user-images.githubusercontent.com/63558665/120123160-43f1f280-c17b-11eb-8a25-a96833914993.png)
        
         * Training:
             * Batch Normalization after every CONV layer
             * Xavier initialization from He et al.
             * SGD + Momentum (0.9)
             * Learning rate: 0.1, divided by 10 when validation error plateaus
             * Mini-batch size 256
             * Weight decay of 1e-5
             * No dropout used
         
        The deeper model performs worse, but it’s not caused by overfitting!-->deeper models are harder to optimize
        
        Deeper models are harder to optimize--> A solution by construction is copying the learned layers from the shallower model and setting additional layers to identity mapping.
        
    * sENet: Improving ResNets
        * Add a “feature recalibration” module that learns to adaptively reweight feature maps
        * Global information (global avg. pooling layer) + 2 FC layers used to determine feature map weights
        
          ![image](https://user-images.githubusercontent.com/63558665/120859229-e8947b80-c551-11eb-8ba1-167845b1bc65.png)
    
    * Identity Mappings in Deep Residual Networks
        * Improved ResNet block design from creators of ResNet
        * Creates a more direct path for propagating information throughout network
        * Gives better performance
        
          ![image](https://user-images.githubusercontent.com/63558665/120859421-2b565380-c552-11eb-8248-29b52991fe7f.png)
   
   * Wide Residual Networks
       * Argues that residuals are the important factor, not depth
       * User wider residual blocks (F x k) filters instead of F filters in each layer)
       * 50-layer wide ResNet outperforms 152-layer original ResNet
       * Increasing width instead of depth more computationally efficient (parallelizable)
       
           ![image](https://user-images.githubusercontent.com/63558665/120859573-63f62d00-c552-11eb-9aed-ec8bb001fe63.png)
    
    * DenseNet
        * Dense blocks where each layer is connected to every other layer in feedforward fashion
        * Alleviates vanishing gradient, strengthens feature propagation, encourages feature reuse
        * Showed that shallow 50-layer network can outperform deeper 152 layer ResNet
        
           ![image](https://user-images.githubusercontent.com/63558665/120859754-a7509b80-c552-11eb-9b4b-1c9322ce09ae.png)
    
    * MobileNets
        * Depthwise separable convolutions replace standard convolutions by factorizing them into a depthwise convolution and a 1x1 convolution
          
          ![image](https://user-images.githubusercontent.com/63558665/120859884-d6ffa380-c552-11eb-9d46-d4fc35942014.png)
    
    * NAS
        * “Controller” network that learns to design a good network architecture (output a string corresponding to network design)
        * Sample an architecture from search space
        * Train the architecture to get a “reward” R corresponding to accuracy
        * Compute gradient of sample probability, and scale by R to perform controller parameter update (i.e. increase likelihood of good architecture being sampled,decrease likelihood of bad architecture)
    
* Summary:
    * AlexNet showed that you can use CNNs to train Computer Vision models.
    * ZFNet, VGG shows that bigger networks work better
    * GoogLeNet is one of the first to focus on efficiency using 1x1 bottleneck convolutions and global avg pool instead of FC layers
    * ResNet showed us how to train extremely deep networks
        * Limited only by GPU & memory!
        * Showed diminishing returns as networks got bigger After ResNet: CNNs were better than the human metric and focus shifted to Efficient networks:
        * Lots of tiny networks aimed at mobile devices: MobileNet, ShuffleNet
    * Neural Architecture Search can now automate architecture design
    * Many popular architectures available in model zoos
    * ResNet and SENet currently good defaults to use
    * Networks have gotten increasingly deep over time
    * Many other aspects of network architectures are also continuously being investigated and improved


# RNN: Recurrent Neural Networks
* Application: 
   * image caption: one to many
   * action prediction: many to one
   * Video Captioning: many to many
   * Video classification on frame level:many to many
   * Visual Question Answering (VQA)
   * Visual Question Answering: RNNs with Attention
   * Visual Dialog: Conversations about images
   * Visual Language Navigation: Go to the living room
   * Visual Question Answering: Dataset Bias
   
* Why not existing convnet?

  Variable sequence length inputs and outputs! --> input and output are also variable

* RNN: RNNs have an “internal state” that is updated as a sequence is processed
    Update internal state and then update output
    1. hidden state update: h_t=fw(h_t_1,x)

        ![image](https://user-images.githubusercontent.com/63558665/120261845-bdb2da80-c266-11eb-9248-3ec140e39bc2.png)

    2. Output generation: y_t=f(h_t)

         ![image](https://user-images.githubusercontent.com/63558665/120261951-f488f080-c266-11eb-806e-b05f191475a8.png)

Notice: the same function and the same set of parameters are used at every time step.

sequence to sequence :many to one (encoder)-->one to many (decoder)

   ![image](https://user-images.githubusercontent.com/63558665/120262874-96f5a380-c268-11eb-8b68-aa3d1cd1e00a.png)

  Re-use the same weight matrix at every time-step

   
   3. Backpropagation through time
       * Forward through entire sequence to compute loss, then backward through entire sequence to compute gradient
       * Run forward and backward through chunks of the sequence instead of whole sequence. Carry hidden states forward in time forever, but only backpropagate for some smaller number of steps.
   
          ![image](https://user-images.githubusercontent.com/63558665/120123592-b2d04b00-c17d-11eb-89ce-c0f7fd4d0b66.png)
   
   4. RNN tradeoffs
      * Advantages:
          * Can process any length input
          * Computation for step t can (in theory) use information from many steps back
          * Model size doesn’t increase for longer input
          * Same weights applied on every timestep, so there is symmetry in how inputs are processed.
      * Disadvantages:
          * Recurrent computation is slow
          * In practice, difficult to access information from many steps back
    
   5. Multiple layers RNN:

       ![image](https://user-images.githubusercontent.com/63558665/120263949-c0afca00-c26a-11eb-95e1-acd51036e1f0.png)

   6. LSTM: RNN-->vanishing gradient
      
      ![image](https://user-images.githubusercontent.com/63558665/120264441-cfe34780-c26b-11eb-815c-996ecdef1ff7.png)
      
      ![image](https://user-images.githubusercontent.com/63558665/120264411-bcd07780-c26b-11eb-99bc-cd7d87bb3ce6.png)

      ![image](https://user-images.githubusercontent.com/63558665/120264615-28b2e000-c26c-11eb-8f5e-f79b7b715b16.png)

      ![image](https://user-images.githubusercontent.com/63558665/120264629-2ea8c100-c26c-11eb-9040-6bcc8dac51a5.png)
      
      * Backpropagation from ct to ct-1 only elementwise multiplication by f, no matrix multiply by W--LSTM
      
      * Notice that the gradient contains the f gate’s vector of activations
          * allows better control of gradients values, using suitable parameter updates of the forget gate.
      * Also notice that are added through the f, i, g, and o gates
          * better balancing of gradient values
      * The LSTM architecture makes it easier for the RNN to preserve information over many timesteps
          * e.g. if the f = 1 and the i = 0, then the information of that cell is preserved
          indefinitely.
          * By contrast, it’s harder for vanilla RNN to learn a recurrent weight matrix Wh that preserves info in hidden state
      * LSTM doesn’t guarantee that there is no vanishing/exploding gradient, but it does provide an easier way for the model to learn long-distance dependencies         * Uninterrupt gradient/Use variants like GRU if you want faster compute and less parameters
      * Common to use LSTM or GRU: their additive interactions improve gradient flow  Backward flow of gradients in RNN can explode or vanish. Exploding is controlled with gradient clipping. Vanishing is controlled with additive interactions (LSTM)
      * Better/simpler architectures are a hot topic of current research, as well as new paradigms for reasoning over sequences
  
  7. GRU
       
       ![image](https://user-images.githubusercontent.com/63558665/120875413-bbf45a00-c579-11eb-8302-333e34f40afd.png)

   8. RNN architecture:
      * many to many
      
          ![image](https://user-images.githubusercontent.com/63558665/120123662-18243c00-c17e-11eb-953e-df597f42170d.png)

      * many-->one: : Encode input sequence in a single vector

          ![image](https://user-images.githubusercontent.com/63558665/120123659-10fd2e00-c17e-11eb-8335-175e50c6648a.png)

      * one-->many: : Produce output sequence from single input vector

          ![image](https://user-images.githubusercontent.com/63558665/120123650-05aa0280-c17e-11eb-94f5-e15ce557d91d.png)

          ![image](https://user-images.githubusercontent.com/63558665/120123743-a7315400-c17e-11eb-9814-85b0e3a5061b.png)
   9. Summay
   * LSTM were a good default choice until this year
   * Use variants like GRU if you want faster compute and less parameters
   * Use transformers (next lecture) as they are dominating NLP models
   * almost everyday there is a new vision transformer model
   * RNNs allow a lot of flexibility in architecture design
   * Vanilla RNNs are simple but don’t work very well
   * Common to use LSTM or GRU: their additive interactions improve gradient flow
   * Backward flow of gradients in RNN can explode or vanish. Exploding is controlled with gradient clipping. Vanishing is controlled with additive interactions (LSTM)
   * Better/simpler architectures are a hot topic of current research as well as new paradigms for reasoning over sequences

# Attention and Transformers
* Attension: NLP and image caption
    * Problem: Model needs to encode everything it wants to say within c (context vector)-->New context vector at every time step +Each context vector will attend to different image regions
    * steps:
        * Compute alignments H x W scores (scalars)
        
           ![image](https://user-images.githubusercontent.com/63558665/120876256-bf89e000-c57d-11eb-80fd-a085a3352bb6.png)
        
        * Attendtion: Normalize to get attention weights-->f_att(.) is an MLP
        * Compute context vector:
         
            ![image](https://user-images.githubusercontent.com/63558665/120876603-b0a42d00-c57f-11eb-8d03-1b0250819c59.png)
            
         * Each timestep of decoder uses a different context vector that looks at different parts of the input image
         * This entire process is differentiable
         * model chooses its own attention weights. No attention supervision is required
    * Methods:
    * General attention layer: mul+add--> Recall that the query vector was a function of the input vectors
         * the input vectors are used for both the alignment as well as the attention calculations.
         
            ![image](https://user-images.githubusercontent.com/63558665/120877089-8f910b80-c582-11eb-9402-0e477fbfe20b.png)
   
   * self-attention:
       * calculate the query vector from the input vectors, therefore, defining a "self_attention layer"
       * Instead, query vectors are calculated using a FC layer and No input query vectors anymore
        
          ![image](https://user-images.githubusercontent.com/63558665/120877263-90766d00-c583-11eb-8c45-1903dbb41722.png)

    Problem: how can we encode ordered sequences like language or spatially ordered image features?
    
    * Positional encoding:
        * Concatenate special positional encoding pj to each input vector xj
        * We use a function pos: N →Rd to process the position j of the vector into a d-dimensional vector
            * It should output a unique encoding for each time-step (word’s position in a sentence)
            * Distance between any two time-steps should be consistent across sentences with different lengths.
            * Our model should generalize to longer sentences without any efforts. Its values should be bounded.
            * It must be deterministic
            * opitons
         * options for pos(.):
             * Learn a lookup table:
                  * Learn parameters to use for pos(t) for t ε [0, T]
                  * Lookup table contains T x d parameters.
              * Design a fixed function with the desiderata
          
            ![image](https://user-images.githubusercontent.com/63558665/120877415-5c4f7c00-c584-11eb-9ad8-ef8481e2ad51.png)
    
    * Masked self-attention layer
        * Prevent vectors from looking at future vectors.
        * Manually set alignment scores to -infinity
          
          ![image](https://user-images.githubusercontent.com/63558665/120877443-8f920b00-c584-11eb-9375-7e819ad1179e.png)
   
    * Multi self-attention layer
    
         ![image](https://user-images.githubusercontent.com/63558665/120877470-b3555100-c584-11eb-9b69-8f49e3fbe500.png)
         
     * General attention vs. self-attention
        
        ![image](https://user-images.githubusercontent.com/63558665/120877503-dda70e80-c584-11eb-9bff-7c05c2abc46f.png)

* Transformer:No recurrence,Perhaps we don't need convolutions at all-->Transformers from pixels to language

    ![image](https://user-images.githubusercontent.com/63558665/120877591-5f973780-c585-11eb-84e5-5db1867d7f12.png)
    
    encoder block:
    
    ![image](https://user-images.githubusercontent.com/63558665/120877608-76d62500-c585-11eb-8e00-d7bb5f35153f.png)
    
    decoder block:
    
    ![image](https://user-images.githubusercontent.com/63558665/120877620-89505e80-c585-11eb-96f7-abfa5a3d979a.png)

* RNNs vs. Transformers
    * RNNs
        * LSTMs work reasonably well for long sequences.
        * Expects an ordered sequences of inputs
        * Sequential computation: subsequent hidden states can only be computed after the previous ones are done.

    * Transformers:
        * Good at long sequences. Each attention calculation looks at all inputs.
        * Can operate over unordered sets or ordered sequences with positional encodings
        * Parallel computation: All alignment and attention scores for all inputs can be done in parallel.
        * Requires a lot of memory: N x M alignment and attention scalers need to be calculated and stored for a single self-attention head. (but GPUs are getting bigger and better)

* Summary:
    * Adding attention to RNNs allows them to "attend" to different parts of the input at every time step
    * The general attention layer is a new type of layer that can be used to design new neural network architectures
    * Transformers are a type of layer that uses self-attention and layer norm.
        * It is highly scalable and highly parallelizable
        * Faster training, larger models, better performance across vision and language tasks
        * They are quickly replacing RNNs, LSTMs, and may even replace convolutions.

# Gans

* Supervised learning: learn a funtion to map x-->y
     * classification,regression, object detection, semantic segmentation, image captioning
* Unsupervised learning:Learn some underlying hidden structure of the data
     * clustering, dimensionality reduction,feature learning, density estimation

* Generative Models:Given training data, generate new samples from same distribution,Learn pmodel(x) that approximates pdata(x).Sampling new x from pmodel(x)
   (Pixel distribution)
   
* why genrative model?
    * Realistic samples for artwork, super-resolution, colorization, etc
    * Learn useful features for downstream tasks such as classification.
    * Getting insights from high-dimensional data (physics, medical imaging, etc.)
    * Modeling physical world for simulation and planning (robotics and reinforcement learning applications)

     ![image](https://user-images.githubusercontent.com/63558665/120584659-621b5500-c3fe-11eb-8b5b-6a16248291db.png)

* PixelRNN and PixelCNN
   * FVBN--RNN (produce each possible pixel)
     * Complex distribution over pixel values

   * pixelRNN:Generate image pixels starting from corner and Dependency on previous pixels modeled using an RNN (LSTM)
       
        
        * Drawback: sequential generation is slow in both training and inference!
        
            ![image](https://user-images.githubusercontent.com/63558665/120584918-d0f8ae00-c3fe-11eb-9087-7c408419bff2.png)

        
   * PixelCNN:Still generate image pixels starting from corner,Dependency on previous pixels now modeled using a CNN over context region
         
        * Training is faster than PixelRNN (can parallelize convolutions since context region values known from training images)
        * Generation is still slow:For a 32x32 image, we need to do forward passes of the network 1024 times for a single image
            
            ![image](https://user-images.githubusercontent.com/63558665/120585029-08675a80-c3ff-11eb-9aa1-f7c02331131e.png)


      ![image](https://user-images.githubusercontent.com/63558665/120585049-11f0c280-c3ff-11eb-88db-0a3ca6f3496f.png)

* VAE:Variational Autoencoders--Probabilistic spin on autoencoders - will let us sample from the model to generate data!
 
    ![image](https://user-images.githubusercontent.com/63558665/120585160-45cbe800-c3ff-11eb-941e-21711c12f057.png)
 
   * No dependencies among pixels, can generate all pixels at the same time!
   * Cannot optimize directly, derive and optimize lower bound on likelihood instead
   
   * Autoencoder:Unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data 

        ![image](https://user-images.githubusercontent.com/63558665/120878117-59ef2100-c588-11eb-8e0b-829c4e25f797.png)

        * z smaller than x
        * Why dimensionality reduction?-->Want features to capture meaningful factors of variation in data
        * How to learn feature?--> trained autoencoder that features can be used to reconstrcut original data with L2 loss function 
        * Train encoder--> z-->decoder-->extract feature :Transfer from large, unlabeled dataset to small, labeled dataset.
        * After training, throw away decoder
        * The same as transfer learning--> using encoder features to fine tune encoder jointly with classifier
        * But we can’t generate new images from an autoencoder because we don’t know the space of z
      
      
     ![image](https://user-images.githubusercontent.com/63558665/120585413-afe48d00-c3ff-11eb-801d-54ded05cd109.png)

     Choose prior p(z) to be simple, e.g.Gaussian. Reasonable for latent attributes, e.g. pose, how much smile

     ![image](https://user-images.githubusercontent.com/63558665/120585901-92fc8980-c400-11eb-823d-40ab9b13d120.png)

    
    * Variational inference is to approximate the unknown posterior distribution from only the observed data x rather than latent z
       
       ![image](https://user-images.githubusercontent.com/63558665/121260154-f6fad400-c87e-11eb-976f-2d6697fb5b56.png)
       
       ![image](https://user-images.githubusercontent.com/63558665/121260440-58bb3e00-c87f-11eb-84b3-e0080642bdc9.png)
       
       ![image](https://user-images.githubusercontent.com/63558665/121260544-79839380-c87f-11eb-93c9-c367f86fdede.png)


* Why dimensionality reduction?
  
  Want features to capture meaningful factors of variation in data

* GANS: not explicit dense function
  
  Sample from a simple distribution we can easily sample from, e.g. random noise.Learn transformation to training distribution.
  
  But we don’t know which sample z maps to which training image -> can’t learn by reconstructing training images
 
  Solution: Use a discriminator network to tell whether the generate image is within data distribution (“real”) or not
  
  Discriminator network: try to distinguish between real and fake images
  
  Generator network: try to fool the discriminator by generating real-looking images. just like minmax game
  
  Discriminator (θd) wants to maximize objective such that D(x) is close to 1 (real) and D(G(z)) is close to 0 (fake)
  
  Generator (θg) wants to maximize likelihood of discriminator being wrong.
  
  ![image](https://user-images.githubusercontent.com/63558665/120586384-714fd200-c401-11eb-8590-c944d7c73f4d.png)
  
  ![image](https://user-images.githubusercontent.com/63558665/120586592-cdb2f180-c401-11eb-88db-4964a8c29e15.png)
  
  ![image](https://user-images.githubusercontent.com/63558665/120586621-da374a00-c401-11eb-920b-5c5201c524cd.png)
  
  after training, using gans to generate new images
* How to choose k?
  
  k=1 is stable
  
*  Generative Adversarial Nets: Convolutional Architectures

   Generator is an upsampling network with fractionally-strided convolution. Discriminator is a convolutional network
   
   * Replace any pooling layers with stried convolutions (descriminator) and fractional_strided convolution (generator)
   * Use BatchNorm in the both generator and descriminator
   * Remove fully connected hidden layers for deeper architecture
   * Use ReLU activation in generator for all layers expect for output, which uses Tanh
   * Use Leaky ReLu activation in descriminator for all layers
   
*  Generative Adversarial Nets: Interpretable Vector Math
*  Gans application:
    *  Target domain transfer
    *  Image synthesis
    *  Label to stress scene
    *  Aerial Mapping
    *  Day to night
    *  BW to color images
    *  edges to photo
    *  high resolution images
    *  Image repair and paiting
    *  scene graphs to gans: specify what kind of image you want to generate
    *  label generation
* Summary:
    * Don’t work with an explicit density function
    * Unstable to train Gans (Wasserstein GAN, LSGAN, many others)
    * Conditional GANs, GANs for all kinds of applications

   Pretext task learning good feature extractor from self-supervised  tasks--> Attach a shallow network on feature extractor; train shallow network on the target task with   small amount of labeled data
   
* Pretext task:predict rotation/predict missing pixel/reconstruct missing pixel (cut image and train model with missing image)
  
  * colorization: color image--> gray image--> training model--> compare color image with predict image
  
* Visualization:
   
  * PCA/T-SNE
  * faster style transfer
  * Many methods for understanding CNN representations
  * Activations: Nearest neighbors, Dimensionality reduction, maximal patches, occlusion
  * Gradients: Saliency maps, class visualization, fooling images, feature inversion 
  * Fun: DeepDream, Style Transfer.

# Detection and Segmentation:
* semantic segmentation:
    * Semantic Segmentation Idea: Sliding Window--> Impossible to classify without context+ very expensive
    * Semantic Segmentation Idea: Fully Convolutional
    * Convultion: convolutions at original image resolution will be very expensive--> downsampling  and upsampling inside network
    * downsapling: pooling/strideed convolution 
    * Upsampling:Nearest Neighbor, bedof nails,max unpooling using the position from pooling layer, learnable upsampling,Transposed convolution

* Object detection:
    * two stages and single stage object detection
    * Treat localization as a regression problem!-l2 loss
    * Softmax loss+L2 loss=multiple task loss
    * Each image needs a different number of outputs!
    * Problem: Need to apply CNN to huge number of locations, scales, and aspect ratios, very computationally expensive!

    * RCNN: Region Proposals:selective Search gives 2000 region proposals in a few seconds on CPU-->Very slow! Need to do ~2kindependent forward passes for each image!
    * Fast-RCNN: convt-->region proposals-->crop+resize features-->CNN -->(linear softmax)+(linear)
    * Running time domain by region proposals
    * Faster-RCNN:Make CNN to proposal(RPN)
    
      ![image](https://user-images.githubusercontent.com/63558665/121267647-848ff100-c88a-11eb-9117-e313b70f53c8.png)
    * single-stage:YOLO / SSD / RetinaNet-->7 x 7 x (5 * B + C)

* Instance segmentaion:
* Mask R-CNN: add small mask network that operates on each ROI and predicts a 28x28 binary mask--> predict a mask for each class


* R-CNN: Bounding boxes are proposed by the “selective search” algorithm, each of which is stretched and features are extracted via a deep convolutional neural network, such as AlexNet, before a final set of object classifications are made with linear SVMs.

  ![image](https://user-images.githubusercontent.com/63558665/121413858-1bfd4e80-c934-11eb-907c-5f30ce9a269c.png)
  
    * It still takes a huge amount of time to train the network as you would have to classify 2000 region proposals per image.
    * It cannot be implemented real time as it takes around 47 seconds for each test image.
    * The selective search algorithm is a fixed algorithm. Therefore, no learning is happening at that stage. This could lead to the generation of bad candidate region proposals.
    
* Fast R-CNN: The approach is similar to the R-CNN algorithm. But, instead of feeding the region proposals to the CNN, we feed the input image to the CNN to generate a convolutional feature map. From the convolutional feature map, we identify the region of proposals and warp them into squares and by using a RoI pooling layer we reshape them into a fixed size so that it can be fed into a fully connected layer. From the RoI feature vector, we use a softmax layer to predict the class of the proposed region and also the offset values for the bounding box.
  
  ![image](https://user-images.githubusercontent.com/63558665/121414350-8910e400-c934-11eb-9168-7d42120a6dd3.png)

* Faster R-CNN: Both of the above algorithms(R-CNN & Fast R-CNN) uses selective search to find out the region proposals. Selective search is a slow and time-consuming process affecting the performance of the network. The image is provided as an input to a convolutional network which provides a convolutional feature map. Instead of using selective search algorithm on the feature map to identify the region proposals, a separate network is used to predict the region proposals. The predicted region proposals are then reshaped using a RoI pooling layer which is then used to classify the image within the proposed region and predict the offset values for the bounding boxes.

* Yolo: All of the previous object detection algorithms use regions to localize the object within the image. The network does not look at the complete image. Instead, parts of the image which have high probabilities of containing the object. YOLO or You Only Look Once is an object detection algorithm much different from the region based algorithms seen above. a single convolutional network predicts the bounding boxes and the class probabilities for these boxes.
* Mask R-CNN: Extension of Faster R-CNN that adds an output model for predicting a mask for each detected object.



