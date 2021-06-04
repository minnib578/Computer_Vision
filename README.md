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

* challenges for image classification?
   * viewpoint variation: all pixels change when the camera moves
   * background cluster: objects are similar to the background (color or texture)
   * illumination: (too dark or too light)-->data driven x
   * occlusion: hiden by other objects
   * deformation: different shape/pose of the same objects
   * intraclass variation:differnt types of the same object
   * scale variation:Visual classes often exhibit variation in their siz
 
 * data-driven methods: k and distance choised, evaluation methods
   * Nearest Neighbor classifier:The nearest neighbor classifier will take a test image, compare it to every single one of the training images, and predict the label of the closest training image.One of the simplest possibilities is to compare the images pixel by pixel and add up all the differences. In other words, given two images and representing them as vectors I1,I2 , a reasonable choice for comparing them might be the L1 distance: (Manhattan) distance. vectoriation calculation
         
        ![image](https://user-images.githubusercontent.com/63558665/120114755-a7b1f680-c14e-11eb-9122-f4c75d58a0b4.png)
     
     L2 distance:(Euclidean) distance
     
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
      
* SVM loss: 错误分类更高的分数，正确分类 is 0, l1 and l2 SVM loss:difference in score between correct and incorrect class

     ![image](https://user-images.githubusercontent.com/63558665/120117034-257aff80-c159-11eb-9ced-793c3a178ba3.png)

     ![image](https://user-images.githubusercontent.com/63558665/120116958-d6cd6580-c158-11eb-9dde-6916a3e0287c.png)
       
     questions:
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
                                 
* softmax: interpret classifier score into probability, probability sum to 1,Choose weights to maximize the
likelihood of the observed data
            
   ![image](https://user-images.githubusercontent.com/63558665/120117288-763f2800-c15a-11eb-9c20-b11ab39c2090.png)
     
   ![image](https://user-images.githubusercontent.com/63558665/120117410-0f6e3e80-c15b-11eb-92cc-1a67f803002a.png)
   
   * Question:
        * What is the min/max possible softmax loss Li?
        * At initialization all sj will be approximately equal; what is the softmax loss Li, assuming C classes?
   
   ![image](https://user-images.githubusercontent.com/63558665/120714559-5cbb1a80-c491-11eb-93b9-ddc8fbed5a4b.png)

* optimization: how to find the best w?
     * strategy 1: random search
     * startegy 2: follow the slope-->gradient descent

Three loss function: linear loss, SVM loss, softmax and data loss_reguralization     
        
# Neural network-multiple layers neural network
linear classifier is not useful and can only draw  linear decision boundaries-->featuere transformation: f(x, y) = (r(x, y), θ(x, y))

   ![image](https://user-images.githubusercontent.com/63558665/120117682-54df3b80-c15c-11eb-9cbc-26906f99b548.png)

   ![image](https://user-images.githubusercontent.com/63558665/120117694-6294c100-c15c-11eb-8fc0-13958b764a2d.png)
  
 why activation?--> W2*W1=W3  end up with linear classifier again!
   
   ![image](https://user-images.githubusercontent.com/63558665/120117745-ab4c7a00-c15c-11eb-8a22-2ff3352efbe1.png)
   
   ![image](https://user-images.githubusercontent.com/63558665/120117758-b6070f00-c15c-11eb-8bf9-efc7eb33bcd4.png)
   
Multiple layer Neural network:
   
   ![image](https://user-images.githubusercontent.com/63558665/120117774-ca4b0c00-c15c-11eb-8930-57187382e583.png)

derive delta_w L on paper?
* Very tedious: Lots of matrix calculus, need lots of paper
* What if we want to change loss? E.g. use softmax instead of SVM? Need to re-derive from scratch
* Not feasible for very complex models!
--> Backpropagation+ computational graph-->chain rule

![image](https://user-images.githubusercontent.com/63558665/120716932-b8d36e00-c494-11eb-8a9d-c4a1c07826cd.png)

- (Fully-connected) Neural Networks are stacks of linear functions and nonlinear activation functions; they have much more representational
power than linear classifiers
- backpropagation = recursive application of the chain rule along a computational graph to compute the gradients of all inputs/parameters/intermediates
- implementations maintain a graph structure, where the nodes implement the forward() / backward() API
- forward: compute result of an operation and save any intermediates needed for gradient computation in memory
- backward: apply the chain rule to compute he gradient of the loss function with respect to the inputs


# linear classifier--> multiple layers neural network-->covolution network
   
   ![image](https://user-images.githubusercontent.com/63558665/120119457-cc659880-c165-11eb-8f05-5f72a60b3440.png)
   ![image](https://user-images.githubusercontent.com/63558665/120119468-d4253d00-c165-11eb-9a37-22f349719691.png)
   ![image](https://user-images.githubusercontent.com/63558665/120119656-e05dca00-c166-11eb-9929-155a02c62103.png)
   low-level features-->high level features
   convolution architecture: convolution layer-->ReLU-->pooling layer-->fully connected layer
   
   ![image](https://user-images.githubusercontent.com/63558665/120119722-53ffd700-c167-11eb-8176-f08cdaa0f499.png)
   
   ![image](https://user-images.githubusercontent.com/63558665/120119757-87426600-c167-11eb-9981-ebc72605b8b7.png)
   In general, common to see CONV layers with stride 1, filters of size FxF, and zero-padding with (F-1)/2. (will preserve size spatially)
   
   ![image](https://user-images.githubusercontent.com/63558665/120119883-1cddf580-c168-11eb-9e3f-3c14d6c66890.png)
   
   pooling:Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting.
   ![image](https://user-images.githubusercontent.com/63558665/120119992-bf967400-c168-11eb-8adb-b38fb349403b.png)
   padding:control the spatial size of the output volumes,most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same
   
# Convolutional Neural Networks and training
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
 
* pooling layer: makes the representations smaller and more manageable, downsample,operates over each activation map independently
        * Maxpooling
        * global pooling
        * average pooling
             
number of parameters is 0
* tips:
        * Trend towards smaller filters and deeper architectures
        * Trend towards getting rid of POOL/FC layers (just CONV)
        * conv-->ReLu-->pool-->softmax
        * 
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
Using ReLu be careful with learning rate--  Don’t use sigmoid or tanh---Try out Leaky ReLU / Maxout / ELU / SELU--To squeeze out some marginal gains
      
   2. preprocessing:consider what happends when the input to a neura is always positive--->zigzag path
        * zero-mean data--> visualize data with PCA and Whitening
            * substract the mean image(AlexNet)
            * substract per-channel mean (VGGNet)
            * substract per-channel mean and divide by per-channel std (ResNet)
        * normalization:Before normalization: classification loss very sensitive to changes in weight matrix; hard to optimize. After normalization: less sensitive to small changes in weights; easier to optimize
      
   3. weight initialization:
        * small random numbers:(gaussian with zero mean and 1e-2 standard deviation)-->work with small network, but no deep network---> vanishing gradient-->no learning
        * “Xavier” Initialization: Activations are nicely scaled for all layers-->ReLU Activations collapse to zero again-->
                     
             ![image](https://user-images.githubusercontent.com/63558665/120121487-c4135a80-c171-11eb-8561-d06a4b12ef89.png)
             
         * Kaiming/MSRA initilization
         
              ![image](https://user-images.githubusercontent.com/63558665/120121622-70554100-c172-11eb-992e-930fa4462046.png)
              
        depending on differnt activation function using differnt weight inilization
      
   4. regularization: zero-mean unit-variance activationa and improve single model performance
        * batchnormalization: During testing batchnorm becomes a linear operator! Can be fused with the previous fully-connected or conv layer
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
            
              ![image](https://user-images.githubusercontent.com/63558665/120122407-882ec400-c176-11eb-9a3a-d2856af4d60a.png)
           
        * dropout:In each forward pass, randomly set some neurons to zero Probability of dropping is a hyperparameter; 0.5 is common
            * Dropout is training a large ensemble of models (that share parameters).
            * Each binary mask is one model
            * drop in train and scale in test
        * data augmentation:
            * Random crops and scales
            * Color Jitter
            * translation/rotation/stretching/shearing/lens distortions/Cutout/random crop/mixup
            * add random noise
       * Dropconnect: Drop connections between neurons (set weights to 0)
       * Fractional Pooling:Use randomized pooling regions
       * stochastic depth: Skip some layers in the network
    
   5. Summary:
        * Consider dropout for large-->fully-connected layers
        * Batch normalization and data augmentation almost always a good idea
        * Try cutout and mixup especially for small classification datasets
        * Training Dynamics  
        * babysitting the learning process
             * Learning rate decays over time/cosine/linear
                            ![image](https://user-images.githubusercontent.com/63558665/120122292-e1e2be80-c175-11eb-8ddc-0cca78a8a650.png)
                            
        * Adam is a good default choice in many cases; it often works ok even with constant learning rate.SGD+Momentum can outperform Adam but mayrequire more tuning of LR and schedule-->Try cosine schedule, very few hyperparameters! If you can afford to do full batch updates then try out-->L-BFGS (and don’t forget to disable all sources of noise)
        * parameters update, hyperparameter optimization
           
             ![image](https://user-images.githubusercontent.com/63558665/120122603-cb3d6700-c177-11eb-9539-f6b9c98211b3.png)
           
        * early stop
        
* Evaluation：
     * model ensembles: Train multiple independent models,At test time average their results
     * test-time augmentation
     * transfer learning
         * Lower learning rate when finetuning; 1/10 of original LR is good starting point
         * With bigger dataset, train more layers

              ![image](https://user-images.githubusercontent.com/63558665/120121953-13f32100-c174-11eb-9ed4-ad345d1f13b6.png)
                 
              ![image](https://user-images.githubusercontent.com/63558665/120122009-3f760b80-c174-11eb-8ced-bc8ab1c32ab5.png)
             
          * They also find that collecting more data is better than finetuning on a related task
 
            ![image](https://user-images.githubusercontent.com/63558665/120122057-8d8b0f00-c174-11eb-8de9-fc32da67cbd0.png)

            ![image](https://user-images.githubusercontent.com/63558665/120122614-e3ad8180-c177-11eb-8f22-5cca118038b5.png)

### 6) CNN architecture
* LeNet-5

![image](https://user-images.githubusercontent.com/63558665/120122674-2ec79480-c178-11eb-9ddf-e0fea9dcef24.png)

* AlexNet:
![image](https://user-images.githubusercontent.com/63558665/120122712-6df5e580-c178-11eb-8ce1-61df9485e55f.png)

![image](https://user-images.githubusercontent.com/63558665/120122711-68000480-c178-11eb-8454-f9cfe86285db.png)

* ZFNet:
   Alexnet: CONV1: change from (11x11 stride 4) to (7x7 stride 2)
   CONV3,4,5: instead of 384, 384, 256 filters use 512, 1024, 512

* VGGNet:Deeper network
8 layers AlexNet-->16 layers-19 layers(VGG16Net)
Only 3x3 CONV stride 1, pad 1 and 2x2 MAX POOL stride 2
Why use smaller filters? (3x3 conv) ?
Stack of three 3x3 conv (stride 1) layers has same effective receptive field as one 7x7 conv layer.But deeper, more non-linearities.fewer parameters

![image](https://user-images.githubusercontent.com/63558665/120122818-46534d00-c179-11eb-89eb-de3a5eff18ab.png)

* GoogleNet: deeper but more computational
22 layers--Efficient “Inception” module--No FC layers

![image](https://user-images.githubusercontent.com/63558665/120122953-21130e80-c17a-11eb-9a0f-367289289f52.png)

Computational complexity/Pooling layer also preserves feature depth, which means total depth after concatenation can only grow at every layer
---> bottleneck that 1x1 convolution to feature channel size
       
![image](https://user-images.githubusercontent.com/63558665/120123059-c4fcba00-c17a-11eb-8796-d1cfe67033b6.png)
![image](https://user-images.githubusercontent.com/63558665/120123098-eeb5e100-c17a-11eb-8eb3-dbb96e622ce7.png)

after the last convolutional layer, a global average pooling layer is used that spatially averages across each feature map, before final FC layer. No longer multiple expensive FC layers!

![image](https://user-images.githubusercontent.com/63558665/120123142-2fadf580-c17b-11eb-99fb-3cba5243e6db.png)

* ResNet:
![image](https://user-images.githubusercontent.com/63558665/120123160-43f1f280-c17b-11eb-8a25-a96833914993.png)
The deeper model performs worse, but it’s not caused by overfitting!
deeper models are harder to optimize--> A solution by construction is copying the learned layers from the shallower model and setting additional layers to identity mapping.

![image](https://user-images.githubusercontent.com/63558665/120123268-dd210900-c17b-11eb-8d03-1cb5d374593f.png)

### 7) RNN: 
* application: 
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
1. hidden state update: h_t=fw(h_t_1,x)

![image](https://user-images.githubusercontent.com/63558665/120261845-bdb2da80-c266-11eb-9248-3ec140e39bc2.png)

2.Output generation:y_t=f(h_t)

![image](https://user-images.githubusercontent.com/63558665/120261951-f488f080-c266-11eb-806e-b05f191475a8.png)

Notice: the same function and the same set of parameters are used at every time step.

sequence to sequence :many to one (encoder)-->one to many (decoder)

![image](https://user-images.githubusercontent.com/63558665/120262874-96f5a380-c268-11eb-8b68-aa3d1cd1e00a.png)

Carry hidden states forward in time forever, but only backpropagate for some smaller number of steps.Re-use the same weight matrix at every time-step
![image](https://user-images.githubusercontent.com/63558665/120123592-b2d04b00-c17d-11eb-89ce-c0f7fd4d0b66.png)

RNN Advantages:
- Can process any length input
- Computation for step t can (in theory) use information from many steps
back
- Model size doesn’t increase for longer input
- Same weights applied on every timestep, so there is symmetry in how
inputs are processed.
RNN Disadvantages:
- Recurrent computation is slow
- In practice, difficult to access information from many steps back

* Multiple layers RNN:

![image](https://user-images.githubusercontent.com/63558665/120263949-c0afca00-c26a-11eb-95e1-acd51036e1f0.png)

* LSTM: RNN-->vanishing gradient

![image](https://user-images.githubusercontent.com/63558665/120264411-bcd07780-c26b-11eb-99bc-cd7d87bb3ce6.png)

![image](https://user-images.githubusercontent.com/63558665/120264441-cfe34780-c26b-11eb-815c-996ecdef1ff7.png)

![image](https://user-images.githubusercontent.com/63558665/120264615-28b2e000-c26c-11eb-8f5e-f79b7b715b16.png)
![image](https://user-images.githubusercontent.com/63558665/120264629-2ea8c100-c26c-11eb-9040-6bcc8dac51a5.png)
Notice that the gradient contains the f gate’s vector of activations
- allows better control of gradients values, using suitable parameter updates of the
forget gate.
Also notice that are added through the f, i, g, and o gates
- better balancing of gradient values
The LSTM architecture makes it easier for the RNN to preserve information over many timesteps
- e.g. if the f = 1 and the i = 0, then the information of that cell is preserved
indefinitely.
- By contrast, it’s harder for vanilla RNN to learn a recurrent weight matrix Wh that preserves info in hidden state
LSTM doesn’t guarantee that there is no vanishing/exploding gradient, but it does provide an easier way for the model to learn long-distance dependencies
Uninterrupt gradient/Use variants like GRU if you want faster compute and less parameters
Common to use LSTM or GRU: their additive interactions improve gradient flow  Backward flow of gradients in RNN can explode or vanish. Exploding is controlled with gradient clipping. Vanishing is controlled with additive interactions (LSTM)
Better/simpler architectures are a hot topic of current research, as well as new paradigms for reasoning over sequences
![image](https://user-images.githubusercontent.com/63558665/120123662-18243c00-c17e-11eb-953e-df597f42170d.png)

many-->one: : Encode input sequence in a single vector

![image](https://user-images.githubusercontent.com/63558665/120123659-10fd2e00-c17e-11eb-8335-175e50c6648a.png)

one-->many: : Produce output sequence from single input vector

![image](https://user-images.githubusercontent.com/63558665/120123650-05aa0280-c17e-11eb-94f5-e15ce557d91d.png)
![image](https://user-images.githubusercontent.com/63558665/120123743-a7315400-c17e-11eb-9814-85b0e3a5061b.png)

###9） supervised learning vs. unsupervised learning

* supervised learning: learn a funtion to map x-->y
     * classification,regression, object detection, semantic segmentation, image captioning
* unsupervised learning:Learn some underlying hidden structure of the data
     * clustering, dimensionality reduction,feature learning, density estimation

* Generative Models:Given training data, generate new samples from same distribution,Learn pmodel(x) that approximates pdata(x).Sampling new x from pmodel(x)

* why genrative model?
    * Realistic samples for artwork, super-resolution, colorization, etc
    * Learn useful features for downstream tasks such as classification.
    * Getting insights from high-dimensional data (physics, medical imaging, etc.)
    * Modeling physical world for simulation and planning (robotics and reinforcement learning applications)

![image](https://user-images.githubusercontent.com/63558665/120584659-621b5500-c3fe-11eb-8b5b-6a16248291db.png)

* pixelRNN and PixelCNN
 FVBN
 
 ![image](https://user-images.githubusercontent.com/63558665/120584795-98f16b00-c3fe-11eb-9f6b-ee5405fb94f6.png)
 
   * pixelRNN:Generate image pixels starting from corner and Dependency on previous pixels modeled using an RNN (LSTM)
       
       ![image](https://user-images.githubusercontent.com/63558665/120584918-d0f8ae00-c3fe-11eb-9087-7c408419bff2.png)

        Drawback: sequential generation is slow in both training and inference!
   * PixelCNN:Still generate image pixels starting from corner,Dependency on previous pixels now modeled using a CNN over context region
         
         ![image](https://user-images.githubusercontent.com/63558665/120585029-08675a80-c3ff-11eb-9aa1-f7c02331131e.png)

      Training is faster than PixelRNN (can parallelize convolutions since context region values known from training images)
      Generation is still slow: For a 32x32 image, we need to do forward passes of the network 1024 times for a single image

![image](https://user-images.githubusercontent.com/63558665/120585049-11f0c280-c3ff-11eb-88db-0a3ca6f3496f.png)
* VAE:Variational Autoencoders
 
 ![image](https://user-images.githubusercontent.com/63558665/120585160-45cbe800-c3ff-11eb-941e-21711c12f057.png)
 No dependencies among pixels, can generate all pixels at the same time!
 Cannot optimize directly, derive and optimize lower bound on likelihood instead
 trained autoencoder-->decoder-->extract feature :Transfer from large, unlabeled dataset to small, labeled dataset.
 Autoencoders can reconstruct data, and can learn features to initialize a supervised model
 Features capture factors of variation in training data. 
 But we can’t generate new images from an autoencoder because we don’t know the space of z
 ![image](https://user-images.githubusercontent.com/63558665/120585413-afe48d00-c3ff-11eb-801d-54ded05cd109.png)
 Choose prior p(z) to be simple, e.g.Gaussian. Reasonable for latent attributes, e.g. pose, how much smile
 
 ![image](https://user-images.githubusercontent.com/63558665/120585901-92fc8980-c400-11eb-823d-40ab9b13d120.png)

 ![image](https://user-images.githubusercontent.com/63558665/120586079-dd7e0600-c400-11eb-814d-b578c9223a57.png)


* Why dimensionality reduction?
Want features to capture meaningful factors of variation in data

* GANS: Sample from a simple distribution we can easily sample from, e.g. random noise.Learn transformation to training distribution.
  But we don’t know which sample z maps to which training image -> can’t learn by reconstructing training images
  Solution: Use a discriminator network to tell whether the generate image is within data distribution (“real”) or not
  
  Discriminator network: try to distinguish between real and fake images
  Generator network: try to fool the discriminator by generating real-looking images
  minmax game
  Discriminator (θd) wants to maximize objective such that D(x) is close to 1 (real) and D(G(z)) is close to 0 (fake)
  Generator (θg) wants to minimize objective such that D(G(z)) is close to 1 (discriminator is fooled into thinking generated G(z) is real)
  ![image](https://user-images.githubusercontent.com/63558665/120586384-714fd200-c401-11eb-8590-c944d7c73f4d.png)
  ![image](https://user-images.githubusercontent.com/63558665/120586592-cdb2f180-c401-11eb-88db-4964a8c29e15.png)
  
  ![image](https://user-images.githubusercontent.com/63558665/120586621-da374a00-c401-11eb-920b-5c5201c524cd.png)

