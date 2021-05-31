# Computer_Vision

### 1) Image classification:
* what is image classification?
Image classification is the process of categorizing and labeling groups of pixels or vectors within an image based on specific rule
* challenges for image classification?
   * viewpoint variation: all pixels change when the camera moves
   * background cluster: objects are similar to the background (color or texture)
   * illumination: (too dark or too light)
   * occlusion: hiden by other objects
   * deformation: different shape/pose of the same objects
   * intraclass variation:differnt types of the same object
   * scale variation:Visual classes often exhibit variation in their siz
 
 * data-driven methods: k and distance choised, evaluation methods
   * Nearest Neighbor classifier:The nearest neighbor classifier will take a test image, compare it to every single one of the training images, and predict the label of the closest training image.One of the simplest possibilities is to compare the images pixel by pixel and add up all the differences. In other words, given two images and representing them as vectors I1,I2 , a reasonable choice for comparing them might be the L1 distance.
         
        ![image](https://user-images.githubusercontent.com/63558665/120114755-a7b1f680-c14e-11eb-9122-f4c75d58a0b4.png)
     
     L2 distance:
         ![image](https://user-images.githubusercontent.com/63558665/120114899-4f2f2900-c14f-11eb-8e70-00fdf0e263e5.png)
           * ad: simple to implement and understand, no need to training
           * disad:computation cost
    * k-nearest neighbor classifier:instead of finding the single closest image in the training set, we will find the top k closest images, and have them vote on the label of the test image. when k==1, it is nearest neghbor classifier. knn with pixel distance never used because it is sensitive to lighting, and distance metrics on pixels are not informative.
           * disad: The classifier must remember all of the training data and store it for future comparisons with the test data. This is space inefficient because datasets may easily be gigabytes in size. Classifying a test image is expensive since it requires a comparison to all training images.
       
       ![image](https://user-images.githubusercontent.com/63558665/120115147-63bff100-c150-11eb-8425-4d92a49cdcff.png)
     * tuning hyperparameters:
          * validation: split the training dataset into training and validation and evaluate the model on test set (don't touch the test set) 70%-90%
          * cross_validation:split data into folds,try each fols as validation and average the results (useful for small datasets),computation expensive-->validation approach.

### 2) linear classification
   
   ![image](https://user-images.githubusercontent.com/63558665/120115968-d7afc880-c153-11eb-88aa-5ad71a1767d8.png)
* loss:

     ![image](https://user-images.githubusercontent.com/63558665/120116601-d6cc6600-c156-11eb-8826-46c3bdeca9d0.png)
      * SVM loss: 错误分类更高的分数，正确分类 is 0 l1 and l2 SVM loss
          ![image](https://user-images.githubusercontent.com/63558665/120117034-257aff80-c159-11eb-9ced-793c3a178ba3.png)
          ![image](https://user-images.githubusercontent.com/63558665/120116958-d6cd6580-c158-11eb-9dde-6916a3e0287c.png)
       questions:
           * if the correct score decrease maybe not affect the loss
           * at inilization, weight is small so all s about to 0
           * whether w is unique? it is no unique w=2w and L=0---> which is better w or 2w?--->reguralization
      * regularization:
           ![image](https://user-images.githubusercontent.com/63558665/120117062-522f1700-c159-11eb-9c99-b09383b1de6b.png)
            * methods:
                ![image](https://user-images.githubusercontent.com/63558665/120117110-9de1c080-c159-11eb-8795-d6ed8875a0ef.png)
             * why?
                  * Express preferneces over weights
                  * make the model simple so it works test data
                  * improve optimization by adding curvature
                                 
     * softmax: interpret classifier score into probability, probability sum to 1
              ![image](https://user-images.githubusercontent.com/63558665/120117288-763f2800-c15a-11eb-9c20-b11ab39c2090.png)
     
     ![image](https://user-images.githubusercontent.com/63558665/120117410-0f6e3e80-c15b-11eb-92cc-1a67f803002a.png)

* optimization: how to find the best w?
     gradient descent
### 3) neural network-multiple layers neural network
   
   ![image](https://user-images.githubusercontent.com/63558665/120117682-54df3b80-c15c-11eb-9cbc-26906f99b548.png)

   ![image](https://user-images.githubusercontent.com/63558665/120117694-6294c100-c15c-11eb-8fc0-13958b764a2d.png)
   why activation?--> W2*W1=W3  end up with linear classifier again!
   
   ![image](https://user-images.githubusercontent.com/63558665/120117745-ab4c7a00-c15c-11eb-8a22-2ff3352efbe1.png)
   ![image](https://user-images.githubusercontent.com/63558665/120117758-b6070f00-c15c-11eb-8bf9-efc7eb33bcd4.png)
   Multiple layer Neural network:
   ![image](https://user-images.githubusercontent.com/63558665/120117774-ca4b0c00-c15c-11eb-8930-57187382e583.png)

### 4) linear classifier--> multiple layers neural network-->covolution network
   
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
   
### 5) training
* one time setup
      * activation function
            * sigmoid:
                       * Saturated neurons “kill” the gradients (-inf,0,inf)--> the gradients flowing back will be zero and weights will never change
                       * Sigmoid outputs are not zero-centered--> local gradient of sigmoid is always positive or negative-->zigzag
                       * exp() computation expensive
             * Tanh:
                       * Saturated neurons “kill” the gradients (-inf,0,inf)--> the gradients flowing back will be zero and weights will never change
                       * Sigmoid outputs are zero-centered (nice)
             * ReLU:
                       * not saturate (in +region)
                       * computation efficiently
                       * converge mush faster than sigmoid/Tanh in pratice (eg.6x)
                       * not zero-centered output
                       * an annoyance
                       * Dead ReLU will never activate and no update weight
              * LeakyReLU:
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
            using ReLu be careful with learning rate--  Don’t use sigmoid or tanh---Try out Leaky ReLU / Maxout / ELU / SELU--To squeeze out some marginal gains
      * preprocessing:consider what happends when the input to a neura is always positive--->zigzag path
                * zero-mean data--> visualize data with PCA and Whitening
                        * substract the mean image(AlexNet)
                        * substract per-channel mean (VGGNet)
                        * substract per-channel mean and divide by per-channel std (ResNet)
                * normalization:Before normalization: classification loss very sensitive to changes in weight matrix; hard to optimize. After normalization: less sensitive to small changes in weights; easier to optimize
      * weight initialization:
                 * small random numbers:(gaussian with zero mean and 1e-2 standard deviation)-->work with small network, but no deep network---> vanishing gradient-->no learning
                 * “Xavier” Initialization: Activations are nicely scaled for all layers-->ReLU Activations collapse to zero again-->
                     ![image](https://user-images.githubusercontent.com/63558665/120121487-c4135a80-c171-11eb-8561-d06a4b12ef89.png)
                 * Kaiming/MSRA initilization
                      ![image](https://user-images.githubusercontent.com/63558665/120121622-70554100-c172-11eb-992e-930fa4462046.png)
        depending on differnt activation function using differnt weight inilization
      * regularization: zero-mean unit-variance activationa and improve single model performance
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
                   * 
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
Consider dropout for large-->fully-connected layers
Batch normalization and data augmentation almost always a good idea
Try cutout and mixup especially for small classification datasets
* Training Dynamics  
       * babysitting the learning process
                        * Learning rate decays over time/cosine/linear
                            ![image](https://user-images.githubusercontent.com/63558665/120122292-e1e2be80-c175-11eb-8ddc-0cca78a8a650.png)
        Adam is a good default choice in many cases; it often works ok even with constant learning rate.SGD+Momentum can outperform Adam but mayrequire more tuning of LR and schedule-->Try cosine schedule, very few hyperparameters! If you can afford to do full batch updates then try out-->L-BFGS (and don’t forget to disable all sources of noise)
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

### 7) RNN
* application: image caption/action prediction/Video Captioning/ Video classification on frame level

![image](https://user-images.githubusercontent.com/63558665/120123592-b2d04b00-c17d-11eb-89ce-c0f7fd4d0b66.png)
1.Re-use the same weight matrix at every time-step
2. many -->many 

![image](https://user-images.githubusercontent.com/63558665/120123662-18243c00-c17e-11eb-953e-df597f42170d.png)

many-->one: : Encode input sequence in a single vector

![image](https://user-images.githubusercontent.com/63558665/120123659-10fd2e00-c17e-11eb-8335-175e50c6648a.png)

one-->many: : Produce output sequence from single input vector

![image](https://user-images.githubusercontent.com/63558665/120123650-05aa0280-c17e-11eb-94f5-e15ce557d91d.png)
![image](https://user-images.githubusercontent.com/63558665/120123743-a7315400-c17e-11eb-9814-85b0e3a5061b.png)






