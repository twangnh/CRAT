# This is implementation of CRAT, Cross-class gRAdient Transfusing for long-tailed object detection and instance segmentation.

Updates:\
[2022.12] creating repository\
[2023.01] uploading code with eqlv2\
TODO:\
integrating Groupsoftmax, Seesaw code and training configs\
combining sampling implementation

Learning object detectors under long-tailed data distribution is challenging and has been widely studied
recently, the prior works mainly focus on balancing the learning signal of classification task such
that samples from tail object classes are effectively recognized. However, the learning difficulty of
other class-wise tasks including bounding box regression and mask segmentation are not explored
before. In this work, we investigate how long-tailed distribution affects the optimization of box regression
and mask segmentation tasks. We find that although the standard class-wise box regression and
mask segmentation offer strong class-specific prediction, they suffer from limited training signal and
instability on the tail object classes. Aiming to address the limitation, our insight is that the knowledge
of box regression and object segmentation is naturally shared across classes. We thus develop
a cross class gradient transfusing (CRAT) approach to transfer the abundant training signal from
head classes to help the training of sample-scarce tail classes. The transferring process is guided by
the Fisher information to aggregate useful signals. CRAT can be seamlessly integrated into existing
end-to-end or decoupled long-tailed object detection pipelines to robustly learn class-wise box regression
and mask segmentation under long-tailed distribution. Our method improves the state-of-the-art
long-tailed object detection and instance segmentation models with an average of 3.0 tail AP on the
LVIS benchmark.

The Fisher gradient statistics on the task (regression/segmentation) layer are surprisingly effective at finding similar shape, appearance and context from the head classes for training the tail classes, here are some example highly weighted sampled selected by CRAT:

![image](https://user-images.githubusercontent.com/18298163/213604504-440c490f-2306-4ec1-9270-1f7d5e27a7af.png)
![image](https://user-images.githubusercontent.com/18298163/213604509-788b4397-6d30-4643-8426-4389e26cef70.png)
