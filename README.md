# This is implementation of CRAT, Cross-class gRAdient Transfusing for long-tailed object detection and instance segmentation.

### Note : This is example implementation with EQLv2, the main implementation of CRAT is at [this commit](https://github.com/twangnh/CRAT/commit/8f892a04ca08d1c018911d5cb888b36fe4847220)

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

## Prepare LVIS Dataset

***for images***

LVIS uses same images as COCO's, so you need to donwload COCO dataset at folder ($COCO), and link those `train`, `val` under folder `lvis`($LVIS).

```
mkdir -p data/lvis
ln -s $COCO/train $LVIS
ln -s $COCO/val $LVIS
ln -s $COCO/test $LVIS
```
***for annotations***

Download the annotations from [lvis webset](https://lvisdataset.org/)

```
cd $LVIS
mkdir annotations
```
then places the annotations at folder ($LVIS/annotations)

Finally you will have the file structure like below:

    data
      ├── lvis
      |   ├── annotations
      │   │   │   ├── lvis_v1_val.json
      │   │   │   ├── lvis_v1_train.json
      │   ├── train2017
      │   │   ├── 000000004134.png
      │   │   ├── 000000031817.png
      │   │   ├── ......
      │   ├── val2017
      │   ├── test2017

***for API***

The official lvis-api and mmlvis can lead to some bugs of multiprocess. See [issue](https://github.com/open-mmlab/mmdetection/issues/4112)

So you can install this LVIS API from my modified repo.
```
pip install git+https://github.com/tztztztztz/lvis-api.git
```

## Testing with pretrain_models
```bash
# ./tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
./tools/dist_test.sh configs/eqlv2/eql_r50_8x2_1x.py data/pretrain_models/eql_r50_8x2_1x.pth 8 --out results.pkl --eval bbox segm
```


## Training

```bash
# ./tools/dist_train.sh ${CONFIG} ${GPU_NUM}
./tools/dist_train.sh ./configs/end2end/eql_r50_8x2_1x.py 8 
```

![image](https://user-images.githubusercontent.com/18298163/213604504-440c490f-2306-4ec1-9270-1f7d5e27a7af.png)
![image](https://user-images.githubusercontent.com/18298163/213604509-788b4397-6d30-4643-8426-4389e26cef70.png)


