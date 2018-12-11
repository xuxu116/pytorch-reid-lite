## Overview

This codebase is for training/deploying models in pytorch (onnx), 
currently it provides basic protocols for model training, evaluation and deploying.

## Features

- Spectral feature transform for post-processing(https://arxiv.org/abs/1811.11405).
- PCB structure (https://arxiv.org/abs/1711.09349); Improved training strategy
- GAN related person generator(unstable)
- AM softmax & triplet loss
- step-wise LR warm-up

## Install Dependency

This code depends on pytorch v0.4 and torchvision, run the 
following command to install pytorch:

```
pip install --user torch==0.4 torchvision==0.2.1 tensorflow==1.8 tensorboardX lmdb -i https://pypi.douban.com/simple/
```

### Model Training

To train a model, clone the repo, modify params.json as you need, and run train.py.

```
cd pytorch-reid-lite
# Modify params.json - specify your own working dir.
# sub_working_dir is optional
python train.py --operation start_train --config_path params.json --sub_working_dir SUB\_WORKING\_DIR\_NAME
```

### On-the-Fly Evaluation:

You can enable on-the-fly automatic evaluation by setting "type" under "evaluation_params" key in params.json (default is "None"). If set, after each epoch the code will run your evaluation, and only saves the best-performing model.

The code currently supports "market\_evaluate" for person-reid and "classification\_evaluate" for image classification, but it is easy to extend this to support other evalutiaons (like LFW). All you need to do is create a new file - say "lfw\_evaluate.py" in the evaluate folder, and expose a run\_eval method which takes in your training config and returns your evaluation result. See evaluate/market\_evalute.py for an example.

### Offline evaluation
```
python evaluator.py eval_params.json 
```

### Tensorboard visualization

You can visualize your training progress with tensorboardX (a pytorch integration of Tensorboard for Tensorflow), the code generates an event file in your sub working dir, to run tensorboard, do so as you would when using Tensorflow: 
```
cd ~/.local/bin
./tensorboard --logdir=YOUR_SUB_WORKING_DIR --port=YOUR_PORT
```
## Tips summary
Which benefits:
 - PCB structure
 - PCB randomly update
 - batchnorm
 - random erasing, zero paddding crop
 - warm-up learning rate
 - global branch
 - small batchsize

Which might helps:
 - feature erasing
 - feature mask
 - tri-loss
 - balanced sampling
 - multi-gpu training (differs in BN layer)
 
 Not working:
 - adam
 - am-softmax
 - bias in FC layer or BN
 

## Baselines
backbone | imgSize | PCB | rank1  | map | aug. | batchsize | comments
--- | --- | --- | --- | --- | --- | --- | ---
resnet-50 | 384*128 |1536/6 |0.628266|0.346756|mirro | 64*1 |classifier no bias, 60 epoch, decay per 40
resnet-50 | 384*128 |1536/6 |0.683492|0.411627|mirro | 64*1 |weight_decay from 4e-5 to 5e-4
resnet-50 | 384*128 |1536/6 |0.837886|0.620621|mirro | 64*1 |add dropout before PCB
resnet-50 | 384*128 |1536/6 |0.856888|0.640600|mirro | 64*1 |last_conv_stride=1
resnet-50 | 384*128 |1536/6 |0.920724|0.755717|mirro | 64*1 |add BN to pcb stripe
resnet-50 | 384*128 |1536/6 |0.921318|0.765050|mirro,RE | 64*1 |add BN to pcb stripe
resnet-50 | 384*128 |1536/6 |0.927553|0.776928|mirro,RE | 64*1 |add global branch
resnet-50 | 384*128 |1536/6 |0.926366|0.784323|mirro,RE | 64*1 |random erase 1 branch, wp
resnet-50 | 384*128 |1536/6 |0.928147|0.785333|mirro,RE | 64*1 |random erase 5 branch, wp
resnet-50 | 384*128 |1536/6 |0.929929|0.790466|mirro,RE | 64*1 |random erase 6 branch, wp
resnet-50 | 384*128 |1536/6 |0.929038|0.787618|mirro,RE | 64*1 |random erase 6 branch, wp, 32X2
resnet-50 | 384*128 |1536/6 |0.927850|0.782085|mirro,RE | 64*1 |random erase 6 branch, wp, 16X4
resnet-50 | 384*128 |1536/6 |0.928741|0.771841|mirro,RE | 64*1 |global branch m=0.1
resnet-50 | 384*128 |1536/6 |0.926960|0.777564|mirro,RE | 64*1 |global branch m=0.3, warm-up
resnet-50 | 384*128 |1536/6 |0.926069|0.764451|mirro,RE | 64*1 |global branch m=0.4, warm-up
resnet-50 | 384*128 |1536/6 |0.924287|0.777912|mirro,RE | 64*1 |global branch m=0.4, warm-up, mask
resnet-50 | 384*128 |1536/6 |0.920428|0.775502|mirro,RE | 64*1 |mask@global branch
resnet-50 | 384*128 |1536/6 |0.930523|0.783172|mirro,RE | 64*1 |change hue
resnet-50 | 384*128 |1536/6 |0.920724|0.768056|mirro | 32*1 |120 epoch, decay per 40, hue
resnet-50 | 256*128 |1024/4 |0.907957|0.731270|mirro | 32*1 |120 epoch, decay per 40
resnet-50 | 256*128 |1024/4 |0.907957|0.750186|mirro,RE | 32*1 |120 epoch, decay per 40

For following settings
- `PCB branchs = 6`
- `batch_size = 64`
- image size `h x w = 384 x 128`

GPU memory usage:
- 9529MiB for `last_conv_stride=1` (130 example/sec)
- 7155MiB for `last_conv_stride=2` (170 example/sec)

## add global branchs at resnet-stage-4(start from no relu and dropout, adaptiveMaxPool)
backbone | imgSize | PCB | rank1  | map | aug. | bs | comments
--- | --- | --- | --- | --- | --- | --- | ---
resnet-50 | 384*128 |1536+256|0.935273|0.802506|mirro,RE| 64*1 |no relu & dropout, global f erasing(RE)
resnet-50 | 384*128 |1536+256|0.940321|0.818069|mirro,RE| 64*1 |padcrop_10
resnet-50 | 384*128 |1536+256|0.935570|0.820962|mirro,RE| 64*1 |random erase 6 branch(RB)
resnet-50 | 384*128 |1536+256|0.935570|0.821505|mirro,RE| 64*1 |dropout, without feature erasing
resnet-50 | 384*128 |1536+256|0.937055|0.818202|mirro,RE| 64*1 |dropout, no_pcbRE, no f_RE
resnet-50 | 384*128 |1536+256|0.937945|0.815731|mirro,RE| 64*1 |pcbFE0.3, no_pcbRE, no f_RE
resnet-50 | 384*128 |1536+256|0.927257|0.793121|mirro,RE| 64*1 |mask@ all bracnchs, pcbRE
resnet-50 | 384*128 |1536+256|0.940024|0.818851|mirro,RE| 64*1 |no f_RE, update max loss branch
resnet-50 | 384*128 |1536+256|0.925178|0.807808|mirro,RE| 32*2 |pcb_s_triloss, no_mask, no_pcbRE
resnet-50 | 384*128 |1536+256|0.932304|0.819861|mirro,RE| 32*2 |pcb_s_triloss m=0.16
resnet-50 | 384*128 |1536+256|0.940618|0.826704|mirro,RE| 32*2 |g_triloss + pcb_g_triloss(soft)
resnet-50 | 384*128 |1536+256|0.940618|0.831889|mirro,RE| 32*2 |g_tri + pcb_g_tri, m=0.16
resnet-50 | 384*128 |1536+256|0.941211|0.835557|mirro,RE| 32*2 |g_tri + pcb_g_tri, pcbRB6
resnet-50 | 384*128 |1536+256|0.943290|0.834388|mirro,RE| 32*2 |g_tri_0.16, pcbRB6
resnet-50 | 384*128 |1536+256|0.939133|0.826529|mirro,RE| 32*2 |g_tri_0.16, pcbRB6+mask
resnet-50 | 384*128 |1536+256|0.883314|0.739106|mirro,RE| 32*2 |g_tri_0.16, pcbRB6+am0.3s15
resnet-50 | 384*128 |1536+256|0.917458|0.791989|mirro,RE| 32*2 |g_tri_0.16, pcbRB6+am0.3s0
resnet-50 | 384*128 |1536+256|0.940024|0.828990|mirro,RE| 32*2 |g_tri_0.16, pcbRB6, no additional stage-4
resnet-50 | 384*128 |1536+256|0.939727|0.830806|mirro,RE| 48*3 |g_tri_0.16, pcbRB6
resnet-50 | 384*128 |1536+256|0.939133|0.829087|mirro,RE| 32*2 |g_tri_0.16, pcbRB6, BN_nobias

Conclusions:
1. global branch after stage-4 helps
2. AM-softmax still cause overfitting
3. Tri-loss only used in global features
4. Update each PCB branches randomly

backbone | imgSize | PCB | rank1  | map | aug. | batchsize | comments
--- | --- | --- | --- | --- | --- | --- | ---
resnet-50 | 256*128 |256*1 |0.802553|0.601922|mirro | 128*1 | last_stride=1
resnet-50 | 256*128 |256*1 |0.869062|0.685709|mirro | 128*1 | add BN, Dropout after feature layer
resnet-50 | 256*128 |256*1 |0.867874|0.685979|mirro | 128*1 | cls no bias (not use)
resnet-50 | 256*128 |256*1 |0.893112|0.740011|mirro | 32*1 | add BN, Dropout after feature layer
resnet-50 | 256*128 |256*1 |0.898753|0.749818|mirro,RE | 32*1 | 120 epoch, decay per 40
resnet-50 | 256*128 |256*1 |0.907660|0.763313|mirro,RE | 32*1 | warm-up before 20 epoch
resnet-50 | 256*128 |256*1 |0.923100|0.782874|mirro,RE | 8*4 | 700+ epochs
resnet-50 | 256*128 |256*1 |0.931116|0.813940|mirro,RE | 8*4 |pad_zero_crop
resnet-50 | 256*128 |256*1 |0.934382|0.819815|mirro,RE | 8*4 | spectral transf 0.5, pad_6
resnet-50 | 256*128 |256*1 |0.933789|0.821830|mirro,RE | 8*4 | st_0.5_norm, pad_6
resnet-50 | 256*128 |256*1 |0.900831|0.774981|mirro,RE | 16*8 | spectral st_0.5_norm, pad_6
resnet-50 | 256*128 |256*1 |0.922506|0.811486|mirro,RE | 8*4 | tri_m=0.16, pad_6
resnet-50 | 256*128 |256*1 |0.921912|0.801184|mirro,RE | 16*2 | tri_m=0.16, pad_6
resnet-50 | 256*128 |256*1 |0.905879|0.756945|mirro,RE | 32*1 | am=0.0
resnet-50 | 256*128 |256*1 |0.898753|0.756945|mirro,RE | 32*1 | am=0.0(w normalized)
resnet-50 | 256*128 |256*1 |0.895190|0.756697|mirro,RE | 32*1 | am=0.1
resnet-50 | 256*128 |256*1 |0.906473|0.774181|mirro,RE | 32*1 | Add feature mask
resnet-50 | 256*128 |256*1 |0.914786|0.788952|mirro,RE | 32*1 | Change hue(with mask)
resnet-50 | 256*128 |256*1 |0.896081|0.738212|mirro,RE | 32*1 | Crop 288*144
resnet-50 | 256*128 |256*1 |0.849169|0.673918|mirro | 32*1 | adam, epoch 20 lr decay
resnet-50 | 256*128 |256*1 |0.864014|0.679649|mirro | 32*1 | adam, epoch 40 lr decay
resnet-50 | 256*128 |256*1 |0.867874|0.704566|mirro | 32*1 | global_pool 2048d as feature

For following settings
- `PCB branchs = 0`
- `batch_size = 128 # 64 causes divergence （w\o BN and dropout)`
- image size `h x w = 256 x 128`

GPU memory usage:
- 10343MiB for `last_conv_stride=1` (215 example/sec)



## Parameters

The params.json file contains the settings you need to run your model, here is a brief 
documentation of what they are about:

1. ***"batch\_size"***: The batch\_size PER GPU.
2. ***"batches\_dir"***: The path to your dataset generated by the open platform.
3. ***"data\_augmentation"*** contains the params related to data\_augmentation.
4. ***"epoch"***: How many epochs to train your model.
5. ***"imagenet\_pretrain"***: Whether to initialze your model with ImageNet pretrained network. Note that some networks might not support this.
6. ***"img\_h" and "img\_w"***: Size of the input image.
7. ***"lr"*** contains the params related to learning rate setting, where "base\_lr" denotes the initial learning rate for the base network and "fc\_lr" denotes the initial learning rate for the fc layers. Also note that "decay_step" here refers to training epochs.
8. ***"model\_params"*** contains the setting of network structure.
9. ***"optimizer"***: Which opitimization algorithm to use, default is is SGD.
10. ***"parallels"***: The GPU(s) to train your model on.
11. ***"pretrain\_snapshot"***: Path to pretrained model.
12. ***"weight\_decay"***: The l2-regularization parameter.
13. ***"fine\_tune"***: If set to "true", train only the final classification layer and freeze all layers before.
14. ***"evaluation\_params"***: Run different types of evaluation accordingly, now supports "market\_evaluate" and "classificaton\_evaluate".
15. ***"working\_dir"***: Where your model will be stored on disk.
19. ***"tri\_loss\_margin"***: If set, the model will be trained with the Triplet loss with batch-hard mining, set to "soft\_margin" to use the soft margin setting, and set to 0 to disbale.
20. ***"tri\_loss\_lambda\_cls"***: If set, the model will be trained with the Triplet loss and The Classicfication loss(softmax/AM-softmax) together, set to 0 to disbale.
21. ***"batch\_sampling\_params"***: If "class\_balanced" is set to true, then the code will sample each batch by first randomly selecting P classes and then randomly selecting K images for each class (batch\_size = P * K); set "class\_balanced" to false to use random sampling. Also note that if "class\_balanced" is set to true, the lr decay step will be counted as each iteration, as opposed to epoch for random sampling.
