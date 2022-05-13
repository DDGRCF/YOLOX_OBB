**YOLOX OBB -- YOLOX 旋转框**

![version](https://img.shields.io/badge/release_version-1.1.0-bule)
***
<img align=center>![result_vis](./assets/obb/vis_resize.png)
***
## **前言**
1. More rotated detection methods can reference [OBBDetection](https://github.com/jbwang1997/OBBDetection.git). 
2. Attention! For meeting my demand, I have changed a lot code in the scripts. And I don't change readme for tight time.

## **Update**
1. Support ignore empty images
2. Support ignore to apply rotate augmentation to horizontal targets
3. Support dota2 train eval test demo
4. Add the scripts which can convert coco datasets format to dota format, reference [coco2dota.py](./datasets/data_convert_tools/coco2dota.py)
4. Add Copy-Paste Augmention 
5. Add Resampling Augmention (PS: It means that the class which contains smaller numbers instances will be chosen with higher probability when copy paste)
6. Add TensorRT Demo reference [OBB](./demo/OBB) *Some Error.wait...* *It will be refactored*
7. Add YOLOv5 config style [configs](./configs)

## **Introduction**
### Method
1. You can view the item as baseline and apply you method on it. Next step, I will add some efficient methods into the project.
2. Origin OBB \ KLD Loss \ GWD Loss
## **Content**

- [Quick&nbsp;Start](#Quick&nbsp;Start)
- [Instruction](#Instruction)
  - [Data](#Data)
  - [Demo](#Demo)
  - [Train](#Train)
  - [Test](#Test)

- [Ralated&nbsp;Hub](#Ralated&nbsp;Hub)

## **Quick&nbsp;Start**

Firstly, create python environment

```shell
$ conda create -n yolox_obb python=3.7 -y
```

then, clone the github of the item

```shell
$ git clone --recursive https://github.com/DDGRCF/YOLOX_OBB.git
```
install BboxToolkit
```shell
$ cd BboxToolkit
$ python setup.py develop
```
then, you can adjust follow the [original quick start](./docs/quick_run.md) 

## **Instruction**
### **Data**
If We want to train your datasets, firstly you first convert your data as dota datasets format. If you have a coco annotation-style datasets, you can just convert it annoatations into dota format. We perpare a script for you.
```shell
$ cd my_exps
$ bash coco2dota.sh
# PS: you should change filename、diranme and so on.
```
### **Demo *remove ...***
I want to attention that we can't support fp16 again. I prepare the shell the demo script so that you can quick run obb demo as :
```shell
$ cd my_exps
$ bash demo_dota_obb.sh [data_type] 0 /path/to/you
# PS: 0 is to assign the train environment to 0 gpu, you can change it by youself and /path/to/you is your demo images. data_type can be selected in [dota1_0, dota2_0]
```
 
### **Train *remove ...***
**Three points**:
1. you must follow the [BboxToolkit](./BboxToolkit/USAGE.md) to prepare your dataset(#PS: if you is not dota dataset format, you can convert into dota dataset format first)
2. We define the model default training parameters as following:
3. If you want to debug with visualizing the augmentation images, you can add `enable_debug=True` to the config and the vis images will be outputed into the work dir.

| model | max epoch | enable_mixup  | enable_mosaic |no aug epoch | obj_loss_weight | cls_loss_weight | iou_loss_weight | reg_loss_weight | 
| :-:   | :-:       | :-:           |:-:            |:-:          | :-:             | :-:             | :-:             | :-:             |
|yolox_l| 40        | True          | True          |5            | 1.0             | 1.0             | 5.0             | 1.0             |

Of course, this group parameters is not the best one, so you can try youself. And for the quick train, I have prepare the shell scripts, too.

```shell
$ cd my_exps
$ bash train_dota_obb.sh data_type 0
```
As I set parameters above with 4 batch size per gpu, results show as following:

| model   | mAP   | lowest mAP       | highest mAP |
| :-:    | :-:  | :-:              | :-:         |
| yolox_l | 63.98 | helicopter 29.92 | tennis-court 90.87 |

Adopt yolox_s model and get results as following:

| model   | mAP   | lowest mAP       | highest mAP |
| :-:    | :-:  | :-:              | :-:         |
| yolox_s | 58.26 | helicopter 33.96 | tennis-court 90.81 |

Adopt yolox_s model(60 epochs) and get results as following:

| model   | mAP   | lowest mAP       | highest mAP |
| :-:    | :-:  | :-:              | :-:         |
| yolox_s | 62.62 | soccer-ball-field 35.58 | tennis-court 90.80 |

After add Copy Paste and Resample:

| model   | mAP   | lowest mAP       | highest mAP |
| :-:    | :-:  | :-:              | :-:         |
| yolox_s | 70.78 | helicopter 49.42 | tennis-court 90.86 |
### **Test *remove ...***
Test contains eval online and generate submission file, they are all convenient.
* for eval online
```shell
$ cd my_exps
$ ./eval_dota_obb.sh [data_type] eval 0
# PS: for convenience, I set default parameters. So, eval means evaluating DOTA val datasets.
```
* for generate submission file
```shell
$ cd my_exps
$ ./eval_dota_obb.sh [data_type] test 0
```
## **Ralated&nbsp;Hub**

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX.git)

- [OBBDetection](https://github.com/jbwang1997/OBBDetection.git)

- [BboxToolkit](https://github.com/jbwang1997/BboxToolkit.git)
