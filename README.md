# Learning Granularity-Aware Affordances from Human-Tool Interaction for Tool-Based Functional Grasping in Dexterous Grasping（GAAF-DEX）

https://github.com/user-attachments/assets/ceefff6a-80fa-46e7-ae9c-970f269b9614

https://github.com/user-attachments/assets/ec3d09d0-f97e-4426-9bf6-a06ca95a7110
## Abstract

To enable robots to use tools, the initial step is teaching robots to employ dexterous gestures for touching specific areas precisely where tasks are performed. Affordance features of objects serve as a bridge in the functional interaction between agents and objects. However, leveraging these affordance cues to help robots achieve functional tool grasping remains unresolved. To address this, we propose a granularity-aware affordance feature extraction method for locating functional affordance areas and predicting dexterous coarse gestures. We study the intrinsic mechanisms of human tool use. On one hand, we use fine-grained affordance features of object-functional finger contact areas to locate functional affordance regions. On the other hand, we use highly activated coarse-grained affordance features in hand-object interaction regions to predict grasp ges tures. Additionally, we introduce a model-based post-processing module that transforms affordance localization and gesture pre diction into executable robotic actions. This forms GAAF-Dex, a complete framework that learns Granularity-Aware Affordances from human-object interaction to enable tool-based functional grasping with dexterous hands. Unlike fully-supervised methods that require extensive data annotation, we employ a weakly supervised approach to extract relevant cues from exocentric (Exo) images of hand-object interactions to supervise feature extraction in egocentric (Ego) images. To support this approach, we have constructed a small-scale dataset, Functional Affordance Hand-object Interaction Dataset (FAH), which includes nearly 6𝐾 images of functional hand-object interaction Exo images and Ego images of 18 commonly used tools performing 6 tasks. Ex tensive experiments on the dataset demonstrate that our method outperforms state-of-the-art methods, and real-world localization and grasping.
## Usage

### 1. Requirements

Code is tested under Pytorch 1.12.1, python 3.7, and CUDA 11.6

```
pip install -r requirements.txt
```

### 2. Dataset

You can download the FAH from [Baidu Pan (3.23G)](https://pan.baidu.com/s/126RmaKBZG_QddX2B4Z6jnw?pwd=ip6q). The extraction code is: `ip6q`.

### 3. Train and Test
Our pretrained model can be downloaded
  from [Baidu Pan ](https://pan.baidu.com/s/1NlPKtQ7gQMfAoSPwRRTMbQ?pwd=rwku). The extraction code is: `rwku`.
Run following commands to start training or testing:

```
python train_gaaf.py --data_root <PATH_TO_DATA>

python test.py --data_root <PATH_TO_DATA> --model_file <PATH_TO_MODEL>
```

## Citation
```
@article{yang2024learning,
  title={Learning granularity-aware affordances from human-object interaction for tool-based functional grasping in dexterous robotics},
  author={Yang, Fan and Chen, Wenrui and Yang, Kailun and Lin, Haoran and Luo, DongSheng and Tang, Conghui and Li, Zhiyong and Wang, Yaonan},
  journal={arXiv preprint arXiv:2407.00614},
  year={2024}
}
}
```
## Anckowledgement

This repo is based on [Cross-View-AG](https://github.com/lhc1224/Cross-View-AG)
, [LOCATE](https://github.com/Reagan1311/LOCATE) Thanks for their great work!
