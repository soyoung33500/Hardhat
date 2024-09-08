# 통계데이터사이언스전공 202010936 전소영

# 학습


```python
#구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
# yolov5 git 다운로드
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!git reset --hard fbe67e465375231474a2ad80a4389efc77ecff99
```

    Cloning into 'yolov5'...
    remote: Enumerating objects: 16013, done.[K
    remote: Counting objects: 100% (46/46), done.[K
    remote: Compressing objects: 100% (33/33), done.[K
    remote: Total 16013 (delta 25), reused 23 (delta 13), pack-reused 15967[K
    Receiving objects: 100% (16013/16013), 14.66 MiB | 24.18 MiB/s, done.
    Resolving deltas: 100% (10987/10987), done.
    /content/yolov5
    HEAD is now at fbe67e4 Fix `OMP_NUM_THREADS=1` for macOS (#8624)
    


```python
#,패키지, 유틸, 모델, 기본가중치 다운로드
!pip install -qr requirements.txt
import torch

from IPython.display import Image, clear_output
from utils.downloads import attempt_download
```


```python
#roboflow 에서 데이터셋 다운로드
%cd /content/yolov5

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="5WLMzX4BhjFxpVuILz1o")
project = rf.workspace("joseph-nelson").project("hard-hat-workers")
dataset = project.version(13).download("yolov5")
```

    /content/yolov5
    Collecting roboflow
      Downloading roboflow-1.1.7-py3-none-any.whl (58 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.8/58.8 kB[0m [31m1.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting certifi==2022.12.7 (from roboflow)
      Downloading certifi-2022.12.7-py3-none-any.whl (155 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m155.3/155.3 kB[0m [31m7.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting chardet==4.0.0 (from roboflow)
      Downloading chardet-4.0.0-py2.py3-none-any.whl (178 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m178.7/178.7 kB[0m [31m21.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cycler==0.10.0 (from roboflow)
      Downloading cycler-0.10.0-py2.py3-none-any.whl (6.5 kB)
    Collecting idna==2.10 (from roboflow)
      Downloading idna-2.10-py2.py3-none-any.whl (58 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.8/58.8 kB[0m [31m7.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.4.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from roboflow) (3.7.1)
    Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.23.5)
    Collecting opencv-python-headless==4.8.0.74 (from roboflow)
      Downloading opencv_python_headless-4.8.0.74-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (49.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.1/49.1 MB[0m [31m34.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from roboflow) (9.4.0)
    Collecting pyparsing==2.4.7 (from roboflow)
      Downloading pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.8/67.8 kB[0m [31m7.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.8.2)
    Collecting python-dotenv (from roboflow)
      Downloading python_dotenv-1.0.0-py3-none-any.whl (19 kB)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.31.0)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.16.0)
    Collecting supervision (from roboflow)
      Downloading supervision-0.16.0-py3-none-any.whl (72 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m72.2/72.2 kB[0m [31m8.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: urllib3>=1.26.6 in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.0.7)
    Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.10/dist-packages (from roboflow) (4.66.1)
    Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (6.0.1)
    Collecting requests-toolbelt (from roboflow)
      Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.5/54.5 kB[0m [31m6.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (1.1.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (4.43.1)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (23.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->roboflow) (3.3.0)
    Requirement already satisfied: scipy<2.0.0,>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from supervision->roboflow) (1.11.3)
    Installing collected packages: python-dotenv, pyparsing, opencv-python-headless, idna, cycler, chardet, certifi, supervision, requests-toolbelt, roboflow
      Attempting uninstall: pyparsing
        Found existing installation: pyparsing 3.1.1
        Uninstalling pyparsing-3.1.1:
          Successfully uninstalled pyparsing-3.1.1
      Attempting uninstall: opencv-python-headless
        Found existing installation: opencv-python-headless 4.8.1.78
        Uninstalling opencv-python-headless-4.8.1.78:
          Successfully uninstalled opencv-python-headless-4.8.1.78
      Attempting uninstall: idna
        Found existing installation: idna 3.4
        Uninstalling idna-3.4:
          Successfully uninstalled idna-3.4
      Attempting uninstall: cycler
        Found existing installation: cycler 0.12.1
        Uninstalling cycler-0.12.1:
          Successfully uninstalled cycler-0.12.1
      Attempting uninstall: chardet
        Found existing installation: chardet 5.2.0
        Uninstalling chardet-5.2.0:
          Successfully uninstalled chardet-5.2.0
      Attempting uninstall: certifi
        Found existing installation: certifi 2023.7.22
        Uninstalling certifi-2023.7.22:
          Successfully uninstalled certifi-2023.7.22
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    lida 0.0.10 requires fastapi, which is not installed.
    lida 0.0.10 requires kaleido, which is not installed.
    lida 0.0.10 requires python-multipart, which is not installed.
    lida 0.0.10 requires uvicorn, which is not installed.[0m[31m
    [0mSuccessfully installed certifi-2022.12.7 chardet-4.0.0 cycler-0.10.0 idna-2.10 opencv-python-headless-4.8.0.74 pyparsing-2.4.7 python-dotenv-1.0.0 requests-toolbelt-1.0.0 roboflow-1.1.7 supervision-0.16.0
    



    loading Roboflow workspace...
    loading Roboflow project...
    

    Downloading Dataset Version Zip in Hard-Hat-Workers-13 to yolov5pytorch:: 100%|██████████| 478642/478642 [00:06<00:00, 71920.41it/s]

    
    

    
    Extracting Dataset Version Zip to Hard-Hat-Workers-13 in yolov5pytorch:: 100%|██████████| 33746/33746 [00:05<00:00, 6677.03it/s]
    


```python
#학습시키기
%%time
%cd /content/yolov5/
!python train.py --img 416 --batch 32 --epochs 22 --data {dataset.location}/data.yaml \
--cfg ./models/yolov5s.yaml --weights '/content/drive/MyDrive/yolov5/yolov5s_results2/weights/best.pt' --project /content/drive/MyDrive/ --name yolov5s_results1
```

    /content/yolov5
    2023-10-24 03:11:52.775252: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-24 03:11:53.733133: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    [34m[1mtrain: [0mweights=/content/drive/MyDrive/yolov5/yolov5s_results2/weights/best.pt, cfg=./models/yolov5s.yaml, data=/content/yolov5/Hard-Hat-Workers-13/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=22, batch_size=32, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=/content/drive/MyDrive/, name=yolov5s_results1, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
    [34m[1mgithub: [0m⚠️ YOLOv5 is out of date by 592 commits. Use `git pull` or `git clone https://github.com/ultralytics/yolov5` to update.
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    [34m[1mhyperparameters: [0mlr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
    [34m[1mWeights & Biases: [0mrun 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir /content/drive/MyDrive', view at http://localhost:6006/
    Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
    100% 755k/755k [00:00<00:00, 15.1MB/s]
    Overriding model.yaml nc=80 with nc=2
    
                     from  n    params  module                                  arguments                     
      0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
      1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
      2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
      3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
      4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
      5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
      6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
      7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
      8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
      9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
     10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
     14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
     18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
     21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
     24      [17, 20, 23]  1     18879  models.yolo.Detect                      [2, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
    YOLOv5s summary: 270 layers, 7025023 parameters, 7025023 gradients, 16.0 GFLOPs
    
    Transferred 348/349 items from /content/drive/MyDrive/yolov5/yolov5s_results2/weights/best.pt
    Scaled weight_decay = 0.0005
    [34m[1moptimizer:[0m SGD with parameter groups 57 weight (no decay), 60 weight, 60 bias
    [34m[1malbumentations: [0mBlur(always_apply=False, p=0.01, blur_limit=(3, 7)), MedianBlur(always_apply=False, p=0.01, blur_limit=(3, 7)), ToGray(always_apply=False, p=0.01), CLAHE(always_apply=False, p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    [34m[1mtrain: [0mScanning '/content/yolov5/Hard-Hat-Workers-13/train/labels' images and labels...14748 found, 0 missing, 1 empty, 0 corrupt: 100% 14748/14748 [00:01<00:00, 10322.59it/s]
    [34m[1mtrain: [0mNew cache created: /content/yolov5/Hard-Hat-Workers-13/train/labels.cache
    [34m[1mval: [0mScanning '/content/yolov5/Hard-Hat-Workers-13/valid/labels' images and labels...1413 found, 0 missing, 0 empty, 0 corrupt: 100% 1413/1413 [00:00<00:00, 7428.12it/s]
    [34m[1mval: [0mNew cache created: /content/yolov5/Hard-Hat-Workers-13/valid/labels.cache
    Plotting labels to /content/drive/MyDrive/yolov5s_results1/labels.jpg... 
    
    [34m[1mAutoAnchor: [0m5.88 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
    Image sizes 416 train, 416 val
    Using 8 dataloader workers
    Logging results to [1m/content/drive/MyDrive/yolov5s_results1[0m
    Starting training for 22 epochs...
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
          7/21        0G   0.04076   0.02661  0.004366       162       416: 100% 461/461 [15:14<00:00,  1.98s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:43<00:00,  1.88s/it]
                     all       1413       5252      0.915       0.86      0.925      0.527
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
          8/21        0G   0.03906   0.02579  0.004082       165       416: 100% 461/461 [16:46<00:00,  2.18s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:43<00:00,  1.88s/it]
                     all       1413       5252      0.918      0.867      0.929      0.528
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
          9/21        0G   0.03828   0.02554  0.003729       170       416: 100% 461/461 [16:50<00:00,  2.19s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:41<00:00,  1.80s/it]
                     all       1413       5252      0.927      0.874      0.935      0.535
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         10/21        0G   0.03766   0.02512  0.003586       207       416: 100% 461/461 [16:48<00:00,  2.19s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:43<00:00,  1.87s/it]
                     all       1413       5252       0.93      0.876      0.936      0.543
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         11/21        0G   0.03699   0.02463  0.003394       172       416: 100% 461/461 [16:49<00:00,  2.19s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:38<00:00,  1.66s/it]
                     all       1413       5252       0.93      0.884      0.937      0.564
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         12/21        0G   0.03624   0.02419  0.003178       167       416: 100% 461/461 [16:50<00:00,  2.19s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:42<00:00,  1.86s/it]
                     all       1413       5252      0.938      0.884       0.94      0.573
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         13/21        0G   0.03566   0.02411  0.002874       148       416: 100% 461/461 [16:58<00:00,  2.21s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:41<00:00,  1.82s/it]
                     all       1413       5252      0.935      0.893       0.94      0.587
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         14/21        0G   0.03562   0.02415  0.003064       135       416: 100% 461/461 [16:47<00:00,  2.18s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:42<00:00,  1.86s/it]
                     all       1413       5252      0.936      0.896      0.945      0.562
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         15/21        0G   0.03522   0.02355  0.003013       116       416: 100% 461/461 [16:55<00:00,  2.20s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:39<00:00,  1.73s/it]
                     all       1413       5252      0.934      0.899      0.944      0.582
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         16/21        0G   0.03486   0.02347  0.003034       159       416: 100% 461/461 [16:43<00:00,  2.18s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:42<00:00,  1.87s/it]
                     all       1413       5252      0.936      0.898      0.946       0.56
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         17/21        0G   0.03438   0.02342  0.002871       160       416: 100% 461/461 [16:52<00:00,  2.20s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:38<00:00,  1.67s/it]
                     all       1413       5252      0.939      0.898      0.947      0.596
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         18/21        0G   0.03403   0.02307   0.00271       128       416: 100% 461/461 [16:45<00:00,  2.18s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:41<00:00,  1.78s/it]
                     all       1413       5252      0.938        0.9      0.948      0.602
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         19/21        0G   0.03389   0.02298   0.00266       206       416: 100% 461/461 [16:37<00:00,  2.16s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:43<00:00,  1.87s/it]
                     all       1413       5252      0.934      0.903      0.949      0.606
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         20/21        0G   0.03359   0.02297   0.00262       168       416: 100% 461/461 [16:47<00:00,  2.19s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:43<00:00,  1.91s/it]
                     all       1413       5252       0.94      0.906      0.949      0.599
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         21/21        0G   0.03171   0.02032  0.001696       180       416:   0% 2/461 [00:03<14:47,  1.93s/it]Exception in thread Thread-1:
    Traceback (most recent call last):
      File "/usr/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
      File "/usr/local/lib/python3.10/dist-packages/tensorboard/summary/writer/event_file_writer.py", line 244, in run
        self._run()
      File "/usr/local/lib/python3.10/dist-packages/tensorboard/summary/writer/event_file_writer.py", line 289, in _run
        self._record_writer.flush()
      File "/usr/local/lib/python3.10/dist-packages/tensorboard/summary/writer/record_writer.py", line 43, in flush
        self._writer.flush()
      File "/usr/local/lib/python3.10/dist-packages/tensorflow/python/lib/io/file_io.py", line 221, in flush
        self._writable_file.flush()
    tensorflow.python.framework.errors_impl.FailedPreconditionError: /content/drive/MyDrive/yolov5s_results1/events.out.tfevents.1698117115.ab8d677ea5e9.6679.0; Transport endpoint is not connected
         21/21        0G   0.03317   0.02264  0.002439       151       416: 100% 461/461 [16:50<00:00,  2.19s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:43<00:00,  1.90s/it]
                     all       1413       5252      0.943      0.905      0.951      0.612
    Traceback (most recent call last):
      File "/content/yolov5/train.py", line 642, in <module>
        main(opt)
      File "/content/yolov5/train.py", line 537, in main
        train(opt.hyp, opt, device, callbacks)
      File "/content/yolov5/train.py", line 384, in train
        callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
      File "/content/yolov5/utils/callbacks.py", line 71, in run
        logger['callback'](*args, **kwargs)
      File "/content/yolov5/utils/loggers/__init__.py", line 144, in on_fit_epoch_end
        self.tb.add_scalar(k, v, epoch)
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/tensorboard/writer.py", line 387, in add_scalar
        self._get_file_writer().add_summary(summary, global_step, walltime)
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/tensorboard/writer.py", line 109, in add_summary
        self.add_event(event, global_step, walltime)
      File "/usr/local/lib/python3.10/dist-packages/torch/utils/tensorboard/writer.py", line 94, in add_event
        self.event_writer.add_event(event)
      File "/usr/local/lib/python3.10/dist-packages/tensorboard/summary/writer/event_file_writer.py", line 117, in add_event
        self._async_writer.write(event.SerializeToString())
      File "/usr/local/lib/python3.10/dist-packages/tensorboard/summary/writer/event_file_writer.py", line 171, in write
        self._check_worker_status()
      File "/usr/local/lib/python3.10/dist-packages/tensorboard/summary/writer/event_file_writer.py", line 212, in _check_worker_status
        raise exception
      File "/usr/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
      File "/usr/local/lib/python3.10/dist-packages/tensorboard/summary/writer/event_file_writer.py", line 244, in run
        self._run()
      File "/usr/local/lib/python3.10/dist-packages/tensorboard/summary/writer/event_file_writer.py", line 289, in _run
        self._record_writer.flush()
      File "/usr/local/lib/python3.10/dist-packages/tensorboard/summary/writer/record_writer.py", line 43, in flush
        self._writer.flush()
      File "/usr/local/lib/python3.10/dist-packages/tensorflow/python/lib/io/file_io.py", line 221, in flush
        self._writable_file.flush()
    tensorflow.python.framework.errors_impl.FailedPreconditionError: /content/drive/MyDrive/yolov5s_results1/events.out.tfevents.1698117115.ab8d677ea5e9.6679.0; Transport endpoint is not connected
    CPU times: user 3min 4s, sys: 24.9 s, total: 3min 29s
    Wall time: 4h 21min 39s
    


```python
#학습시키기
%%time
%cd /content/yolov5/
!python train.py --img 416 --batch 32 --epochs 10 --data {dataset.location}/data.yaml \
--cfg ./models/yolov5s.yaml --weights '/content/drive/MyDrive/yolov5s_results1/weights/best.pt' --project /content/drive/MyDrive/ --name yolov5s_results2
```

    /content/yolov5
    2023-10-30 13:31:00.419474: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-30 13:31:01.418771: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    [34m[1mtrain: [0mweights=/content/drive/MyDrive/yolov5s_results1/weights/best.pt, cfg=./models/yolov5s.yaml, data=/content/yolov5/Hard-Hat-Workers-13/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=10, batch_size=32, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=/content/drive/MyDrive/, name=yolov5s_results2, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
    [34m[1mgithub: [0m⚠️ YOLOv5 is out of date by 595 commits. Use `git pull` or `git clone https://github.com/ultralytics/yolov5` to update.
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    [34m[1mhyperparameters: [0mlr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
    [34m[1mWeights & Biases: [0mrun 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir /content/drive/MyDrive', view at http://localhost:6006/
    Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
    100% 755k/755k [00:00<00:00, 13.7MB/s]
    Overriding model.yaml nc=80 with nc=2
    
                     from  n    params  module                                  arguments                     
      0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
      1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
      2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
      3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
      4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
      5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
      6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
      7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
      8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
      9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
     10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
     14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
     18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
     21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
     24      [17, 20, 23]  1     18879  models.yolo.Detect                      [2, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
    YOLOv5s summary: 270 layers, 7025023 parameters, 7025023 gradients, 16.0 GFLOPs
    
    Transferred 348/349 items from /content/drive/MyDrive/yolov5s_results1/weights/best.pt
    Scaled weight_decay = 0.0005
    [34m[1moptimizer:[0m SGD with parameter groups 57 weight (no decay), 60 weight, 60 bias
    /content/drive/MyDrive/yolov5s_results1/weights/best.pt has been trained for 19 epochs. Fine-tuning for 10 more epochs.
    [34m[1malbumentations: [0mBlur(always_apply=False, p=0.01, blur_limit=(3, 7)), MedianBlur(always_apply=False, p=0.01, blur_limit=(3, 7)), ToGray(always_apply=False, p=0.01), CLAHE(always_apply=False, p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    [34m[1mtrain: [0mScanning '/content/yolov5/Hard-Hat-Workers-13/train/labels' images and labels...14748 found, 0 missing, 1 empty, 0 corrupt: 100% 14748/14748 [00:01<00:00, 9405.13it/s]
    [34m[1mtrain: [0mNew cache created: /content/yolov5/Hard-Hat-Workers-13/train/labels.cache
    [34m[1mval: [0mScanning '/content/yolov5/Hard-Hat-Workers-13/valid/labels' images and labels...1413 found, 0 missing, 0 empty, 0 corrupt: 100% 1413/1413 [00:00<00:00, 7450.83it/s]
    [34m[1mval: [0mNew cache created: /content/yolov5/Hard-Hat-Workers-13/valid/labels.cache
    Plotting labels to /content/drive/MyDrive/yolov5s_results2/labels.jpg... 
    
    [34m[1mAutoAnchor: [0m5.88 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
    Image sizes 416 train, 416 val
    Using 8 dataloader workers
    Logging results to [1m/content/drive/MyDrive/yolov5s_results2[0m
    Starting training for 29 epochs...
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         20/28        0G   0.03309   0.02243  0.002313       162       416: 100% 461/461 [18:50<00:00,  2.45s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:50<00:00,  2.21s/it]
                     all       1413       5252      0.938      0.908      0.951      0.609
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         21/28        0G   0.03361   0.02268  0.002484       165       416: 100% 461/461 [19:03<00:00,  2.48s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:46<00:00,  2.02s/it]
                     all       1413       5252      0.935      0.907      0.948      0.594
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         22/28        0G   0.03337   0.02273  0.002388       170       416: 100% 461/461 [19:44<00:00,  2.57s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:42<00:00,  1.86s/it]
                     all       1413       5252      0.941      0.901      0.951      0.604
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         23/28        0G   0.03319   0.02255  0.002374       207       416: 100% 461/461 [18:13<00:00,  2.37s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:43<00:00,  1.91s/it]
                     all       1413       5252      0.941        0.9       0.95      0.602
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         24/28        0G   0.03275   0.02216  0.002269       172       416: 100% 461/461 [18:12<00:00,  2.37s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:40<00:00,  1.76s/it]
                     all       1413       5252       0.94      0.906       0.95      0.611
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         25/28        0G   0.03243   0.02192  0.002111       167       416: 100% 461/461 [18:16<00:00,  2.38s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:42<00:00,  1.86s/it]
                     all       1413       5252      0.948      0.903      0.952      0.609
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         26/28        0G   0.03207   0.02194  0.001944       148       416: 100% 461/461 [18:32<00:00,  2.41s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:46<00:00,  2.01s/it]
                     all       1413       5252      0.946      0.905       0.95      0.612
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         27/28        0G   0.03194   0.02194  0.001943       135       416: 100% 461/461 [18:39<00:00,  2.43s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:47<00:00,  2.07s/it]
                     all       1413       5252      0.939      0.913      0.952      0.612
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         28/28        0G   0.03173   0.02144   0.00188       116       416: 100% 461/461 [18:24<00:00,  2.40s/it]
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:40<00:00,  1.77s/it]
                     all       1413       5252      0.942      0.908      0.951      0.616
    
    9 epochs completed in 2.912 hours.
    Optimizer stripped from /content/drive/MyDrive/yolov5s_results2/weights/last.pt, 14.3MB
    Optimizer stripped from /content/drive/MyDrive/yolov5s_results2/weights/best.pt, 14.3MB
    
    Validating /content/drive/MyDrive/yolov5s_results2/weights/best.pt...
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:40<00:00,  1.75s/it]
                     all       1413       5252      0.942      0.908      0.951      0.616
                    head       1413       1339      0.929      0.904      0.937      0.607
                  helmet       1413       3913      0.955      0.912      0.965      0.624
    Results saved to [1m/content/drive/MyDrive/yolov5s_results2[0m
    CPU times: user 1min 41s, sys: 14.2 s, total: 1min 55s
    Wall time: 2h 55min 55s
    


```python
#결과 폴더를 압축해서 구글 드라이브에 저장
!zip -r /content/drive/MyDrive/result2.zip /content/yolov5/runs/train/yolov5s_results/
```

      adding: content/yolov5/runs/train/yolov5s_results/ (stored 0%)
      adding: content/yolov5/runs/train/yolov5s_results/weights/ (stored 0%)
      adding: content/yolov5/runs/train/yolov5s_results/weights/best.pt (deflated 10%)
      adding: content/yolov5/runs/train/yolov5s_results/weights/last.pt (deflated 10%)
      adding: content/yolov5/runs/train/yolov5s_results/R_curve.png (deflated 11%)
      adding: content/yolov5/runs/train/yolov5s_results/PR_curve.png (deflated 20%)
      adding: content/yolov5/runs/train/yolov5s_results/val_batch2_pred.jpg (deflated 8%)
      adding: content/yolov5/runs/train/yolov5s_results/events.out.tfevents.1666505440.7bcadb40cced.486.0 (deflated 23%)
      adding: content/yolov5/runs/train/yolov5s_results/val_batch1_labels.jpg (deflated 8%)
      adding: content/yolov5/runs/train/yolov5s_results/F1_curve.png (deflated 9%)
      adding: content/yolov5/runs/train/yolov5s_results/labels.jpg (deflated 29%)
      adding: content/yolov5/runs/train/yolov5s_results/val_batch2_labels.jpg (deflated 8%)
      adding: content/yolov5/runs/train/yolov5s_results/val_batch1_pred.jpg (deflated 8%)
      adding: content/yolov5/runs/train/yolov5s_results/confusion_matrix.png (deflated 16%)
      adding: content/yolov5/runs/train/yolov5s_results/val_batch0_labels.jpg (deflated 6%)
      adding: content/yolov5/runs/train/yolov5s_results/val_batch0_pred.jpg (deflated 6%)
      adding: content/yolov5/runs/train/yolov5s_results/opt.yaml (deflated 43%)
      adding: content/yolov5/runs/train/yolov5s_results/train_batch1.jpg (deflated 4%)
      adding: content/yolov5/runs/train/yolov5s_results/results.csv (deflated 83%)
      adding: content/yolov5/runs/train/yolov5s_results/train_batch2.jpg (deflated 3%)
      adding: content/yolov5/runs/train/yolov5s_results/hyp.yaml (deflated 45%)
      adding: content/yolov5/runs/train/yolov5s_results/train_batch0.jpg (deflated 4%)
      adding: content/yolov5/runs/train/yolov5s_results/results.png (deflated 6%)
      adding: content/yolov5/runs/train/yolov5s_results/P_curve.png (deflated 11%)
      adding: content/yolov5/runs/train/yolov5s_results/labels_correlogram.jpg (deflated 32%)
    

# 테스트 데이터 검증


```python
# data.yaml에 test 경로 추가해야 함
!python /content/yolov5/val.py --task test --data /content/yolov5/Hard-Hat-Workers-13/data.yaml \
--weights /content/drive/MyDrive/yolov5s_results2/weights/best.pt --img 416 --save-txt --save-conf
```

    [34m[1mval: [0mdata=/content/yolov5/Hard-Hat-Workers-13/data.yaml, weights=['/content/drive/MyDrive/yolov5s_results2/weights/best.pt'], batch_size=32, imgsz=416, conf_thres=0.001, iou_thres=0.6, task=test, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=True, save_hybrid=False, save_conf=True, save_json=False, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    [34m[1mtest: [0mScanning '/content/yolov5/Hard-Hat-Workers-13/test/labels' images and labels...706 found, 0 missing, 0 empty, 0 corrupt: 100% 706/706 [00:00<00:00, 8045.06it/s]
    [34m[1mtest: [0mNew cache created: /content/yolov5/Hard-Hat-Workers-13/test/labels.cache
                   Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:33<00:00,  1.44s/it]
                     all        706       2641       0.94      0.913      0.959      0.622
                    head        706        726      0.938      0.915      0.952      0.627
                  helmet        706       1915      0.942      0.912      0.967      0.618
    Speed: 0.2ms pre-process, 33.1ms inference, 0.8ms NMS per image at shape (32, 3, 416, 416)
    Results saved to [1mruns/val/exp4[0m
    706 labels saved to runs/val/exp4/labels
    


```python
#학습된 가중치로 test이미지 탐색
%cd /content/yolov5/
!python detect.py --weights /content/drive/MyDrive/yolov5s_results2/weights/best.pt --img 416 --conf 0.4 --source ./Hard-Hat-Workers-13/test/images --save-txt
```

    /content/yolov5
    [34m[1mdetect: [0mweights=['/content/drive/MyDrive/yolov5s_results2/weights/best.pt'], source=./Hard-Hat-Workers-13/test/images, data=data/coco128.yaml, imgsz=[416, 416], conf_thres=0.4, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=True, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    image 1/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005298_jpg.rf.8735bf2b956d26c1f6ef23b09c93fe97.jpg: 416x416 8 helmets, Done. (0.048s)
    image 2/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005299_jpg.rf.7e8cbce0496f979e698687003725214c.jpg: 416x416 2 helmets, Done. (0.035s)
    image 3/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005300_jpg.rf.2c31335b78e142e0ab5b0cabb142091b.jpg: 416x416 1 helmet, Done. (0.037s)
    image 4/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005301_jpg.rf.fdc6cb9fe5fe9a5132e80033cb81f049.jpg: 416x416 5 heads, Done. (0.034s)
    image 5/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005302_jpg.rf.e65eb388c2cfb4fb39130ed61ff7c283.jpg: 416x416 1 helmet, Done. (0.034s)
    image 6/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005303_jpg.rf.afcdb38c41ef7faad117a0cb8a8f3f3d.jpg: 416x416 5 helmets, Done. (0.034s)
    image 7/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005304_jpg.rf.5b58b8e11864d2382bcc58af36b34f5a.jpg: 416x416 4 helmets, Done. (0.036s)
    image 8/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005305_jpg.rf.ed65c8960d04e48658061aacd174b32f.jpg: 416x416 1 helmet, Done. (0.035s)
    image 9/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005306_jpg.rf.17db387e900596dda895e70f81795501.jpg: 416x416 1 helmet, Done. (0.036s)
    image 10/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005307_jpg.rf.ea4cdc4e607af2519b0c1342c2111499.jpg: 416x416 2 helmets, Done. (0.035s)
    image 11/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005308_jpg.rf.b458e5186e4d28eada03fd465498f8ff.jpg: 416x416 1 head, 3 helmets, Done. (0.035s)
    image 12/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005309_jpg.rf.fcb932d4906635e8e9b9cf167ba96f82.jpg: 416x416 2 helmets, Done. (0.036s)
    image 13/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005310_jpg.rf.115019400a78de9e817046c79df56d6e.jpg: 416x416 1 helmet, Done. (0.043s)
    image 14/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005311_jpg.rf.5a370192903e1db275393bc7e66d0894.jpg: 416x416 3 helmets, Done. (0.043s)
    image 15/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005312_jpg.rf.34a5c60c8b7aa488555c910796366d52.jpg: 416x416 1 helmet, Done. (0.044s)
    image 16/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005313_jpg.rf.13937be5913ae443bf4b827447d189a5.jpg: 416x416 1 helmet, Done. (0.036s)
    image 17/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005314_jpg.rf.21cf66ba03b880449a5b18e014f9f025.jpg: 416x416 3 helmets, Done. (0.036s)
    image 18/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005315_jpg.rf.54b172cef11b2a515df3172690872c75.jpg: 416x416 2 helmets, Done. (0.036s)
    image 19/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005316_jpg.rf.4fdc14267fb6e1c6f8e85126bdddb391.jpg: 416x416 4 helmets, Done. (0.032s)
    image 20/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005317_jpg.rf.ef111dd929bbd458073fe46e89a42c81.jpg: 416x416 6 helmets, Done. (0.033s)
    image 21/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005318_jpg.rf.5af3424f713ee8f8efe5eeb056851328.jpg: 416x416 4 helmets, Done. (0.034s)
    image 22/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005319_jpg.rf.9bd13cab215f7d1c212fab5260c87265.jpg: 416x416 1 helmet, Done. (0.032s)
    image 23/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005320_jpg.rf.6d61785eb4ef0f82a15e32c903b729bd.jpg: 416x416 1 helmet, Done. (0.034s)
    image 24/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005321_jpg.rf.8c3438096deb0b72b47446cf9e3b2086.jpg: 416x416 2 helmets, Done. (0.031s)
    image 25/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005322_jpg.rf.a8f5d9eff4ef2647d498e8e335f375e3.jpg: 416x416 1 helmet, Done. (0.032s)
    image 26/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005323_jpg.rf.73ab45f5e202214ae52fec52af75fed7.jpg: 416x416 3 helmets, Done. (0.033s)
    image 27/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005324_jpg.rf.5740368cc94b4c38192f9ffb7f119a5a.jpg: 416x416 1 helmet, Done. (0.035s)
    image 28/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005325_jpg.rf.62d1a5d60c29fb43ee6cfd188b2562d8.jpg: 416x416 8 heads, Done. (0.035s)
    image 29/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005326_jpg.rf.16809a15d081346ffa231707d577e04f.jpg: 416x416 Done. (0.033s)
    image 30/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005327_jpg.rf.6d37171a746ff27480282d0bdf124005.jpg: 416x416 4 helmets, Done. (0.035s)
    image 31/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005328_jpg.rf.25280ba69ec6c6f709f1e9a555da7ece.jpg: 416x416 1 helmet, Done. (0.032s)
    image 32/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005329_jpg.rf.bc7f492c322575cdc2e929d9e7b0894f.jpg: 416x416 1 helmet, Done. (0.035s)
    image 33/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005330_jpg.rf.bb41c0b093af9cd26629db9610393fac.jpg: 416x416 3 helmets, Done. (0.032s)
    image 34/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005331_jpg.rf.b84f88520c520156edddd6cef2ca0255.jpg: 416x416 12 helmets, Done. (0.034s)
    image 35/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005332_jpg.rf.b5ce6e98f888e20fb18ccb88c705b525.jpg: 416x416 2 helmets, Done. (0.033s)
    image 36/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005333_jpg.rf.cd1a6f174a7c8a6235f376b4058dcfc0.jpg: 416x416 4 helmets, Done. (0.034s)
    image 37/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005334_jpg.rf.d2e15a6e2d9b140886bbcfd61b03799d.jpg: 416x416 2 helmets, Done. (0.032s)
    image 38/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005335_jpg.rf.94614e153b0de65c7a07e118359caf16.jpg: 416x416 1 helmet, Done. (0.034s)
    image 39/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005336_jpg.rf.2707b52d4a8955172a182f7970b232f6.jpg: 416x416 2 helmets, Done. (0.035s)
    image 40/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005337_jpg.rf.288be659c1cf3a2930e8f095a4e6c193.jpg: 416x416 3 helmets, Done. (0.033s)
    image 41/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005338_jpg.rf.9cb5fd0b628585fd4c933e929edd596d.jpg: 416x416 2 helmets, Done. (0.036s)
    image 42/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005339_jpg.rf.d1426d03b9f16e745fb49229ecf2d85b.jpg: 416x416 1 helmet, Done. (0.032s)
    image 43/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005340_jpg.rf.86e5a442381dab86bce0627d63a8a80c.jpg: 416x416 1 helmet, Done. (0.034s)
    image 44/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005341_jpg.rf.a184cc9658a97098d0c019554a519ab8.jpg: 416x416 2 helmets, Done. (0.033s)
    image 45/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005342_jpg.rf.c3c9eaf856fd5fc0364e3e60bd1f214a.jpg: 416x416 3 helmets, Done. (0.033s)
    image 46/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005343_jpg.rf.d032e5d57a5d8ca52ad0af9595a734a1.jpg: 416x416 5 helmets, Done. (0.037s)
    image 47/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005344_jpg.rf.ca25db008d2526fe6566421d5e8431ab.jpg: 416x416 2 helmets, Done. (0.035s)
    image 48/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005345_jpg.rf.be2a4fb55664eca46d695961ba8b1e78.jpg: 416x416 1 helmet, Done. (0.033s)
    image 49/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005346_jpg.rf.eb0ac67b7842d3c3d7f7a05044d541b2.jpg: 416x416 1 head, 8 helmets, Done. (0.035s)
    image 50/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005347_jpg.rf.8ab0292288ba3b43893869108fc403f6.jpg: 416x416 2 helmets, Done. (0.036s)
    image 51/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005348_jpg.rf.4f646d58c383a703ededaecb8bfa0469.jpg: 416x416 3 helmets, Done. (0.032s)
    image 52/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005349_jpg.rf.d4ab30e52a760d6aa41b17354082db6b.jpg: 416x416 1 helmet, Done. (0.035s)
    image 53/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005350_jpg.rf.c00f3ebc44cc370ddcd419dada7a2a06.jpg: 416x416 1 helmet, Done. (0.034s)
    image 54/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005351_jpg.rf.da148a770e93504daa79f244ca293b53.jpg: 416x416 2 helmets, Done. (0.034s)
    image 55/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005352_jpg.rf.8226fb3dbbaebdd2a80ecfb3b32b26d6.jpg: 416x416 3 helmets, Done. (0.035s)
    image 56/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005353_jpg.rf.9c0a98eab79571f74332373de062fe4e.jpg: 416x416 1 helmet, Done. (0.039s)
    image 57/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005354_jpg.rf.46270b396b2b874357c301fdfff23a4e.jpg: 416x416 1 helmet, Done. (0.039s)
    image 58/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005355_jpg.rf.193f432b1f7e45980e3701313cecd00d.jpg: 416x416 8 helmets, Done. (0.036s)
    image 59/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005356_jpg.rf.2bcc5cdbf05b402b854d59613b6d8f95.jpg: 416x416 3 helmets, Done. (0.033s)
    image 60/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005357_jpg.rf.fb6dda6f625a6f5fabb0772da37d9f80.jpg: 416x416 1 head, 1 helmet, Done. (0.034s)
    image 61/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005358_jpg.rf.35fdd6ee9c50a20d2cea430f183810a8.jpg: 416x416 1 head, 6 helmets, Done. (0.043s)
    image 62/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005359_jpg.rf.c1df757f7696fd925e01a162cc78aa1b.jpg: 416x416 2 helmets, Done. (0.038s)
    image 63/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005360_jpg.rf.a79da3d410435e224ca57d2ff4f48849.jpg: 416x416 7 heads, 5 helmets, Done. (0.041s)
    image 64/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005361_jpg.rf.54201c26336e920e4d3cfc0aba0a7668.jpg: 416x416 2 helmets, Done. (0.036s)
    image 65/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005362_jpg.rf.b67c8bbf8e71d662e2e7e02d6ed92ac7.jpg: 416x416 10 helmets, Done. (0.037s)
    image 66/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005363_jpg.rf.67764bf818143438411935b84fc7da85.jpg: 416x416 4 helmets, Done. (0.039s)
    image 67/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005364_jpg.rf.252a1dd9f902f4e0c8d04059142ac762.jpg: 416x416 3 helmets, Done. (0.038s)
    image 68/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005365_jpg.rf.5aa50bb53040b793323fc7b1d7d27b19.jpg: 416x416 6 helmets, Done. (0.038s)
    image 69/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005366_jpg.rf.f1e5bd0b594be749189a07692ab516f9.jpg: 416x416 1 helmet, Done. (0.034s)
    image 70/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005367_jpg.rf.6e4193229648f03de049fabcc277288c.jpg: 416x416 2 heads, Done. (0.037s)
    image 71/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005368_jpg.rf.90619a532d936c33c7cf8ad080edc61e.jpg: 416x416 4 helmets, Done. (0.035s)
    image 72/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005369_jpg.rf.e89cce35a4d69340afeaffba5d8edd5f.jpg: 416x416 3 helmets, Done. (0.034s)
    image 73/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005370_jpg.rf.23e172db40925efc446cc43c5d208122.jpg: 416x416 1 helmet, Done. (0.034s)
    image 74/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005371_jpg.rf.d1dad5bef2b27e0894b046d512554702.jpg: 416x416 5 heads, 2 helmets, Done. (0.033s)
    image 75/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005372_jpg.rf.68228d6d2b2c145a838c5fb4a0568519.jpg: 416x416 7 helmets, Done. (0.032s)
    image 76/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005373_jpg.rf.31f3670dce9fef5d88fccf0f8c7f9766.jpg: 416x416 6 helmets, Done. (0.032s)
    image 77/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005374_jpg.rf.cc5d1c3722e68b385f3b1108f97c9979.jpg: 416x416 2 helmets, Done. (0.040s)
    image 78/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005375_jpg.rf.33dd0d8e8c0caaee5fb494a93d306cb4.jpg: 416x416 1 helmet, Done. (0.038s)
    image 79/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005376_jpg.rf.06bf4fabfdb7a5edddd8d3db5351c05e.jpg: 416x416 1 head, 4 helmets, Done. (0.040s)
    image 80/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005377_jpg.rf.868defecc607d0308de58b8571f08f42.jpg: 416x416 1 helmet, Done. (0.036s)
    image 81/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005378_jpg.rf.86b59dca11081775251fa2cc52076d54.jpg: 416x416 1 head, 10 helmets, Done. (0.034s)
    image 82/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005379_jpg.rf.3065c9255c38e1ff5188f8f2aa70d62e.jpg: 416x416 9 heads, 1 helmet, Done. (0.039s)
    image 83/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005380_jpg.rf.abe81c479e819821b26580bf1d20c62e.jpg: 416x416 1 helmet, Done. (0.035s)
    image 84/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005381_jpg.rf.f5f42eb12ef5cf402bac6acb8e2b4160.jpg: 416x416 3 helmets, Done. (0.032s)
    image 85/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005382_jpg.rf.7b0584f7361eddb89fcb6cef40968f3e.jpg: 416x416 1 head, 6 helmets, Done. (0.033s)
    image 86/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005383_jpg.rf.4e92a61fb44ade668b7151b30c698d32.jpg: 416x416 6 heads, Done. (0.035s)
    image 87/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005384_jpg.rf.75f4f902f98060f9f26aa3ef6ca0cf83.jpg: 416x416 2 helmets, Done. (0.034s)
    image 88/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005385_jpg.rf.420ff1c02ed6740a82da1e4d8d9906aa.jpg: 416x416 7 heads, Done. (0.034s)
    image 89/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005386_jpg.rf.92917d883c327402471d771a214d0ce4.jpg: 416x416 8 heads, Done. (0.034s)
    image 90/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005387_jpg.rf.dace49250a38708b33c41436a0198c8a.jpg: 416x416 6 helmets, Done. (0.033s)
    image 91/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005388_jpg.rf.0802dafb12c5086a81ac59e7cc1c08a8.jpg: 416x416 3 helmets, Done. (0.036s)
    image 92/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005389_jpg.rf.2c90219192247f17586224f17264823b.jpg: 416x416 3 helmets, Done. (0.035s)
    image 93/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005390_jpg.rf.1ecd310154c5694f0a36f3b0f3ec29ca.jpg: 416x416 1 helmet, Done. (0.034s)
    image 94/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005391_jpg.rf.fdc10dc6bea1bdb5e2c3b4b8732414c4.jpg: 416x416 8 heads, Done. (0.033s)
    image 95/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005392_jpg.rf.72fe90add658ae6153c4d90917942267.jpg: 416x416 7 helmets, Done. (0.032s)
    image 96/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005393_jpg.rf.50109e16af2d4bb4024174d0f8df7e7f.jpg: 416x416 16 helmets, Done. (0.036s)
    image 97/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005394_jpg.rf.c01e4772e33fd2cbca96b36358bacae0.jpg: 416x416 1 helmet, Done. (0.033s)
    image 98/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005395_jpg.rf.1fc5e2381e7f5f527cccf9f0608c9b84.jpg: 416x416 4 helmets, Done. (0.038s)
    image 99/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005396_jpg.rf.08ed5f6a54e1c86be0e5b71a80a54de6.jpg: 416x416 1 helmet, Done. (0.037s)
    image 100/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005397_jpg.rf.47bdfd0f230f1dd4e1d0f6b391f14650.jpg: 416x416 5 helmets, Done. (0.038s)
    image 101/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005398_jpg.rf.0b9b6f33406aa0f6c4452f548af8d81c.jpg: 416x416 5 heads, 3 helmets, Done. (0.037s)
    image 102/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005399_jpg.rf.12b9aec3f61a835134f8b46176cb10ac.jpg: 416x416 2 helmets, Done. (0.039s)
    image 103/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005400_jpg.rf.31d48d08d796a3947593f18cc182050f.jpg: 416x416 2 helmets, Done. (0.036s)
    image 104/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005401_jpg.rf.e71bfec530da9138a097365a59e3466d.jpg: 416x416 5 helmets, Done. (0.042s)
    image 105/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005402_jpg.rf.1da0d2f21d2458443e3a1738f804e20c.jpg: 416x416 2 helmets, Done. (0.037s)
    image 106/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005403_jpg.rf.080420f275e6cb1dbfd3b61d54f550b4.jpg: 416x416 5 helmets, Done. (0.037s)
    image 107/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005404_jpg.rf.aaa4e8df776f1382feeab513ebe32e1a.jpg: 416x416 8 heads, Done. (0.034s)
    image 108/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005405_jpg.rf.5937db199fb8a700594ab3f449623224.jpg: 416x416 1 helmet, Done. (0.036s)
    image 109/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005406_jpg.rf.740fbd9b1b04a9d21dc89d35fa68fbd1.jpg: 416x416 2 helmets, Done. (0.038s)
    image 110/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005407_jpg.rf.e6b68417a6381db4850b991d46b9e6d0.jpg: 416x416 2 heads, 3 helmets, Done. (0.037s)
    image 111/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005408_jpg.rf.1880b8a53cac4706dbb12afc4a6b1414.jpg: 416x416 2 helmets, Done. (0.034s)
    image 112/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005409_jpg.rf.1102619150c1d273881dd1741ff24fc9.jpg: 416x416 5 helmets, Done. (0.036s)
    image 113/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005410_jpg.rf.7128627f732dcaf173d2dbfa51ff99c4.jpg: 416x416 2 helmets, Done. (0.037s)
    image 114/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005411_jpg.rf.e51e6917a309bb222476374a99273747.jpg: 416x416 3 helmets, Done. (0.038s)
    image 115/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005412_jpg.rf.a9fca9d8750aaf4932f5dd9bda732693.jpg: 416x416 4 heads, Done. (0.035s)
    image 116/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005413_jpg.rf.2e7f7f6b8b91aab1754772922c0958db.jpg: 416x416 2 helmets, Done. (0.040s)
    image 117/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005414_jpg.rf.2721a2e4b20b8d4bbfd14fb3c0d50dd7.jpg: 416x416 4 heads, Done. (0.041s)
    image 118/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005415_jpg.rf.c4d4df4b988f28e41e496739744ea1b0.jpg: 416x416 5 helmets, Done. (0.040s)
    image 119/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005416_jpg.rf.287ebe5e342bf46887cb330704eb59a4.jpg: 416x416 2 helmets, Done. (0.037s)
    image 120/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005417_jpg.rf.3ac8fcc5c9aa4f3ffbff0bd6bd5b406e.jpg: 416x416 5 heads, Done. (0.036s)
    image 121/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005418_jpg.rf.82f3a90b54e0e48463d99340c26d9f32.jpg: 416x416 3 helmets, Done. (0.033s)
    image 122/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005419_jpg.rf.201064cef8d3c25258ad209261b70e34.jpg: 416x416 14 helmets, Done. (0.033s)
    image 123/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005420_jpg.rf.0677158315dd53f409cb1bb3c51f92d4.jpg: 416x416 9 heads, 1 helmet, Done. (0.040s)
    image 124/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005421_jpg.rf.69e3e738577377effa763503e61550d4.jpg: 416x416 2 helmets, Done. (0.041s)
    image 125/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005422_jpg.rf.e796a1d4812258857573820aac67844d.jpg: 416x416 2 helmets, Done. (0.040s)
    image 126/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005423_jpg.rf.3a3ca64f87ff8f36f7a3c9a1e5bbf859.jpg: 416x416 3 helmets, Done. (0.037s)
    image 127/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005424_jpg.rf.dd6a8d00336350e8a0dcc105909cf770.jpg: 416x416 5 heads, Done. (0.033s)
    image 128/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005425_jpg.rf.9bd3089bdb406dde407442fb5a2563c2.jpg: 416x416 8 helmets, Done. (0.034s)
    image 129/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005426_jpg.rf.2e0ad65c275f1ad3e9b0a495d729db52.jpg: 416x416 7 heads, Done. (0.035s)
    image 130/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005427_jpg.rf.e5452e133a40ddbd9a683665adc2b1ca.jpg: 416x416 1 helmet, Done. (0.034s)
    image 131/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005428_jpg.rf.e09f6f90b66aeac61fe8d3e4bad7c63b.jpg: 416x416 7 heads, 3 helmets, Done. (0.036s)
    image 132/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005429_jpg.rf.d4468cdec52cf8333d0d1b5dd9837511.jpg: 416x416 1 head, 8 helmets, Done. (0.034s)
    image 133/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005430_jpg.rf.250deb366c88ffb920e3c39748df564e.jpg: 416x416 1 head, 1 helmet, Done. (0.034s)
    image 134/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005431_jpg.rf.65b9f4549bd0bb4293dc16873b8c3101.jpg: 416x416 5 helmets, Done. (0.035s)
    image 135/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005432_jpg.rf.c2f626ff42bdd4089eb05ed2e964b0dc.jpg: 416x416 8 helmets, Done. (0.034s)
    image 136/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005433_jpg.rf.ab4aeceb3fdfc21c71aa0076bdd42a44.jpg: 416x416 2 helmets, Done. (0.034s)
    image 137/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005434_jpg.rf.beb549ce16a8790ee71adf27910c9c63.jpg: 416x416 1 helmet, Done. (0.033s)
    image 138/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005435_jpg.rf.22830b280d725a3bc5f00f508953256b.jpg: 416x416 2 heads, 6 helmets, Done. (0.033s)
    image 139/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005436_jpg.rf.f58b47c9487a9ac3cbe6ddfe5922f191.jpg: 416x416 1 helmet, Done. (0.035s)
    image 140/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005437_jpg.rf.c3ad4490999288054dca4672b6f15de7.jpg: 416x416 3 helmets, Done. (0.038s)
    image 141/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005438_jpg.rf.ba05832548d8e932af7f7f63484c157a.jpg: 416x416 4 helmets, Done. (0.038s)
    image 142/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005439_jpg.rf.f2de8d6d5dba45845ac63c901d34b00b.jpg: 416x416 4 helmets, Done. (0.037s)
    image 143/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005440_jpg.rf.c3c531be5939b1dec6cff6f67649d622.jpg: 416x416 8 helmets, Done. (0.033s)
    image 144/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005441_jpg.rf.ea928014773b11c685e7326ed304da33.jpg: 416x416 2 helmets, Done. (0.038s)
    image 145/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005442_jpg.rf.974cc70fac8fab35bb1610c4f834b832.jpg: 416x416 11 helmets, Done. (0.034s)
    image 146/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005443_jpg.rf.b28b4c844aed3ff877aca1ac0cddf2ef.jpg: 416x416 7 helmets, Done. (0.033s)
    image 147/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005444_jpg.rf.73bdf9a176205c17affffd5707e2819b.jpg: 416x416 3 helmets, Done. (0.035s)
    image 148/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005445_jpg.rf.9bf0092eb6efd2c82c38f98a16174b29.jpg: 416x416 14 helmets, Done. (0.033s)
    image 149/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005446_jpg.rf.adfc66d5d67c39e6970dad4ef024ad4e.jpg: 416x416 1 head, 3 helmets, Done. (0.032s)
    image 150/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005447_jpg.rf.1dbc29f522b0099bad1acc7f1af7baca.jpg: 416x416 3 helmets, Done. (0.033s)
    image 151/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005448_jpg.rf.595d860fd925ffe230a38b027a23c4b6.jpg: 416x416 3 helmets, Done. (0.032s)
    image 152/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005449_jpg.rf.667569d6762fb0f9b01b8284566f313e.jpg: 416x416 1 helmet, Done. (0.033s)
    image 153/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005450_jpg.rf.bc6cd1e2a4cbff83b84bd10fb21b5f27.jpg: 416x416 15 heads, Done. (0.034s)
    image 154/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005451_jpg.rf.a8f46e4b5f187e6e91c1099a6b6acc32.jpg: 416x416 7 helmets, Done. (0.034s)
    image 155/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005452_jpg.rf.de10d47219daebdb3178a6c56ee4f1e2.jpg: 416x416 6 helmets, Done. (0.033s)
    image 156/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005453_jpg.rf.08423f7ac575e2e19520f501a715f212.jpg: 416x416 1 helmet, Done. (0.034s)
    image 157/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005454_jpg.rf.54f98acecd20e364111d49af180fb612.jpg: 416x416 6 helmets, Done. (0.035s)
    image 158/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005455_jpg.rf.c447748b2269ae53b7e2f797f8ed962b.jpg: 416x416 8 helmets, Done. (0.033s)
    image 159/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005456_jpg.rf.44ff2bea1384208b08a9c01263075339.jpg: 416x416 5 heads, 1 helmet, Done. (0.034s)
    image 160/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005457_jpg.rf.56b83de1948935994ca98c5bd380ea81.jpg: 416x416 8 heads, 5 helmets, Done. (0.032s)
    image 161/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005458_jpg.rf.cebeae3077003360113a05f75f123a2a.jpg: 416x416 3 heads, 4 helmets, Done. (0.033s)
    image 162/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005459_jpg.rf.2916ca54793171c49f4cf11ce494678e.jpg: 416x416 4 helmets, Done. (0.032s)
    image 163/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005460_jpg.rf.d844c4402a87361897b4a5a01d0e518d.jpg: 416x416 3 helmets, Done. (0.032s)
    image 164/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005461_jpg.rf.cb8804fa866d8047779e4ffc97e5ad26.jpg: 416x416 1 helmet, Done. (0.032s)
    image 165/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005462_jpg.rf.ab94968610f6187ac7a5ddfc89b4c706.jpg: 416x416 3 heads, Done. (0.033s)
    image 166/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005463_jpg.rf.20ebf32bfa5c8e3487ad840dc1c34986.jpg: 416x416 3 helmets, Done. (0.036s)
    image 167/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005464_jpg.rf.1cdc11c33355e8ed168e238d8255921f.jpg: 416x416 2 helmets, Done. (0.034s)
    image 168/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005465_jpg.rf.13d6c14b6582b5cb58aed8d901f4fe00.jpg: 416x416 1 head, 2 helmets, Done. (0.036s)
    image 169/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005466_jpg.rf.4d0ad4d127e81875b1a64e741d00d6df.jpg: 416x416 3 helmets, Done. (0.036s)
    image 170/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005467_jpg.rf.f5773d1c21c719a0e428667117e4efde.jpg: 416x416 4 helmets, Done. (0.038s)
    image 171/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005468_jpg.rf.7e961822de6f7761a8daf89bc7bd5826.jpg: 416x416 1 helmet, Done. (0.035s)
    image 172/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005469_jpg.rf.f621e311f0f986ac1f081b67296eac74.jpg: 416x416 1 helmet, Done. (0.037s)
    image 173/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005470_jpg.rf.ee2cc87c684e76ce39bda831da6d60d5.jpg: 416x416 2 helmets, Done. (0.036s)
    image 174/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005471_jpg.rf.ea65bf50ff9154e287315320c2ef253e.jpg: 416x416 3 helmets, Done. (0.036s)
    image 175/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005472_jpg.rf.3ee42db131501d94a5cf004c2db0663e.jpg: 416x416 9 heads, Done. (0.035s)
    image 176/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005473_jpg.rf.5f2a6cc7244cb04e7a5148798a3d79c3.jpg: 416x416 1 helmet, Done. (0.035s)
    image 177/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005474_jpg.rf.c4db00bf57dccdba393429c01211c5ca.jpg: 416x416 1 helmet, Done. (0.034s)
    image 178/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005475_jpg.rf.2ece692e7ae3064809190f171b4a1aff.jpg: 416x416 2 helmets, Done. (0.033s)
    image 179/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005476_jpg.rf.a41d9a090c6787071cee97d5e2d98ce8.jpg: 416x416 1 helmet, Done. (0.035s)
    image 180/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005477_jpg.rf.89fa1d4f8286f5b990c5ce4446855b0e.jpg: 416x416 10 heads, Done. (0.037s)
    image 181/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005478_jpg.rf.dac4eef77e906e5358eb84f6759f0c37.jpg: 416x416 1 helmet, Done. (0.034s)
    image 182/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005479_jpg.rf.313a98957fbdb978445f66ed58c48c69.jpg: 416x416 5 helmets, Done. (0.040s)
    image 183/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005480_jpg.rf.19bd14f15f8842cf730434d54c5baaa1.jpg: 416x416 1 helmet, Done. (0.039s)
    image 184/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005481_jpg.rf.5c13de4b11c49f2db0fef5841035dda3.jpg: 416x416 2 helmets, Done. (0.036s)
    image 185/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005482_jpg.rf.f06b7c525d7cf8f9d065bae2cbb2f2ab.jpg: 416x416 1 helmet, Done. (0.036s)
    image 186/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005483_jpg.rf.e4a1e149d7e3a2cc14a0f224325e4aae.jpg: 416x416 4 helmets, Done. (0.039s)
    image 187/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005484_jpg.rf.48368386a3eac7274643a68076eca984.jpg: 416x416 11 heads, Done. (0.044s)
    image 188/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005485_jpg.rf.37baab3de5d74ac55ea782ed9a245c98.jpg: 416x416 1 helmet, Done. (0.042s)
    image 189/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005486_jpg.rf.f907a7404330a52f03321cb1529db2ee.jpg: 416x416 1 head, 10 helmets, Done. (0.038s)
    image 190/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005487_jpg.rf.cee4fb3ce204bc0d980fcce3137491fc.jpg: 416x416 2 helmets, Done. (0.032s)
    image 191/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005488_jpg.rf.c88bd3fae90abda9fa9aa3e9ef1f0615.jpg: 416x416 7 heads, Done. (0.036s)
    image 192/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005489_jpg.rf.1b3d473102401aa89936ac6e6835bd11.jpg: 416x416 1 helmet, Done. (0.033s)
    image 193/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005490_jpg.rf.91b476d8f7ddb3cedce909ac462a210f.jpg: 416x416 5 heads, Done. (0.034s)
    image 194/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005491_jpg.rf.c30c452ad71055c838a8c22eb1a9b6a5.jpg: 416x416 3 helmets, Done. (0.036s)
    image 195/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005492_jpg.rf.7c301db3fe93e16a899f544a0bbec57c.jpg: 416x416 2 helmets, Done. (0.035s)
    image 196/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005493_jpg.rf.d5148a647bb94aa631037fd1a7472380.jpg: 416x416 2 helmets, Done. (0.034s)
    image 197/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005494_jpg.rf.8f56d8930e97c103b2906e80ac2b6440.jpg: 416x416 1 helmet, Done. (0.033s)
    image 198/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005495_jpg.rf.776f971b4b5dab1dfa6c77f07a61f219.jpg: 416x416 5 heads, Done. (0.035s)
    image 199/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005496_jpg.rf.6d0486d64451dddeaf00108b30343eec.jpg: 416x416 3 heads, 1 helmet, Done. (0.034s)
    image 200/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005497_jpg.rf.de2071c9693344ba13bcf08bf9dcd637.jpg: 416x416 3 helmets, Done. (0.033s)
    image 201/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005498_jpg.rf.7b1bb6e179c769e2bee49f9dfc034c78.jpg: 416x416 Done. (0.032s)
    image 202/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005499_jpg.rf.de52ff0f9c9aea975abd716a8a628f14.jpg: 416x416 1 head, 3 helmets, Done. (0.031s)
    image 203/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005500_jpg.rf.6bfd1dbdccaa5d63a661157c8e88465d.jpg: 416x416 1 helmet, Done. (0.036s)
    image 204/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005501_jpg.rf.0c761e175be69cb8b4789db60aadf093.jpg: 416x416 6 helmets, Done. (0.038s)
    image 205/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005502_jpg.rf.281348d95a2da8aa897eb79886ce3051.jpg: 416x416 8 helmets, Done. (0.040s)
    image 206/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005503_jpg.rf.ea77615390c8f8aff052357d8b4de5cd.jpg: 416x416 1 helmet, Done. (0.035s)
    image 207/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005504_jpg.rf.8b7026b8d73dc26ab3e05ba0cf96ad9a.jpg: 416x416 3 helmets, Done. (0.034s)
    image 208/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005505_jpg.rf.b97dd4e1c3e6efd0b7afc66efbf2a569.jpg: 416x416 4 helmets, Done. (0.039s)
    image 209/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005506_jpg.rf.d5f18f05126b6d3de8f16532510bb725.jpg: 416x416 2 helmets, Done. (0.043s)
    image 210/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005507_jpg.rf.9d7397f134ad91cba1ef0289a53d0501.jpg: 416x416 1 helmet, Done. (0.036s)
    image 211/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005508_jpg.rf.3113ab89595e0e7c9b5ad2d9da82f3ba.jpg: 416x416 1 helmet, Done. (0.036s)
    image 212/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005509_jpg.rf.66f21f3fdd4cf33baf17f891e5aa5ecb.jpg: 416x416 2 helmets, Done. (0.037s)
    image 213/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005510_jpg.rf.1030f9b6272e2527dc2de295de8a56dd.jpg: 416x416 2 helmets, Done. (0.041s)
    image 214/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005511_jpg.rf.795077ac49e2fcf2c388033eef3c1ea7.jpg: 416x416 4 helmets, Done. (0.042s)
    image 215/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005512_jpg.rf.d7a47979c84c84f536791db9c918fa17.jpg: 416x416 4 heads, 5 helmets, Done. (0.038s)
    image 216/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005513_jpg.rf.105587b4771b352e2e53672428700880.jpg: 416x416 3 helmets, Done. (0.037s)
    image 217/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005514_jpg.rf.0b31c75feac173273dd6c93d8cd0bd0b.jpg: 416x416 12 helmets, Done. (0.040s)
    image 218/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005515_jpg.rf.646411017d682535f3c2695e1ccdde6d.jpg: 416x416 3 helmets, Done. (0.038s)
    image 219/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005516_jpg.rf.fe0c6d05bf27ef158e589351401054d4.jpg: 416x416 1 head, Done. (0.038s)
    image 220/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005517_jpg.rf.0f446778fb58b954926ab178002d930c.jpg: 416x416 1 head, 5 helmets, Done. (0.038s)
    image 221/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005518_jpg.rf.34ad0022a9bd9e7caaf855fb74ed9629.jpg: 416x416 8 helmets, Done. (0.038s)
    image 222/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005519_jpg.rf.681b7fd4502eabb7bf11871c274465db.jpg: 416x416 2 helmets, Done. (0.035s)
    image 223/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005520_jpg.rf.b8c9dd1cf92b974a12fa54541573afcb.jpg: 416x416 2 helmets, Done. (0.040s)
    image 224/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005521_jpg.rf.4d65963270acf18825cde12d99b6149a.jpg: 416x416 6 helmets, Done. (0.036s)
    image 225/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005522_jpg.rf.8b1a41393cd10687665aac5c512cea59.jpg: 416x416 1 helmet, Done. (0.037s)
    image 226/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005523_jpg.rf.d50d85fa6cea4aabc1ae94b8a5afd2d7.jpg: 416x416 2 helmets, Done. (0.037s)
    image 227/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005524_jpg.rf.b34f151cc5f84455c19c0e097b6657f7.jpg: 416x416 2 helmets, Done. (0.037s)
    image 228/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005525_jpg.rf.a7a5b49be897eefee1697fd6e107290f.jpg: 416x416 6 helmets, Done. (0.038s)
    image 229/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005526_jpg.rf.6642fbe7c6f60863a3b4acda9996e238.jpg: 416x416 2 helmets, Done. (0.037s)
    image 230/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005527_jpg.rf.2bb690a8fee07a1b6f98d01262b3b46d.jpg: 416x416 1 helmet, Done. (0.036s)
    image 231/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005528_jpg.rf.2e871167eff2ff6dd29add3a1c6e8be8.jpg: 416x416 7 helmets, Done. (0.037s)
    image 232/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005529_jpg.rf.39a93ae1648bcf965ac3dc99ede8fcf9.jpg: 416x416 2 helmets, Done. (0.037s)
    image 233/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005530_jpg.rf.61d5cff9c7c6824dba6ffb2faffe4387.jpg: 416x416 2 helmets, Done. (0.039s)
    image 234/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005531_jpg.rf.02de427a2bfc5e410c9550d2489a7f43.jpg: 416x416 1 helmet, Done. (0.042s)
    image 235/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005532_jpg.rf.c1cec0a2e4ca7ef877132b820c135c4e.jpg: 416x416 1 helmet, Done. (0.040s)
    image 236/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005533_jpg.rf.c29ee969f371cf96ef18e1eacd2c17bc.jpg: 416x416 1 helmet, Done. (0.039s)
    image 237/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005534_jpg.rf.fe2fd2149142985cd9aec2a4c2b1cc53.jpg: 416x416 1 helmet, Done. (0.036s)
    image 238/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005535_jpg.rf.6bb49268f84b49678b9c81af761089f5.jpg: 416x416 4 helmets, Done. (0.036s)
    image 239/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005536_jpg.rf.7e2c0556195559f0c9c46c5de1eb098a.jpg: 416x416 7 heads, 2 helmets, Done. (0.036s)
    image 240/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005537_jpg.rf.8b5596e418f1e75e9428a7b783db8eba.jpg: 416x416 2 helmets, Done. (0.037s)
    image 241/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005538_jpg.rf.a8e1ef9837fb5d3173b0519dddef0e83.jpg: 416x416 1 helmet, Done. (0.037s)
    image 242/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005539_jpg.rf.a49bc12ea77e32d04587b556781ffd9c.jpg: 416x416 2 helmets, Done. (0.038s)
    image 243/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005540_jpg.rf.8bb2d4b79e82eb02513bda8d731d9336.jpg: 416x416 1 helmet, Done. (0.038s)
    image 244/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005541_jpg.rf.636833739fb081e57dea386cc1c12642.jpg: 416x416 6 heads, Done. (0.040s)
    image 245/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005542_jpg.rf.64089d7cfd4d78d2ae2dd0cb0ecb9aab.jpg: 416x416 1 helmet, Done. (0.040s)
    image 246/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005543_jpg.rf.a69261bd8d6152e328bfd2c46f81f9db.jpg: 416x416 2 helmets, Done. (0.036s)
    image 247/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005544_jpg.rf.297285acaad066827adfa7951bdf9b37.jpg: 416x416 10 heads, Done. (0.037s)
    image 248/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005545_jpg.rf.34543927e80378fbff51e7f208e230e4.jpg: 416x416 1 helmet, Done. (0.044s)
    image 249/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005546_jpg.rf.76ba514802a11bc92ac4142c19b82abd.jpg: 416x416 5 helmets, Done. (0.043s)
    image 250/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005547_jpg.rf.507089a61448b69c94b8ca3835070868.jpg: 416x416 5 helmets, Done. (0.039s)
    image 251/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005548_jpg.rf.42b6e31b169ca3e512fe1c53b07fbacb.jpg: 416x416 2 helmets, Done. (0.037s)
    image 252/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005549_jpg.rf.2b5a3f7591c364331880d0cc309366b2.jpg: 416x416 1 helmet, Done. (0.036s)
    image 253/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005550_jpg.rf.ba3648ed3fa0afafe53158b6627d6361.jpg: 416x416 3 helmets, Done. (0.039s)
    image 254/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005551_jpg.rf.4b72e296a5fa37d01ca6848d37104c08.jpg: 416x416 3 helmets, Done. (0.037s)
    image 255/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005552_jpg.rf.ceb7634f81bc7d56e6792939ed566b55.jpg: 416x416 3 helmets, Done. (0.039s)
    image 256/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005553_jpg.rf.7850dc5f93915cbcfd8602c1de7f0742.jpg: 416x416 2 helmets, Done. (0.039s)
    image 257/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005554_jpg.rf.aba6e696852df1d06dbbd19eb0c2a14a.jpg: 416x416 1 helmet, Done. (0.039s)
    image 258/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005555_jpg.rf.ad5a45126912a618bb581022a1e681b0.jpg: 416x416 1 helmet, Done. (0.037s)
    image 259/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005556_jpg.rf.fb47697df25386b37896e6029b83fdbd.jpg: 416x416 1 helmet, Done. (0.039s)
    image 260/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005557_jpg.rf.c18832b276d34124a547ac15252248b1.jpg: 416x416 2 helmets, Done. (0.037s)
    image 261/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005558_jpg.rf.47241f2ca1a5febe55eb00f01a512006.jpg: 416x416 2 helmets, Done. (0.038s)
    image 262/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005559_jpg.rf.8bbef6996455279c792b66f17ef26c49.jpg: 416x416 1 helmet, Done. (0.038s)
    image 263/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005560_jpg.rf.c9f7e3b3e435ef6de42362036b4c23c0.jpg: 416x416 1 helmet, Done. (0.038s)
    image 264/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005561_jpg.rf.45872828b2f9cca87c196f7702ed22ec.jpg: 416x416 3 helmets, Done. (0.039s)
    image 265/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005562_jpg.rf.9ee357cd99c03b120b37c141a2915f36.jpg: 416x416 4 heads, 4 helmets, Done. (0.043s)
    image 266/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005563_jpg.rf.6b8b037dbc8574656163364fdc70572b.jpg: 416x416 6 helmets, Done. (0.042s)
    image 267/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005564_jpg.rf.4b63ebe33332a7d3cc4be7720f8c8a16.jpg: 416x416 2 helmets, Done. (0.040s)
    image 268/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005565_jpg.rf.3ba55af26bf99896bc9d7fe4699e1e1f.jpg: 416x416 Done. (0.040s)
    image 269/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005566_jpg.rf.b2d4f175a66f88dd0661d8d3a945a952.jpg: 416x416 2 helmets, Done. (0.042s)
    image 270/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005567_jpg.rf.7b1423ea14479e2efbfc4d88b691c796.jpg: 416x416 4 heads, 2 helmets, Done. (0.040s)
    image 271/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005568_jpg.rf.950b200e4a0f3a65dd98a21bf3262f7a.jpg: 416x416 2 helmets, Done. (0.040s)
    image 272/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005569_jpg.rf.bb7980cfeec9a5d13ee255426cfbbc87.jpg: 416x416 2 helmets, Done. (0.045s)
    image 273/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005570_jpg.rf.e02e03ee6c129a81a0288aa37564039b.jpg: 416x416 3 heads, 1 helmet, Done. (0.042s)
    image 274/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005571_jpg.rf.6a03d7e91e5c0a8ba57064b442757ef2.jpg: 416x416 4 helmets, Done. (0.041s)
    image 275/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005572_jpg.rf.45bb1518901b8b1003ee99a732845e03.jpg: 416x416 4 helmets, Done. (0.042s)
    image 276/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005573_jpg.rf.1273961a3064bd923afc049e717bf85b.jpg: 416x416 8 heads, Done. (0.042s)
    image 277/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005574_jpg.rf.65c696aee78baf021f8803aa5f5141d8.jpg: 416x416 3 helmets, Done. (0.040s)
    image 278/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005575_jpg.rf.8a410341ed826fd6550e11bb1c39251f.jpg: 416x416 2 helmets, Done. (0.038s)
    image 279/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005576_jpg.rf.2b2efd7252016cf7339ec43d2679ec44.jpg: 416x416 5 helmets, Done. (0.038s)
    image 280/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005577_jpg.rf.b4dd522891b21bb8fb5534fd4db49c65.jpg: 416x416 3 helmets, Done. (0.037s)
    image 281/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005578_jpg.rf.c18f4c655db56371e67f7ecd6d18238a.jpg: 416x416 2 helmets, Done. (0.038s)
    image 282/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005579_jpg.rf.5d1eafde099628c42196efd756e1c41b.jpg: 416x416 5 heads, Done. (0.036s)
    image 283/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005580_jpg.rf.b36bfc536f59bed5094bc1b9699e023e.jpg: 416x416 3 helmets, Done. (0.039s)
    image 284/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005581_jpg.rf.c87558a98f085270ebb641d92eea502e.jpg: 416x416 4 heads, Done. (0.037s)
    image 285/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005582_jpg.rf.4964e79c38bd48b0c0c1e71f63a01088.jpg: 416x416 1 head, 6 helmets, Done. (0.038s)
    image 286/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005583_jpg.rf.e08968cd0d538a083e9f5d16f6b5bc81.jpg: 416x416 1 helmet, Done. (0.037s)
    image 287/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005584_jpg.rf.687209f20109e749c2471f92846fe8c4.jpg: 416x416 1 helmet, Done. (0.041s)
    image 288/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005585_jpg.rf.f982cab1b5e3ed7ca3309d23fbd809f3.jpg: 416x416 9 heads, 1 helmet, Done. (0.041s)
    image 289/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005586_jpg.rf.a7fb635e37b0169cb788f1e43703ba16.jpg: 416x416 3 helmets, Done. (0.039s)
    image 290/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005587_jpg.rf.127159a140013673b46e551174e12d2f.jpg: 416x416 4 helmets, Done. (0.036s)
    image 291/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005588_jpg.rf.ad4d98f076b39ebb1fd87c94e9afca96.jpg: 416x416 2 helmets, Done. (0.036s)
    image 292/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005589_jpg.rf.f3206faa3cd5476d58634016f28c9281.jpg: 416x416 2 helmets, Done. (0.036s)
    image 293/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005590_jpg.rf.c44b660ad390551458ef71064fc39b14.jpg: 416x416 1 helmet, Done. (0.039s)
    image 294/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005591_jpg.rf.089e3605776898de0ccabff339428685.jpg: 416x416 9 heads, Done. (0.040s)
    image 295/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005592_jpg.rf.cc8a2f4dfecf33089230fda203cc07cd.jpg: 416x416 1 helmet, Done. (0.039s)
    image 296/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005593_jpg.rf.5cf5578f484c0d8a266ddf75f0ad1742.jpg: 416x416 7 heads, Done. (0.038s)
    image 297/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005594_jpg.rf.eb1c446cbf48c80c01e94320c89bc4dd.jpg: 416x416 5 heads, Done. (0.038s)
    image 298/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005595_jpg.rf.5bb2b7ecea0b09728ac370c64214a07a.jpg: 416x416 1 helmet, Done. (0.037s)
    image 299/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005596_jpg.rf.8dbe813911161298a0cdcafe0cb28531.jpg: 416x416 1 helmet, Done. (0.035s)
    image 300/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005597_jpg.rf.fd812d145a52466f9736426d50b2a32b.jpg: 416x416 2 helmets, Done. (0.035s)
    image 301/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005598_jpg.rf.836d5bf50030a32df69dbdd38abf2a2d.jpg: 416x416 2 helmets, Done. (0.037s)
    image 302/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005599_jpg.rf.14c0dc3297f45115bf76c48861e2a4ff.jpg: 416x416 9 heads, 3 helmets, Done. (0.037s)
    image 303/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005600_jpg.rf.abf496a95686373373d3c155f18a36d8.jpg: 416x416 7 helmets, Done. (0.046s)
    image 304/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005601_jpg.rf.108f0470a07b7f728ada906bcfba28f6.jpg: 416x416 4 helmets, Done. (0.043s)
    image 305/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005602_jpg.rf.6c5516b2ce0560a416e37e4919aff212.jpg: 416x416 9 helmets, Done. (0.040s)
    image 306/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005603_jpg.rf.4b157b1ff1cd3520cf5282ff312733c3.jpg: 416x416 2 heads, Done. (0.041s)
    image 307/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005604_jpg.rf.22ef8d4a3ea9a82de4495d8024380de6.jpg: 416x416 1 helmet, Done. (0.042s)
    image 308/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005605_jpg.rf.5bcd0883531b4dcca42a42126b8555c3.jpg: 416x416 5 helmets, Done. (0.045s)
    image 309/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005606_jpg.rf.5b9b8c809d2caffcdc8b619a5727d7be.jpg: 416x416 15 heads, Done. (0.047s)
    image 310/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005607_jpg.rf.ae4177f44d44d75f7a41d1d0eefea3a3.jpg: 416x416 3 helmets, Done. (0.039s)
    image 311/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005608_jpg.rf.61cf830405c66beb0d35586167c78fd6.jpg: 416x416 3 helmets, Done. (0.035s)
    image 312/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005609_jpg.rf.02180dc9cb9d924fb41b986f02f5872f.jpg: 416x416 5 helmets, Done. (0.039s)
    image 313/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005610_jpg.rf.98b36989415b3d8f7c2fbc3abaaf3f59.jpg: 416x416 1 helmet, Done. (0.035s)
    image 314/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005611_jpg.rf.692db27e88ca37d794675e465298b63d.jpg: 416x416 5 helmets, Done. (0.038s)
    image 315/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005612_jpg.rf.fa21ad7e1b21bb2d608faf7c1c0a2f0e.jpg: 416x416 3 heads, 9 helmets, Done. (0.038s)
    image 316/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005613_jpg.rf.6b1916742d4403418286adcff04198b8.jpg: 416x416 2 helmets, Done. (0.039s)
    image 317/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005614_jpg.rf.bfdb9f2ff8b65b56527bce5ac8e5d5ca.jpg: 416x416 3 helmets, Done. (0.039s)
    image 318/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005615_jpg.rf.ac40a4e60092eaae756bf48fdfee2084.jpg: 416x416 1 helmet, Done. (0.039s)
    image 319/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005616_jpg.rf.5bc49ff94385e871a7997ada881845c6.jpg: 416x416 2 helmets, Done. (0.036s)
    image 320/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005617_jpg.rf.d86aa9c13443a48e25183b5c0aa9e289.jpg: 416x416 8 helmets, Done. (0.037s)
    image 321/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005618_jpg.rf.eb7d31edde3e51f79dc7972f8dffa4ba.jpg: 416x416 8 helmets, Done. (0.037s)
    image 322/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005619_jpg.rf.c76f457dc7a1195bda4a16baf4d2ef46.jpg: 416x416 1 helmet, Done. (0.038s)
    image 323/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005620_jpg.rf.a8854452afb42cb73d6639e0368897a4.jpg: 416x416 1 helmet, Done. (0.040s)
    image 324/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005621_jpg.rf.9b7282b3cdd483e6c6c328f7a5458c85.jpg: 416x416 11 helmets, Done. (0.041s)
    image 325/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005622_jpg.rf.8816e5269f32875fc3dceb6c19e1adb4.jpg: 416x416 2 helmets, Done. (0.041s)
    image 326/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005623_jpg.rf.2f6e3d8339dc1f8784ac92782da33e24.jpg: 416x416 1 helmet, Done. (0.040s)
    image 327/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005624_jpg.rf.c56ccb97e8446de480ac17e5f74b9ce4.jpg: 416x416 5 heads, 3 helmets, Done. (0.040s)
    image 328/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005625_jpg.rf.884e9ff0f0aa8c503878857b371caadb.jpg: 416x416 1 helmet, Done. (0.043s)
    image 329/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005626_jpg.rf.201b353f2b125a9f3884ca2ffdb74bb6.jpg: 416x416 2 helmets, Done. (0.038s)
    image 330/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005627_jpg.rf.327c37a52ce47f0019b67767d9b0171e.jpg: 416x416 1 helmet, Done. (0.038s)
    image 331/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005628_jpg.rf.b473a3497ef07502e147eb5617318cc4.jpg: 416x416 2 helmets, Done. (0.039s)
    image 332/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005629_jpg.rf.da5398e2decff8ba584ddc746df203c4.jpg: 416x416 3 helmets, Done. (0.037s)
    image 333/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005630_jpg.rf.d79d6c8da00c74ce2b80a0d29288a023.jpg: 416x416 1 helmet, Done. (0.038s)
    image 334/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005631_jpg.rf.1c24657db94db6635820044ee638e4f3.jpg: 416x416 1 helmet, Done. (0.037s)
    image 335/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005632_jpg.rf.d229a6d2c4b0ca4ced208b50f6bf8e5f.jpg: 416x416 1 helmet, Done. (0.041s)
    image 336/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005633_jpg.rf.5a342871e536cca01b53e252a6e51218.jpg: 416x416 13 helmets, Done. (0.039s)
    image 337/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005634_jpg.rf.ff5cc0c8607d540e9db5c0c8b7aecc5a.jpg: 416x416 5 heads, Done. (0.038s)
    image 338/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005635_jpg.rf.3c9664d91bcf448d1c46fbb9954d94d7.jpg: 416x416 7 helmets, Done. (0.037s)
    image 339/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005636_jpg.rf.1b1eab718bfca1df733dc3a0a3ba9ee7.jpg: 416x416 4 helmets, Done. (0.037s)
    image 340/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005637_jpg.rf.74ec7e05f90875601f967ce181c7b6ab.jpg: 416x416 1 helmet, Done. (0.039s)
    image 341/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005638_jpg.rf.813aa6c389c9341876bac0055470655a.jpg: 416x416 1 helmet, Done. (0.036s)
    image 342/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005639_jpg.rf.2539413c46ae2c4babdeefbe0e5a21d7.jpg: 416x416 6 helmets, Done. (0.038s)
    image 343/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005640_jpg.rf.e438f28eb5fa0152865e871f5736c779.jpg: 416x416 3 helmets, Done. (0.037s)
    image 344/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005641_jpg.rf.28f71cbbb60d6c33fcb59e848a3cdff3.jpg: 416x416 3 heads, Done. (0.038s)
    image 345/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005642_jpg.rf.174773f9822c669c298784e57286c2c5.jpg: 416x416 2 helmets, Done. (0.037s)
    image 346/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005643_jpg.rf.f63965956ede6a50838ac69f97f6e341.jpg: 416x416 3 helmets, Done. (0.038s)
    image 347/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005644_jpg.rf.58c8dbd002af74aa2dcbe48497dbb88d.jpg: 416x416 13 helmets, Done. (0.040s)
    image 348/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005645_jpg.rf.1083be2cb091c66816b08843874b1cd5.jpg: 416x416 4 helmets, Done. (0.044s)
    image 349/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005646_jpg.rf.c12e2f7eac53697a0e5e6649f5df5634.jpg: 416x416 1 helmet, Done. (0.043s)
    image 350/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005647_jpg.rf.71dc440e59e536296019a9b69d16f0b5.jpg: 416x416 13 heads, Done. (0.043s)
    image 351/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005648_jpg.rf.ee5ee8abaf8bcbb4beec9b08eb1973c1.jpg: 416x416 2 helmets, Done. (0.043s)
    image 352/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005649_jpg.rf.bc5841acf1b5af8402ac6d240edb3f98.jpg: 416x416 2 helmets, Done. (0.042s)
    image 353/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005650_jpg.rf.bebbc00ea8c9c4c324b73659b4ac8526.jpg: 416x416 2 helmets, Done. (0.038s)
    image 354/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005651_jpg.rf.34b366e98e4f21c28a60f818ff9b178e.jpg: 416x416 5 helmets, Done. (0.038s)
    image 355/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005652_jpg.rf.c80e839bb3060981d254a9200a3f53ce.jpg: 416x416 1 helmet, Done. (0.035s)
    image 356/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005653_jpg.rf.a09aa9bd274c964504584a5dfe6acc64.jpg: 416x416 7 helmets, Done. (0.034s)
    image 357/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005654_jpg.rf.a3b4f3c23e3f5203ebceee0fac22b290.jpg: 416x416 2 heads, 1 helmet, Done. (0.032s)
    image 358/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005655_jpg.rf.096c1db691700109c0096e0dd50cfa24.jpg: 416x416 1 helmet, Done. (0.036s)
    image 359/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005656_jpg.rf.65039468d2fbc4cf9694397f2473a505.jpg: 416x416 4 helmets, Done. (0.034s)
    image 360/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005657_jpg.rf.5a6989694c2a65baa79c8b34e14d7c86.jpg: 416x416 9 heads, Done. (0.035s)
    image 361/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005658_jpg.rf.bc1902ceacef2c0140641a5ddffd1f4f.jpg: 416x416 3 helmets, Done. (0.032s)
    image 362/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005659_jpg.rf.454b95418151152592c7309d0e9ce920.jpg: 416x416 2 helmets, Done. (0.041s)
    image 363/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005660_jpg.rf.7578736cbbf455efbd7f3362be4c93cc.jpg: 416x416 2 helmets, Done. (0.038s)
    image 364/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005661_jpg.rf.d182b2c1da495fec058138282cc4ea40.jpg: 416x416 1 helmet, Done. (0.035s)
    image 365/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005662_jpg.rf.ffece27cbd6d99b9564b2474e2d312c9.jpg: 416x416 2 helmets, Done. (0.032s)
    image 366/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005663_jpg.rf.647d834045bf9e1bd516f52ea654d8ee.jpg: 416x416 2 helmets, Done. (0.034s)
    image 367/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005664_jpg.rf.62e558c9ad3edef74aebaec48ec9db66.jpg: 416x416 2 helmets, Done. (0.040s)
    image 368/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005665_jpg.rf.e89587c4ac78e64663cf2747ddc80f08.jpg: 416x416 2 helmets, Done. (0.041s)
    image 369/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005666_jpg.rf.d2addd076765bbd5b9b64da053ac7453.jpg: 416x416 6 helmets, Done. (0.037s)
    image 370/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005667_jpg.rf.c4b46fa7990229c6387a28fd9875e4c3.jpg: 416x416 2 helmets, Done. (0.035s)
    image 371/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005668_jpg.rf.876af844b74257837bae27278c66eeee.jpg: 416x416 8 helmets, Done. (0.032s)
    image 372/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005669_jpg.rf.f6dad7601fe65fd0a7be2a9e86190917.jpg: 416x416 1 helmet, Done. (0.034s)
    image 373/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005670_jpg.rf.fef21ae4b97610e91f51abe1b956532f.jpg: 416x416 3 helmets, Done. (0.033s)
    image 374/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005671_jpg.rf.d74f524c83069136f1312569924845ce.jpg: 416x416 1 helmet, Done. (0.032s)
    image 375/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005672_jpg.rf.a5e3e09d34d3251a7bd0ce4183179299.jpg: 416x416 12 helmets, Done. (0.032s)
    image 376/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005673_jpg.rf.8220890429e09f5dbd8a97ff6cafefbb.jpg: 416x416 2 heads, 9 helmets, Done. (0.034s)
    image 377/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005674_jpg.rf.8520855af3cb374fcfefa4014eb703ae.jpg: 416x416 3 helmets, Done. (0.032s)
    image 378/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005675_jpg.rf.b0049b1b858bb55f463f9116f93fb650.jpg: 416x416 2 heads, 3 helmets, Done. (0.032s)
    image 379/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005676_jpg.rf.778ee9bd76da671a5e5e3f1dbda66ede.jpg: 416x416 2 helmets, Done. (0.034s)
    image 380/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005677_jpg.rf.8ec6135b40dc657cbb7ad6f241861ea7.jpg: 416x416 8 helmets, Done. (0.035s)
    image 381/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005678_jpg.rf.52b5bb5b2b01e762e2f3438aa7f39517.jpg: 416x416 1 helmet, Done. (0.034s)
    image 382/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005679_jpg.rf.0311bf8789f5ec2889b202b37bdcbe41.jpg: 416x416 1 helmet, Done. (0.033s)
    image 383/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005680_jpg.rf.d9a7470b08b3a7d6f197d845097a805b.jpg: 416x416 1 head, Done. (0.036s)
    image 384/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005681_jpg.rf.71bb4a9ed6edcf4a106c26a3325a748e.jpg: 416x416 6 helmets, Done. (0.039s)
    image 385/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005682_jpg.rf.fe81787cec27b7961dce5be580c6f1f7.jpg: 416x416 5 heads, Done. (0.041s)
    image 386/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005683_jpg.rf.4d4a772d8ae8a7802c2f1fd1a80299eb.jpg: 416x416 2 helmets, Done. (0.042s)
    image 387/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005684_jpg.rf.d34cb8761040e262c8335fce7fcd6c67.jpg: 416x416 4 helmets, Done. (0.040s)
    image 388/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005685_jpg.rf.927a23addaafba6a2371218982f1de8b.jpg: 416x416 6 heads, Done. (0.037s)
    image 389/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005686_jpg.rf.d5199e940a5ba1958de4f7420aa8e303.jpg: 416x416 3 helmets, Done. (0.039s)
    image 390/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005687_jpg.rf.15bb7b3e3c6104b3033379884b7eb58c.jpg: 416x416 2 helmets, Done. (0.035s)
    image 391/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005688_jpg.rf.c2aac2c2be331f2aadd604676c8e7da0.jpg: 416x416 2 helmets, Done. (0.032s)
    image 392/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005689_jpg.rf.00046e7ed486ae8557fe42403b9dc95d.jpg: 416x416 6 helmets, Done. (0.033s)
    image 393/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005690_jpg.rf.5c402da3c3b392f72309674baeaf4490.jpg: 416x416 1 helmet, Done. (0.035s)
    image 394/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005691_jpg.rf.577424196c76ae045bab7e76cad5ace4.jpg: 416x416 5 helmets, Done. (0.035s)
    image 395/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005692_jpg.rf.7b0fa9d2e5ab83a20257155d0e6ffc1b.jpg: 416x416 3 helmets, Done. (0.033s)
    image 396/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005693_jpg.rf.979149ff5bb4ff16cfdb1626684b406b.jpg: 416x416 1 head, 1 helmet, Done. (0.033s)
    image 397/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005694_jpg.rf.7bf2e7e27140c8badb14a9315dfb574b.jpg: 416x416 3 helmets, Done. (0.036s)
    image 398/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005695_jpg.rf.5f0e40e73d6480eadfa6e119af2b720a.jpg: 416x416 2 helmets, Done. (0.036s)
    image 399/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005696_jpg.rf.300eccb2a8e8d1a347481a6fe5d29aeb.jpg: 416x416 2 helmets, Done. (0.033s)
    image 400/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005697_jpg.rf.e5853507b89138600e5a06a099d31a57.jpg: 416x416 5 heads, Done. (0.034s)
    image 401/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005698_jpg.rf.c7a608d48efcd27cb9195d9c306ea222.jpg: 416x416 1 helmet, Done. (0.035s)
    image 402/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005699_jpg.rf.12fd8e7c221d5516a7fb775ffbd031c8.jpg: 416x416 4 helmets, Done. (0.035s)
    image 403/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005700_jpg.rf.8903feeb89fc4c0a93e49d0e8f5bd28a.jpg: 416x416 3 helmets, Done. (0.036s)
    image 404/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005701_jpg.rf.dda262c20dd1cf7fd1e2da7c97288dd8.jpg: 416x416 1 helmet, Done. (0.035s)
    image 405/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005702_jpg.rf.dc254eefd0c300c62e80214124b03a42.jpg: 416x416 2 helmets, Done. (0.037s)
    image 406/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005703_jpg.rf.1a3586cd1c9d11b789b16c0b33a32e42.jpg: 416x416 4 helmets, Done. (0.034s)
    image 407/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005704_jpg.rf.0e1afa9ccc3adde3dfc3e8e2208b065c.jpg: 416x416 2 helmets, Done. (0.034s)
    image 408/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005705_jpg.rf.e27ba3295a2229983b06152288225c3d.jpg: 416x416 16 heads, Done. (0.037s)
    image 409/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005706_jpg.rf.1fa5c01e174778220044e1c7a53b88c4.jpg: 416x416 2 helmets, Done. (0.037s)
    image 410/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005707_jpg.rf.fe90981b4e2874ac73ef70180b54582e.jpg: 416x416 4 helmets, Done. (0.038s)
    image 411/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005708_jpg.rf.5c0c8c8224f718a2db0d3767656f0f82.jpg: 416x416 1 helmet, Done. (0.039s)
    image 412/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005709_jpg.rf.0b5787ea4e4689760f80a5acac98d124.jpg: 416x416 2 helmets, Done. (0.035s)
    image 413/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005710_jpg.rf.a87c710178d6178ecbab5ecea04648aa.jpg: 416x416 3 heads, Done. (0.032s)
    image 414/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005711_jpg.rf.2dd02e79a2df8587bdcd8aaac874f37e.jpg: 416x416 11 helmets, Done. (0.033s)
    image 415/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005712_jpg.rf.b9593dada8658f5d2066bcd09a34b3a9.jpg: 416x416 3 helmets, Done. (0.035s)
    image 416/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005713_jpg.rf.c7aa01e2a7d0e01c99757fc401c5cc8e.jpg: 416x416 4 helmets, Done. (0.038s)
    image 417/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005714_jpg.rf.e89e1d33a4c52221ea8fb9c3b7c60819.jpg: 416x416 Done. (0.034s)
    image 418/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005715_jpg.rf.a06cd6d4f079514f0c67e72cab0949d3.jpg: 416x416 6 helmets, Done. (0.033s)
    image 419/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005716_jpg.rf.33e1efc73f755a8eeab0546d4cc73fd3.jpg: 416x416 3 helmets, Done. (0.035s)
    image 420/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005717_jpg.rf.522ddc6e3536ca076d3525c6809f090c.jpg: 416x416 10 helmets, Done. (0.032s)
    image 421/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005718_jpg.rf.2247381f84c9764c99e5a9de09a699e2.jpg: 416x416 1 helmet, Done. (0.033s)
    image 422/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005719_jpg.rf.98c7e8494db9c730346153c8fe17e158.jpg: 416x416 1 helmet, Done. (0.035s)
    image 423/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005720_jpg.rf.4330a6771fc2d4be192a7c66099c65e2.jpg: 416x416 Done. (0.034s)
    image 424/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005721_jpg.rf.ca797f9764c0c2316167465a48a6ca0a.jpg: 416x416 2 helmets, Done. (0.034s)
    image 425/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005722_jpg.rf.7017b164bd44ff399244a0f052c71676.jpg: 416x416 2 helmets, Done. (0.037s)
    image 426/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005723_jpg.rf.dad32ff3138a968224e7e6949ecb6da6.jpg: 416x416 7 helmets, Done. (0.040s)
    image 427/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005724_jpg.rf.0810c555221d9e936d85fdf3eb7150c5.jpg: 416x416 1 helmet, Done. (0.040s)
    image 428/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005725_jpg.rf.22c1cedfaa215adaccfe41506eaa893b.jpg: 416x416 2 helmets, Done. (0.037s)
    image 429/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005726_jpg.rf.6496a3d7fe3e8859cfd57f721917fe46.jpg: 416x416 2 helmets, Done. (0.036s)
    image 430/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005727_jpg.rf.562ef237520922ab266e77266cfcc0f3.jpg: 416x416 6 heads, 1 helmet, Done. (0.034s)
    image 431/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005728_jpg.rf.2615f228984b61efbac229bc1fdb53af.jpg: 416x416 1 helmet, Done. (0.041s)
    image 432/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005729_jpg.rf.0e2a9057d0234b070e381a3408181b4d.jpg: 416x416 3 helmets, Done. (0.037s)
    image 433/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005730_jpg.rf.b35b91f01a79734b15b95f7bee73c4f6.jpg: 416x416 4 helmets, Done. (0.040s)
    image 434/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005731_jpg.rf.64b7ce51d7f14ae518cdd2dfb75a0fab.jpg: 416x416 2 helmets, Done. (0.050s)
    image 435/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005732_jpg.rf.ac81cd8151a81d79951728c9e3edb2f7.jpg: 416x416 2 helmets, Done. (0.053s)
    image 436/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005733_jpg.rf.f5457ba28bfcbfa99dcb006b7edd4e19.jpg: 416x416 3 helmets, Done. (0.051s)
    image 437/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005734_jpg.rf.fb24a57c3a8ccb51a2930dd2b49b25a3.jpg: 416x416 6 helmets, Done. (0.050s)
    image 438/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005735_jpg.rf.03bb805ce7f52e66a947a1671a62663a.jpg: 416x416 2 helmets, Done. (0.046s)
    image 439/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005736_jpg.rf.713290046454f80f587518433f04d7f0.jpg: 416x416 1 helmet, Done. (0.034s)
    image 440/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005737_jpg.rf.379602d767bc684b8a8b8d1b214601b0.jpg: 416x416 1 helmet, Done. (0.031s)
    image 441/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005738_jpg.rf.23616de9b057158e2672aeddb9f9d509.jpg: 416x416 3 helmets, Done. (0.033s)
    image 442/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005739_jpg.rf.26b55b661cce3dce821df218cd1dd194.jpg: 416x416 2 heads, 7 helmets, Done. (0.033s)
    image 443/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005740_jpg.rf.b65c35c0913caba057bc38230535b743.jpg: 416x416 4 helmets, Done. (0.038s)
    image 444/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005741_jpg.rf.36440144282f03e6df279144c2a76124.jpg: 416x416 1 helmet, Done. (0.035s)
    image 445/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005742_jpg.rf.0bb8ae0ae9b8f0b7e863498ba3bf51b5.jpg: 416x416 2 helmets, Done. (0.036s)
    image 446/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005743_jpg.rf.5e61c9c121aea043f7f0e3d277cb3819.jpg: 416x416 4 helmets, Done. (0.039s)
    image 447/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005744_jpg.rf.5a40ea87cd1605c9115a1a0e2ddd5ed0.jpg: 416x416 10 heads, Done. (0.039s)
    image 448/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005745_jpg.rf.e5e1ad75d8a860cf49bc1db944aa8328.jpg: 416x416 1 helmet, Done. (0.038s)
    image 449/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005746_jpg.rf.54927fca1526177951c7ae5fdb4d4444.jpg: 416x416 2 helmets, Done. (0.037s)
    image 450/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005747_jpg.rf.297d181143743f26d3a748f2b55bbc41.jpg: 416x416 1 helmet, Done. (0.038s)
    image 451/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005748_jpg.rf.7145880e1f4da208e450950bf3052e25.jpg: 416x416 2 helmets, Done. (0.032s)
    image 452/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005749_jpg.rf.f44672ab39779b836bdbf5ee9302e33e.jpg: 416x416 1 helmet, Done. (0.036s)
    image 453/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005750_jpg.rf.13f1c6f04460c5259bbd4a505fdcbe4a.jpg: 416x416 1 helmet, Done. (0.033s)
    image 454/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005751_jpg.rf.236a13a5594cb7eaff46847f998a5cfd.jpg: 416x416 1 helmet, Done. (0.033s)
    image 455/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005752_jpg.rf.d43675fb35e6b0df2d790ebad7fe0f05.jpg: 416x416 2 helmets, Done. (0.033s)
    image 456/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005753_jpg.rf.ad3f2f6646c805a36542c1658c30efe2.jpg: 416x416 1 helmet, Done. (0.034s)
    image 457/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005754_jpg.rf.0faa8d8d5b162f444439be4dae56fa5f.jpg: 416x416 1 head, 3 helmets, Done. (0.033s)
    image 458/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005755_jpg.rf.a3a5bca1285f9b0a3c86a1860ce7d092.jpg: 416x416 2 helmets, Done. (0.033s)
    image 459/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005756_jpg.rf.5844aaad6a6d6a68a0b96c0bd8970b6d.jpg: 416x416 2 helmets, Done. (0.033s)
    image 460/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005757_jpg.rf.9931a34a930702007db8b38c2c994845.jpg: 416x416 3 helmets, Done. (0.037s)
    image 461/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005758_jpg.rf.a08f91b2ade97e2bb4880800c87a0948.jpg: 416x416 3 helmets, Done. (0.035s)
    image 462/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005759_jpg.rf.fc13732ee787ad2e36759385f6fc3651.jpg: 416x416 2 helmets, Done. (0.034s)
    image 463/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005760_jpg.rf.df5dd2b4cd96487b552bb46dd939f2b2.jpg: 416x416 1 helmet, Done. (0.034s)
    image 464/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005761_jpg.rf.a14d926d627bea6e30a4fb2eb403dae4.jpg: 416x416 1 helmet, Done. (0.038s)
    image 465/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005762_jpg.rf.3a7bff6b4e645a4eabc8bc6440dad78f.jpg: 416x416 3 helmets, Done. (0.037s)
    image 466/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005763_jpg.rf.d7f392c4f3ca9da7fccf2c976de08976.jpg: 416x416 9 heads, Done. (0.036s)
    image 467/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005764_jpg.rf.49aa8ec0c6edf766de6f6ae857cd88dd.jpg: 416x416 1 helmet, Done. (0.038s)
    image 468/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005765_jpg.rf.0dd270d8e75e6e7c864b2299ac91093e.jpg: 416x416 2 helmets, Done. (0.035s)
    image 469/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005766_jpg.rf.d568e1747edbc4d79c9c8572b25abf35.jpg: 416x416 2 helmets, Done. (0.034s)
    image 470/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005767_jpg.rf.16c01fea482a25175bd2bed14b93819b.jpg: 416x416 3 helmets, Done. (0.032s)
    image 471/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005768_jpg.rf.29dfbb8391300c2e082ed71761b6ef7d.jpg: 416x416 3 helmets, Done. (0.032s)
    image 472/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005769_jpg.rf.7d8e1786fa056a032b0a053e57d93205.jpg: 416x416 2 helmets, Done. (0.037s)
    image 473/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005770_jpg.rf.f9286a0481e353b43baada680281e046.jpg: 416x416 1 helmet, Done. (0.037s)
    image 474/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005771_jpg.rf.7b58e4771ee4de3a98f7af6a6f0d58de.jpg: 416x416 2 heads, 3 helmets, Done. (0.038s)
    image 475/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005772_jpg.rf.6a7ce76981ce950973c4c2a457c5e999.jpg: 416x416 4 helmets, Done. (0.036s)
    image 476/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005773_jpg.rf.089a994c1dbb8a223d255f65c7f6bb60.jpg: 416x416 1 helmet, Done. (0.034s)
    image 477/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005774_jpg.rf.54dce38a49fd4ea18faf246c31b87f3e.jpg: 416x416 1 helmet, Done. (0.034s)
    image 478/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005775_jpg.rf.44bcac4094e0d7986669d179ecbf7da3.jpg: 416x416 2 helmets, Done. (0.038s)
    image 479/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005776_jpg.rf.eb018ccd57b1ffd7c9a843994e17ffe1.jpg: 416x416 8 helmets, Done. (0.035s)
    image 480/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005777_jpg.rf.a6f3b3cfa8af81c42ab43175eb68213b.jpg: 416x416 1 helmet, Done. (0.032s)
    image 481/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005778_jpg.rf.32799b7537ac74b3219fbbdf53e79f22.jpg: 416x416 17 helmets, Done. (0.034s)
    image 482/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005779_jpg.rf.4ca024e1f5a7123e0705bc74665b0427.jpg: 416x416 9 heads, Done. (0.036s)
    image 483/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005780_jpg.rf.263d75cb0c0cf0d5d3e790f35414865b.jpg: 416x416 4 helmets, Done. (0.036s)
    image 484/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005781_jpg.rf.1d50eb86ce1020601923472e68db7f4b.jpg: 416x416 2 helmets, Done. (0.034s)
    image 485/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005782_jpg.rf.9e11dedb4390f1adfbbb3346c5f561a4.jpg: 416x416 2 helmets, Done. (0.033s)
    image 486/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005783_jpg.rf.35f211226ca1adf09f70943711f9411e.jpg: 416x416 1 helmet, Done. (0.038s)
    image 487/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005784_jpg.rf.db9f4e2ea9ee96d406fb28fec72d6a10.jpg: 416x416 1 head, 2 helmets, Done. (0.033s)
    image 488/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005785_jpg.rf.74643f75557e53fcace1e362e12d5094.jpg: 416x416 14 heads, Done. (0.040s)
    image 489/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005786_jpg.rf.802d75a67ca77a08b041182ef4631272.jpg: 416x416 8 heads, Done. (0.043s)
    image 490/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005787_jpg.rf.1c472d358c7cdbb8126ffae5dba990f8.jpg: 416x416 2 helmets, Done. (0.035s)
    image 491/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005788_jpg.rf.fe6c8d7622a3d2bfdbc7d1b683ae56f1.jpg: 416x416 2 helmets, Done. (0.032s)
    image 492/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005789_jpg.rf.00c8c06e4587a278db845c4098898c24.jpg: 416x416 5 heads, 1 helmet, Done. (0.036s)
    image 493/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005790_jpg.rf.db31f1854e2fb9e70bc661147974ec75.jpg: 416x416 3 heads, 3 helmets, Done. (0.044s)
    image 494/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005791_jpg.rf.d9c6c98fedd83668ab4dc6d96b50e47b.jpg: 416x416 3 helmets, Done. (0.043s)
    image 495/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005792_jpg.rf.c7013517e00477c8a5669c90474a25d6.jpg: 416x416 4 helmets, Done. (0.035s)
    image 496/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005793_jpg.rf.830deb16d5a69535c793025ae8259e7c.jpg: 416x416 3 helmets, Done. (0.035s)
    image 497/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005794_jpg.rf.c31fbd4ed5aea1bb0c71ffe5c9ffcad0.jpg: 416x416 10 heads, Done. (0.034s)
    image 498/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005795_jpg.rf.3174da9c4205dac441a0088097ab93a3.jpg: 416x416 4 heads, 1 helmet, Done. (0.034s)
    image 499/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005796_jpg.rf.2d2bfd0c72e86423ff795f28c95a4131.jpg: 416x416 3 helmets, Done. (0.033s)
    image 500/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005797_jpg.rf.cefae3ac7cf8a7c6ab8b3b69fe7c725b.jpg: 416x416 2 helmets, Done. (0.033s)
    image 501/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005798_jpg.rf.24f2fd5a122c7933b245b1a336c26140.jpg: 416x416 1 helmet, Done. (0.032s)
    image 502/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005799_jpg.rf.0c705e438f6e9760be55804763ed796a.jpg: 416x416 7 heads, Done. (0.033s)
    image 503/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005800_jpg.rf.a993d804cec1d03299362f50389e2c7d.jpg: 416x416 2 helmets, Done. (0.033s)
    image 504/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005801_jpg.rf.ba07c2b0e9ba017c617be0259d60c41e.jpg: 416x416 2 helmets, Done. (0.033s)
    image 505/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005802_jpg.rf.55fc20f0e534a993d7712b5147b16f8a.jpg: 416x416 2 helmets, Done. (0.032s)
    image 506/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005803_jpg.rf.8c9d99f0158f2863f3a229fb80724f24.jpg: 416x416 2 helmets, Done. (0.031s)
    image 507/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005804_jpg.rf.821ebb199f4d0c94fed0d5a5adff06cb.jpg: 416x416 3 helmets, Done. (0.033s)
    image 508/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005805_jpg.rf.25cb5b8877ddf16f4da76f105941c27a.jpg: 416x416 3 helmets, Done. (0.032s)
    image 509/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005806_jpg.rf.4b148e987d1fe636c00b893b3a7b011a.jpg: 416x416 3 helmets, Done. (0.035s)
    image 510/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005807_jpg.rf.735e8ad0013fedca91a0e690765a46c1.jpg: 416x416 1 helmet, Done. (0.036s)
    image 511/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005808_jpg.rf.44ffcc6cfbd7ca0daa0d440393e7b040.jpg: 416x416 2 helmets, Done. (0.042s)
    image 512/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005809_jpg.rf.ceee6bcb2f6a35279f8049f6a0d0c340.jpg: 416x416 3 helmets, Done. (0.043s)
    image 513/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005810_jpg.rf.13520bf4afcfde20d6f2bc5a5796c456.jpg: 416x416 7 helmets, Done. (0.039s)
    image 514/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005811_jpg.rf.b2e7bc6c0083b162ab7e6fe7cd83d348.jpg: 416x416 8 heads, Done. (0.038s)
    image 515/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005812_jpg.rf.433a168e304c66797bd22333384db4a2.jpg: 416x416 1 head, 12 helmets, Done. (0.040s)
    image 516/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005813_jpg.rf.49c43d4d4315049d8f0bd42b890dea50.jpg: 416x416 2 helmets, Done. (0.037s)
    image 517/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005814_jpg.rf.1bf8ab4758fe19362a0432ffe3e5b697.jpg: 416x416 2 helmets, Done. (0.040s)
    image 518/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005815_jpg.rf.f2973f528f13e345a8e30373bc385101.jpg: 416x416 7 heads, Done. (0.037s)
    image 519/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005816_jpg.rf.a830fd3a7a9e9ffaa1d6a425e2f2f077.jpg: 416x416 1 helmet, Done. (0.033s)
    image 520/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005817_jpg.rf.ac10da8a2be0ce94e7c9267694313715.jpg: 416x416 1 helmet, Done. (0.034s)
    image 521/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005818_jpg.rf.6f9a2f6c9640f00818c3cf675a8ad6d6.jpg: 416x416 9 heads, Done. (0.034s)
    image 522/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005819_jpg.rf.39ac27e1c854eab85e01438952e0a77f.jpg: 416x416 5 helmets, Done. (0.033s)
    image 523/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005820_jpg.rf.ecbb2a1ad286d329c494895f9fb35c91.jpg: 416x416 2 helmets, Done. (0.038s)
    image 524/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005821_jpg.rf.766a4544a1483d21df641950d38c3ae1.jpg: 416x416 1 helmet, Done. (0.034s)
    image 525/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005822_jpg.rf.659a76e0591e2cef4f22ec0d51256e61.jpg: 416x416 2 helmets, Done. (0.037s)
    image 526/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005823_jpg.rf.251056e416c8e856c1e49420a8d42b11.jpg: 416x416 1 helmet, Done. (0.032s)
    image 527/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005824_jpg.rf.a3efd13759638c102ae99d932d6fe582.jpg: 416x416 2 helmets, Done. (0.033s)
    image 528/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005825_jpg.rf.0bbb8500e8cf2ae09d0222281f8cd5f7.jpg: 416x416 7 heads, 1 helmet, Done. (0.033s)
    image 529/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005826_jpg.rf.5bc1b1fef81f1f668453e0cd8f33e351.jpg: 416x416 4 helmets, Done. (0.035s)
    image 530/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005827_jpg.rf.138439ccf9ea892b14130e8d85b6f295.jpg: 416x416 5 heads, Done. (0.040s)
    image 531/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005828_jpg.rf.9bda7482ed1da80787c489129b169af5.jpg: 416x416 2 helmets, Done. (0.038s)
    image 532/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005829_jpg.rf.c436e9e831b2c9785febe11adefca1ce.jpg: 416x416 4 helmets, Done. (0.034s)
    image 533/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005830_jpg.rf.0f309e7eff08aad4384bd06ff442ae9e.jpg: 416x416 3 heads, 5 helmets, Done. (0.040s)
    image 534/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005831_jpg.rf.d457b34919d76354344bc2aa419ade53.jpg: 416x416 7 helmets, Done. (0.036s)
    image 535/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005832_jpg.rf.4d8132e239ecaa44ad30fd2d3feb6140.jpg: 416x416 9 heads, Done. (0.040s)
    image 536/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005833_jpg.rf.cf46826511d5015e74d08ddaf3eb0c7e.jpg: 416x416 4 helmets, Done. (0.035s)
    image 537/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005834_jpg.rf.f6e37d314c09ab958cd7d43a2cd9b782.jpg: 416x416 6 heads, Done. (0.034s)
    image 538/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005835_jpg.rf.754601aea95419e717b6be4086879cb1.jpg: 416x416 6 helmets, Done. (0.033s)
    image 539/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005836_jpg.rf.bd42d3c43bfba00addecde6f0c99befa.jpg: 416x416 3 helmets, Done. (0.032s)
    image 540/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005837_jpg.rf.471c63b43b27676a7f487499c26fe43d.jpg: 416x416 Done. (0.034s)
    image 541/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005838_jpg.rf.a1f71fdfaddbff67c2c79e8a1c3a6ab4.jpg: 416x416 3 helmets, Done. (0.035s)
    image 542/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005839_jpg.rf.852b4174c1d14a73fc9739331b10e80e.jpg: 416x416 2 helmets, Done. (0.034s)
    image 543/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005840_jpg.rf.f29f3a8d36c9e7e3ba8863a876171686.jpg: 416x416 2 helmets, Done. (0.033s)
    image 544/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005841_jpg.rf.8468c6bf5090a7f1f8058b117b7896e4.jpg: 416x416 9 helmets, Done. (0.033s)
    image 545/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005842_jpg.rf.e7b9c6a59450ce060f25d77e780fb1a3.jpg: 416x416 1 helmet, Done. (0.034s)
    image 546/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005843_jpg.rf.b3358bd626ebaad9ed40dd3cc07be3a8.jpg: 416x416 2 helmets, Done. (0.035s)
    image 547/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005844_jpg.rf.b9315eadbd974552a824270166fbd48f.jpg: 416x416 9 heads, Done. (0.035s)
    image 548/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005845_jpg.rf.a3c1aee89c4bf2e3f80e80eec8329d29.jpg: 416x416 3 helmets, Done. (0.033s)
    image 549/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005846_jpg.rf.4e4134c28198cf32e7c9b02a7b5c91f9.jpg: 416x416 2 helmets, Done. (0.033s)
    image 550/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005847_jpg.rf.f9d214afc2da2acaedd9e67ac06966af.jpg: 416x416 6 heads, Done. (0.033s)
    image 551/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005848_jpg.rf.2f69b5d873bef4553b729986a7dd1452.jpg: 416x416 4 helmets, Done. (0.040s)
    image 552/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005849_jpg.rf.571a474ad00b5a4bdd26cef21248ab96.jpg: 416x416 3 helmets, Done. (0.038s)
    image 553/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005850_jpg.rf.90b7d1935903130a8d6d4bf9a7634e27.jpg: 416x416 4 helmets, Done. (0.043s)
    image 554/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005851_jpg.rf.0e2c07cad78a8a5bf5ef2dfc7ae2ca2d.jpg: 416x416 1 helmet, Done. (0.032s)
    image 555/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005852_jpg.rf.e0af3211e99cf876dc87949a0a3da49d.jpg: 416x416 3 helmets, Done. (0.037s)
    image 556/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005853_jpg.rf.60af6c7a36a9e0b2cc80178fa84cf9f5.jpg: 416x416 5 helmets, Done. (0.040s)
    image 557/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005854_jpg.rf.15b5c87d1d887b707336b3e75e94fec2.jpg: 416x416 1 helmet, Done. (0.040s)
    image 558/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005855_jpg.rf.09c6648c2383c6772e2c05d71e32e4aa.jpg: 416x416 3 helmets, Done. (0.039s)
    image 559/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005856_jpg.rf.6e355078efd7b2d7fe39e1813db2b21e.jpg: 416x416 2 helmets, Done. (0.033s)
    image 560/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005857_jpg.rf.16c856e30334c7c6a46f9e8892e52921.jpg: 416x416 14 heads, Done. (0.034s)
    image 561/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005858_jpg.rf.da955557683e278239cd58dd98a237a9.jpg: 416x416 2 helmets, Done. (0.034s)
    image 562/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005859_jpg.rf.188110096ab59026aa3f3a1683eb4d77.jpg: 416x416 1 helmet, Done. (0.032s)
    image 563/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005860_jpg.rf.fe995b02173423e3d265265408e734a1.jpg: 416x416 1 head, 7 helmets, Done. (0.033s)
    image 564/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005861_jpg.rf.e1a83ff604ef4982a874481fa364e204.jpg: 416x416 1 helmet, Done. (0.032s)
    image 565/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005862_jpg.rf.99e60279f75293a5d676bea8fb38173b.jpg: 416x416 3 helmets, Done. (0.035s)
    image 566/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005863_jpg.rf.10b0100b8b8d8c166202033a021e0f0d.jpg: 416x416 2 helmets, Done. (0.037s)
    image 567/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005864_jpg.rf.61887ccb83009b40cce9758adba12fdc.jpg: 416x416 3 helmets, Done. (0.039s)
    image 568/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005865_jpg.rf.cc350e3f07cc7ab9e3770a4b86a622c7.jpg: 416x416 7 heads, Done. (0.039s)
    image 569/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005866_jpg.rf.fa0297ba39a027b7f5be1b8b03cd5dab.jpg: 416x416 4 helmets, Done. (0.036s)
    image 570/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005867_jpg.rf.bcd18e931e9969d8be182c51d949f702.jpg: 416x416 3 helmets, Done. (0.038s)
    image 571/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005868_jpg.rf.61d1e0f9f0324e6f2ab6440d0db421a3.jpg: 416x416 2 helmets, Done. (0.036s)
    image 572/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005869_jpg.rf.9fe1a6e6602e9b36fe7859005576e99d.jpg: 416x416 3 helmets, Done. (0.035s)
    image 573/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005870_jpg.rf.f9ec9176a618ebee8e2374bb99cafe7e.jpg: 416x416 4 helmets, Done. (0.041s)
    image 574/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005871_jpg.rf.9f158261158f3acee5f2fd29b56a7e0c.jpg: 416x416 1 helmet, Done. (0.041s)
    image 575/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005872_jpg.rf.345f568c3ce9ed7fba875aac60b1cb9e.jpg: 416x416 2 helmets, Done. (0.038s)
    image 576/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005873_jpg.rf.a5d029b21cc6331737db188b7cbeadeb.jpg: 416x416 3 helmets, Done. (0.036s)
    image 577/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005874_jpg.rf.31ae08cda03e178b88a33befa59630c6.jpg: 416x416 1 helmet, Done. (0.044s)
    image 578/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005875_jpg.rf.644a99b5ed410dfcb73dfa0c4ffee027.jpg: 416x416 1 helmet, Done. (0.037s)
    image 579/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005876_jpg.rf.85896082dc6b0a80d5a89fd19cd1ba19.jpg: 416x416 3 helmets, Done. (0.037s)
    image 580/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005877_jpg.rf.c99e275dc1b6a4f4f5f168bfc12ccb7e.jpg: 416x416 4 heads, 4 helmets, Done. (0.037s)
    image 581/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005878_jpg.rf.e3323afd2446ba526a0dd520f9ee5279.jpg: 416x416 3 helmets, Done. (0.037s)
    image 582/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005879_jpg.rf.b1441a9e28813eaaff7a66f2af8e67c3.jpg: 416x416 1 helmet, Done. (0.036s)
    image 583/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005880_jpg.rf.727a6b3b1494174d0b4763e43512e9c2.jpg: 416x416 13 helmets, Done. (0.036s)
    image 584/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005881_jpg.rf.bd3ffc7f1560e2449be7f1e1c9a7692f.jpg: 416x416 1 helmet, Done. (0.037s)
    image 585/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005882_jpg.rf.a34415a58a2be81b052323d307ecfa42.jpg: 416x416 2 helmets, Done. (0.039s)
    image 586/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005883_jpg.rf.e350e7b1253bde53ae4f4779a7ae9294.jpg: 416x416 5 helmets, Done. (0.041s)
    image 587/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005884_jpg.rf.0e7bf3d51349913031b2e63ea9e1ecc4.jpg: 416x416 3 helmets, Done. (0.042s)
    image 588/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005885_jpg.rf.184a0d4e94a308a9896fdd8063efb9a6.jpg: 416x416 Done. (0.041s)
    image 589/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005886_jpg.rf.83b8a331d11b995853b82265307f7f24.jpg: 416x416 4 heads, 4 helmets, Done. (0.042s)
    image 590/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005887_jpg.rf.233abad035f9c6736ace01d944285384.jpg: 416x416 10 helmets, Done. (0.040s)
    image 591/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005888_jpg.rf.a91a851afbdc8aa94d790ea4ee1f12e2.jpg: 416x416 1 helmet, Done. (0.041s)
    image 592/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005889_jpg.rf.c087ab397ef5dcc6c077516f44915dc5.jpg: 416x416 2 helmets, Done. (0.043s)
    image 593/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005890_jpg.rf.9a7d9c3220ee008c48b1b86f6d99f892.jpg: 416x416 2 helmets, Done. (0.039s)
    image 594/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005891_jpg.rf.24a9cf6d080b4c5bb0468affc470dcda.jpg: 416x416 1 helmet, Done. (0.042s)
    image 595/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005892_jpg.rf.bc0b7e07ef6f7296baac5a9da0ee971f.jpg: 416x416 5 helmets, Done. (0.040s)
    image 596/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005893_jpg.rf.7bf6905cbc0d495a52eeb4b061904f0e.jpg: 416x416 2 helmets, Done. (0.039s)
    image 597/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005894_jpg.rf.8fa1c516784223066ca6533a7988729f.jpg: 416x416 4 helmets, Done. (0.043s)
    image 598/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005895_jpg.rf.5cdc7141a2862f75735ab3c20313be0f.jpg: 416x416 2 helmets, Done. (0.037s)
    image 599/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005896_jpg.rf.d6d2c1e71884fa9598802185637a18d2.jpg: 416x416 1 helmet, Done. (0.038s)
    image 600/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005897_jpg.rf.a5b512a0424717e63b5404abe4248bbf.jpg: 416x416 5 helmets, Done. (0.036s)
    image 601/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005898_jpg.rf.e849d5fbaa50af10c24fcf7d6c61f509.jpg: 416x416 3 helmets, Done. (0.036s)
    image 602/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005899_jpg.rf.b6358c36a689ed72e1e85ce19a5ca33b.jpg: 416x416 10 helmets, Done. (0.037s)
    image 603/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005900_jpg.rf.aa6a684b4e160c89ad128d5a39690954.jpg: 416x416 2 helmets, Done. (0.038s)
    image 604/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005901_jpg.rf.78d2ae05f4235c128c8f9a286895a946.jpg: 416x416 2 helmets, Done. (0.036s)
    image 605/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005902_jpg.rf.c127e0fb8cd4b8334371a0dbbb7847bf.jpg: 416x416 5 heads, 1 helmet, Done. (0.037s)
    image 606/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005903_jpg.rf.972824b23dcf50fdfee4eb153d7aa4f8.jpg: 416x416 1 helmet, Done. (0.038s)
    image 607/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005904_jpg.rf.9256122dcc44a37c2f5fed1c335d4949.jpg: 416x416 2 helmets, Done. (0.038s)
    image 608/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005905_jpg.rf.3b47ba0e6fe593ebaadeccd95f3794ba.jpg: 416x416 1 helmet, Done. (0.039s)
    image 609/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005906_jpg.rf.40babcc7fa3665275a1a694d62fe2bea.jpg: 416x416 2 helmets, Done. (0.039s)
    image 610/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005907_jpg.rf.3d68480055ed4ca470ffecf23b69cd94.jpg: 416x416 5 helmets, Done. (0.039s)
    image 611/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005908_jpg.rf.fcc43b1042c80a15e2d189e353f110fe.jpg: 416x416 2 helmets, Done. (0.038s)
    image 612/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005909_jpg.rf.a1f47973034ca36571f952913749d5b6.jpg: 416x416 1 helmet, Done. (0.043s)
    image 613/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005910_jpg.rf.a6a001c0eba3f64b467f02c4c6bb9d81.jpg: 416x416 8 helmets, Done. (0.044s)
    image 614/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005911_jpg.rf.10463492464ddb65e71c0c4451bc1b84.jpg: 416x416 2 helmets, Done. (0.038s)
    image 615/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005912_jpg.rf.a46a1abb27865023d2845f634b89f3c8.jpg: 416x416 5 heads, 1 helmet, Done. (0.037s)
    image 616/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005913_jpg.rf.8cc89280d74ec22973ff0eb34b84678e.jpg: 416x416 1 helmet, Done. (0.042s)
    image 617/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005914_jpg.rf.3206a254cb025683e8bbbb975a5c0eda.jpg: 416x416 3 helmets, Done. (0.043s)
    image 618/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005915_jpg.rf.5a06708ab280731d4142faa7dae78221.jpg: 416x416 4 helmets, Done. (0.045s)
    image 619/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005916_jpg.rf.bc9d3ad9a19b477523fdf4825ed81917.jpg: 416x416 1 helmet, Done. (0.038s)
    image 620/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005917_jpg.rf.a8e24fd60457af17b4f47b7191a26fb0.jpg: 416x416 2 helmets, Done. (0.036s)
    image 621/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005918_jpg.rf.b8e0d7072df31cbea107eaea93621c12.jpg: 416x416 2 helmets, Done. (0.036s)
    image 622/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005919_jpg.rf.0d040c4dd981d7c2de4a7814d2d526dc.jpg: 416x416 12 heads, Done. (0.036s)
    image 623/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005920_jpg.rf.0732617d2b58af652779fbbc99642fd2.jpg: 416x416 1 helmet, Done. (0.036s)
    image 624/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005921_jpg.rf.4e604cb56bbe4335b69011749cb9b708.jpg: 416x416 4 helmets, Done. (0.039s)
    image 625/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005922_jpg.rf.91de2e21f84a09a27fc15814d118da9c.jpg: 416x416 2 helmets, Done. (0.036s)
    image 626/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005923_jpg.rf.c5428e6d5dde026cfb01dd6544591683.jpg: 416x416 3 helmets, Done. (0.039s)
    image 627/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005924_jpg.rf.02e6386d32a1cbb3fd0f07dd5eb8d2c3.jpg: 416x416 3 helmets, Done. (0.037s)
    image 628/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005925_jpg.rf.0f3fc47ba57375befcecd2d13563bbf2.jpg: 416x416 1 helmet, Done. (0.039s)
    image 629/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005926_jpg.rf.136dc6b421aefb28dc098bc182fd7427.jpg: 416x416 1 helmet, Done. (0.034s)
    image 630/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005927_jpg.rf.5920ecc13688bdec5e4494761b29d2ed.jpg: 416x416 2 helmets, Done. (0.035s)
    image 631/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005928_jpg.rf.83f6390147105595096ca604e3d4fad3.jpg: 416x416 1 helmet, Done. (0.035s)
    image 632/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005929_jpg.rf.890eff44e6cfd05d0be23e99c0139c70.jpg: 416x416 9 heads, 1 helmet, Done. (0.038s)
    image 633/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005930_jpg.rf.24e494855b8c5f2d33f34f11aa9f5891.jpg: 416x416 6 heads, Done. (0.042s)
    image 634/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005931_jpg.rf.c3bc1469bdbaa063d9494863c3747132.jpg: 416x416 7 heads, Done. (0.041s)
    image 635/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005932_jpg.rf.8d0d0a0fbe36d5b35d588d4be34631e8.jpg: 416x416 1 helmet, Done. (0.042s)
    image 636/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005933_jpg.rf.7b83bc5c4495c822d5f85b87216ccf78.jpg: 416x416 2 helmets, Done. (0.038s)
    image 637/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005934_jpg.rf.d054768577c7ebd1782ac0c717a41f5e.jpg: 416x416 11 heads, Done. (0.041s)
    image 638/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005935_jpg.rf.f018ab32fcf7e79fd2de1f1f8e8f7b14.jpg: 416x416 2 helmets, Done. (0.039s)
    image 639/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005936_jpg.rf.de4eb81d22125faa33a45502f3c908c0.jpg: 416x416 4 helmets, Done. (0.040s)
    image 640/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005937_jpg.rf.28845b753b3ff022188ca6d70fafb3fa.jpg: 416x416 2 helmets, Done. (0.040s)
    image 641/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005938_jpg.rf.a739652c9154405ba4110168d45eb59b.jpg: 416x416 4 helmets, Done. (0.036s)
    image 642/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005939_jpg.rf.b90204f60ae18812ed82eb6597e35730.jpg: 416x416 11 helmets, Done. (0.036s)
    image 643/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005940_jpg.rf.3a0b6f318f4c4a786808113298662681.jpg: 416x416 1 head, 7 helmets, Done. (0.037s)
    image 644/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005941_jpg.rf.84c8b50cec94ba7f967a1bfa1b33dfa9.jpg: 416x416 1 helmet, Done. (0.038s)
    image 645/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005942_jpg.rf.96c1565ccfa467db731c90d4e1a85ea7.jpg: 416x416 3 helmets, Done. (0.039s)
    image 646/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005943_jpg.rf.0f09bdb9db409bc3537d4fb4b4ab3680.jpg: 416x416 2 helmets, Done. (0.041s)
    image 647/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005944_jpg.rf.7380030ca3c3789c4e40e5ba63e55bc4.jpg: 416x416 5 helmets, Done. (0.037s)
    image 648/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005945_jpg.rf.5480ba2d3afcadc5d1a4400a29bf2111.jpg: 416x416 1 helmet, Done. (0.036s)
    image 649/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005946_jpg.rf.70b215cb5f23d14bd1b7b444a118ff46.jpg: 416x416 4 helmets, Done. (0.037s)
    image 650/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005947_jpg.rf.de81968e72921c5010e1e29b73015d87.jpg: 416x416 3 helmets, Done. (0.042s)
    image 651/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005948_jpg.rf.ab207c2f1177d9e1cbf8c89fb89df2b0.jpg: 416x416 1 helmet, Done. (0.037s)
    image 652/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005949_jpg.rf.28dc7da8a306a23ea445faf8206d8ea0.jpg: 416x416 15 heads, Done. (0.038s)
    image 653/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005950_jpg.rf.0ef05d0ad108fb5f1b8453b99b0ddb66.jpg: 416x416 5 helmets, Done. (0.042s)
    image 654/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005951_jpg.rf.0c6a7ac1726856040dd43d2d8ae9dd33.jpg: 416x416 2 helmets, Done. (0.042s)
    image 655/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005952_jpg.rf.b7b95c43e58e60af1b1afdeb69997031.jpg: 416x416 1 head, 1 helmet, Done. (0.044s)
    image 656/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005953_jpg.rf.0087a900f13f806899735c395da37ceb.jpg: 416x416 1 head, 7 helmets, Done. (0.045s)
    image 657/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005954_jpg.rf.96bec24a24aa0879e141769c895bcc8d.jpg: 416x416 3 helmets, Done. (0.042s)
    image 658/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005955_jpg.rf.66ff0f09b35a5e22760d74f1e8593cf9.jpg: 416x416 2 helmets, Done. (0.042s)
    image 659/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005956_jpg.rf.532f82f540462067b6b5afdde79bdef4.jpg: 416x416 1 helmet, Done. (0.042s)
    image 660/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005957_jpg.rf.4fa8b462da56a3e5d4bc2c0252a7cbd5.jpg: 416x416 1 helmet, Done. (0.043s)
    image 661/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005958_jpg.rf.2e616e18eed6f06af51d43c7822e07f3.jpg: 416x416 3 helmets, Done. (0.045s)
    image 662/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005959_jpg.rf.3d069f0b816d8d91cb00bc3012965371.jpg: 416x416 1 helmet, Done. (0.043s)
    image 663/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005960_jpg.rf.612a3c89e44a92cf1be6c517e2b347a3.jpg: 416x416 1 helmet, Done. (0.045s)
    image 664/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005961_jpg.rf.d8997791eafd2537791bb9da7a42c15b.jpg: 416x416 10 helmets, Done. (0.043s)
    image 665/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005962_jpg.rf.f3407865113eff5fbeec1c570d36b4a5.jpg: 416x416 4 helmets, Done. (0.044s)
    image 666/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005963_jpg.rf.93f84f8b4ae28143fdc46a142f4b1e6c.jpg: 416x416 2 helmets, Done. (0.045s)
    image 667/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005964_jpg.rf.224b293ebb84ba4f04e41b2fee7640b6.jpg: 416x416 3 helmets, Done. (0.044s)
    image 668/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005965_jpg.rf.548f78b24880dacb496b0354d3b4291f.jpg: 416x416 3 helmets, Done. (0.042s)
    image 669/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005966_jpg.rf.81f2813b21112bde80ae54a46c7ba33a.jpg: 416x416 2 helmets, Done. (0.045s)
    image 670/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005967_jpg.rf.93a33f084679d5e65778b62580fc9acf.jpg: 416x416 2 helmets, Done. (0.046s)
    image 671/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005968_jpg.rf.f90e3192cd9008badb5107f182068377.jpg: 416x416 1 helmet, Done. (0.047s)
    image 672/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005969_jpg.rf.39b3d1c01cf4b96979694cab69783134.jpg: 416x416 2 helmets, Done. (0.042s)
    image 673/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005970_jpg.rf.a58758c086e953cd337799c0e495b19e.jpg: 416x416 2 heads, 1 helmet, Done. (0.045s)
    image 674/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005971_jpg.rf.2ea63911fb44e74a0e18bb016b1e9070.jpg: 416x416 1 helmet, Done. (0.051s)
    image 675/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005972_jpg.rf.d30a2e7064c044b01cbebe4891f42bb7.jpg: 416x416 1 helmet, Done. (0.050s)
    image 676/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005973_jpg.rf.5016fc348fa2a0129f0c09e0e2d1292d.jpg: 416x416 7 helmets, Done. (0.045s)
    image 677/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005974_jpg.rf.b82282f8f75830427e80f65ecb88cb58.jpg: 416x416 1 head, 6 helmets, Done. (0.044s)
    image 678/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005975_jpg.rf.8998fb232605f216c6cbe6f173c41397.jpg: 416x416 2 heads, 1 helmet, Done. (0.045s)
    image 679/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005976_jpg.rf.7366146e46790d5f75667e99db3ac588.jpg: 416x416 1 helmet, Done. (0.046s)
    image 680/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005977_jpg.rf.3ed63fd7c442bf4c49e32f7ca171dab8.jpg: 416x416 1 helmet, Done. (0.044s)
    image 681/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005978_jpg.rf.0c2501773e3f85736c535e0df4617e1e.jpg: 416x416 12 helmets, Done. (0.039s)
    image 682/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005979_jpg.rf.1e79c7d4b17954db65825eaa605ab9a9.jpg: 416x416 3 helmets, Done. (0.036s)
    image 683/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005980_jpg.rf.64c720282816e790fdd283e9dae26b83.jpg: 416x416 2 helmets, Done. (0.038s)
    image 684/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005981_jpg.rf.92317c747c32c09745202e97f2223329.jpg: 416x416 5 helmets, Done. (0.043s)
    image 685/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005982_jpg.rf.85efd0a64c73eb6045fc75beeffcfc7a.jpg: 416x416 2 helmets, Done. (0.038s)
    image 686/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005983_jpg.rf.87e4bc3baa0bf272f05d714990dfe3bf.jpg: 416x416 2 helmets, Done. (0.039s)
    image 687/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005984_jpg.rf.804640ceb0672d5818b70965cf2d74a4.jpg: 416x416 3 helmets, Done. (0.040s)
    image 688/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005985_jpg.rf.f4320e46e3cd5d0681e298add811382c.jpg: 416x416 2 helmets, Done. (0.042s)
    image 689/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005986_jpg.rf.4e50da12ff483eefa51c619c1bb3b6d4.jpg: 416x416 6 helmets, Done. (0.045s)
    image 690/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005987_jpg.rf.65eb033aeb052c0ac2c9fb61f33f71bc.jpg: 416x416 14 heads, Done. (0.045s)
    image 691/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005988_jpg.rf.4fab067bc5892bc146280893ad452dbf.jpg: 416x416 13 heads, Done. (0.043s)
    image 692/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005989_jpg.rf.073ae0b120f04a63d1ae1da30443367b.jpg: 416x416 3 helmets, Done. (0.044s)
    image 693/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005990_jpg.rf.f159c29f25cb4c0e0bd1a88909aefea6.jpg: 416x416 9 helmets, Done. (0.043s)
    image 694/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005991_jpg.rf.46ec2f9d612581a8bf84a8ac8245fe20.jpg: 416x416 1 helmet, Done. (0.041s)
    image 695/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005992_jpg.rf.aabc5c8b3ad8d86e60552d5eed9bfe8c.jpg: 416x416 1 helmet, Done. (0.043s)
    image 696/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005993_jpg.rf.b44e708863b5b78005b8a6b36718089d.jpg: 416x416 2 helmets, Done. (0.038s)
    image 697/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005994_jpg.rf.7751f1536b904dc45c9ccdea6d89be6d.jpg: 416x416 4 helmets, Done. (0.037s)
    image 698/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005995_jpg.rf.40c2a396511845bb33b3a3af412e9e7f.jpg: 416x416 5 helmets, Done. (0.037s)
    image 699/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005996_jpg.rf.0fffc23815b1d196a48d85e2511780ef.jpg: 416x416 2 helmets, Done. (0.038s)
    image 700/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005997_jpg.rf.797a8206adec7c9fa3856ccd00fb20fa.jpg: 416x416 1 helmet, Done. (0.034s)
    image 701/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005998_jpg.rf.e7fafd0ee0dd4248296e77a578fe7bdf.jpg: 416x416 2 helmets, Done. (0.037s)
    image 702/706 /content/yolov5/Hard-Hat-Workers-13/test/images/005999_jpg.rf.bfd278c1fe9d42a052b40d5c3fa03e6d.jpg: 416x416 10 heads, Done. (0.033s)
    image 703/706 /content/yolov5/Hard-Hat-Workers-13/test/images/006000_jpg.rf.b5ef6634f75f00d08ca93ec25d31fd6f.jpg: 416x416 2 helmets, Done. (0.033s)
    image 704/706 /content/yolov5/Hard-Hat-Workers-13/test/images/006001_jpg.rf.4d09405a9e96a9de7a402735a0e70e19.jpg: 416x416 2 helmets, Done. (0.032s)
    image 705/706 /content/yolov5/Hard-Hat-Workers-13/test/images/006002_jpg.rf.ae0ce1a4db8c752334dfc68c567e7a5d.jpg: 416x416 2 helmets, Done. (0.034s)
    image 706/706 /content/yolov5/Hard-Hat-Workers-13/test/images/006003_jpg.rf.05c0e24b36b2b85449921c35259aab39.jpg: 416x416 6 helmets, Done. (0.033s)
    Speed: 0.2ms pre-process, 37.1ms inference, 0.8ms NMS per image at shape (1, 3, 416, 416)
    Results saved to [1mruns/detect/exp[0m
    699 labels saved to runs/detect/exp/labels
    


```python
#이미지 탐색결과 exp폴더에 저장
import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'):
    display(Image(filename=imageName))
    print("\n")
```


    
![jpeg](output_12_0.jpg)
    



    
![jpeg](output_12_1.jpg)
    



    
![jpeg](output_12_2.jpg)
    



    
![jpeg](output_12_3.jpg)
    


    
    
    
    
    
    
    
    
    


    
![jpeg](output_12_5.jpg)
    


    
    
    


    
![jpeg](output_12_7.jpg)
    


    
    
    


    
![jpeg](output_12_9.jpg)
    


    
    
    


    
![jpeg](output_12_11.jpg)
    


    
    
    


    
![jpeg](output_12_13.jpg)
    


    
    
    


    
![jpeg](output_12_15.jpg)
    


    
    
    


    
![jpeg](output_12_17.jpg)
    


    
    
    


    
![jpeg](output_12_19.jpg)
    


    
    
    


    
![jpeg](output_12_21.jpg)
    


    
    
    


    
![jpeg](output_12_23.jpg)
    


    
    
    


    
![jpeg](output_12_25.jpg)
    


    
    
    


    
![jpeg](output_12_27.jpg)
    


    
    
    


    
![jpeg](output_12_29.jpg)
    


    
    
    


    
![jpeg](output_12_31.jpg)
    


    
    
    


    
![jpeg](output_12_33.jpg)
    


    
    
    


    
![jpeg](output_12_35.jpg)
    


    
    
    


    
![jpeg](output_12_37.jpg)
    


    
    
    


    
![jpeg](output_12_39.jpg)
    


    
    
    


    
![jpeg](output_12_41.jpg)
    


    
    
    


    
![jpeg](output_12_43.jpg)
    


    
    
    


    
![jpeg](output_12_45.jpg)
    


    
    
    


    
![jpeg](output_12_47.jpg)
    


    
    
    


    
![jpeg](output_12_49.jpg)
    


    
    
    


    
![jpeg](output_12_51.jpg)
    


    
    
    


    
![jpeg](output_12_53.jpg)
    


    
    
    


    
![jpeg](output_12_55.jpg)
    


    
    
    


    
![jpeg](output_12_57.jpg)
    


    
    
    


    
![jpeg](output_12_59.jpg)
    


    
    
    


    
![jpeg](output_12_61.jpg)
    


    
    
    


    
![jpeg](output_12_63.jpg)
    


    
    
    


    
![jpeg](output_12_65.jpg)
    


    
    
    


    
![jpeg](output_12_67.jpg)
    


    
    
    


    
![jpeg](output_12_69.jpg)
    


    
    
    


    
![jpeg](output_12_71.jpg)
    


    
    
    


    
![jpeg](output_12_73.jpg)
    


    
    
    


    
![jpeg](output_12_75.jpg)
    


    
    
    


    
![jpeg](output_12_77.jpg)
    


    
    
    


    
![jpeg](output_12_79.jpg)
    


    
    
    


    
![jpeg](output_12_81.jpg)
    


    
    
    


    
![jpeg](output_12_83.jpg)
    


    
    
    


    
![jpeg](output_12_85.jpg)
    


    
    
    


    
![jpeg](output_12_87.jpg)
    


    
    
    


    
![jpeg](output_12_89.jpg)
    


    
    
    


    
![jpeg](output_12_91.jpg)
    


    
    
    


    
![jpeg](output_12_93.jpg)
    


    
    
    


    
![jpeg](output_12_95.jpg)
    


    
    
    


    
![jpeg](output_12_97.jpg)
    


    
    
    


    
![jpeg](output_12_99.jpg)
    


    
    
    


    
![jpeg](output_12_101.jpg)
    


    
    
    


    
![jpeg](output_12_103.jpg)
    


    
    
    


    
![jpeg](output_12_105.jpg)
    


    
    
    


    
![jpeg](output_12_107.jpg)
    


    
    
    


    
![jpeg](output_12_109.jpg)
    


    
    
    


    
![jpeg](output_12_111.jpg)
    


    
    
    


    
![jpeg](output_12_113.jpg)
    


    
    
    


    
![jpeg](output_12_115.jpg)
    


    
    
    


    
![jpeg](output_12_117.jpg)
    


    
    
    


    
![jpeg](output_12_119.jpg)
    


    
    
    


    
![jpeg](output_12_121.jpg)
    


    
    
    


    
![jpeg](output_12_123.jpg)
    


    
    
    


    
![jpeg](output_12_125.jpg)
    


    
    
    


    
![jpeg](output_12_127.jpg)
    


    
    
    


    
![jpeg](output_12_129.jpg)
    


    
    
    


    
![jpeg](output_12_131.jpg)
    


    
    
    


    
![jpeg](output_12_133.jpg)
    


    
    
    


    
![jpeg](output_12_135.jpg)
    


    
    
    


    
![jpeg](output_12_137.jpg)
    


    
    
    


    
![jpeg](output_12_139.jpg)
    


    
    
    


    
![jpeg](output_12_141.jpg)
    


    
    
    


    
![jpeg](output_12_143.jpg)
    


    
    
    


    
![jpeg](output_12_145.jpg)
    


    
    
    


    
![jpeg](output_12_147.jpg)
    


    
    
    


    
![jpeg](output_12_149.jpg)
    


    
    
    


    
![jpeg](output_12_151.jpg)
    


    
    
    


    
![jpeg](output_12_153.jpg)
    


    
    
    


    
![jpeg](output_12_155.jpg)
    


    
    
    


    
![jpeg](output_12_157.jpg)
    


    
    
    


    
![jpeg](output_12_159.jpg)
    


    
    
    


    
![jpeg](output_12_161.jpg)
    


    
    
    


    
![jpeg](output_12_163.jpg)
    


    
    
    


    
![jpeg](output_12_165.jpg)
    


    
    
    


    
![jpeg](output_12_167.jpg)
    


    
    
    


    
![jpeg](output_12_169.jpg)
    


    
    
    


    
![jpeg](output_12_171.jpg)
    


    
    
    


    
![jpeg](output_12_173.jpg)
    


    
    
    


    
![jpeg](output_12_175.jpg)
    


    
    
    


    
![jpeg](output_12_177.jpg)
    


    
    
    


    
![jpeg](output_12_179.jpg)
    


    
    
    


    
![jpeg](output_12_181.jpg)
    


    
    
    


    
![jpeg](output_12_183.jpg)
    


    
    
    


    
![jpeg](output_12_185.jpg)
    


    
    
    


    
![jpeg](output_12_187.jpg)
    


    
    
    


    
![jpeg](output_12_189.jpg)
    


    
    
    


    
![jpeg](output_12_191.jpg)
    


    
    
    


    
![jpeg](output_12_193.jpg)
    


    
    
    


    
![jpeg](output_12_195.jpg)
    


    
    
    


    
![jpeg](output_12_197.jpg)
    


    
    
    


    
![jpeg](output_12_199.jpg)
    


    
    
    


    
![jpeg](output_12_201.jpg)
    


    
    
    


    
![jpeg](output_12_203.jpg)
    


    
    
    


    
![jpeg](output_12_205.jpg)
    


    
    
    


    
![jpeg](output_12_207.jpg)
    


    
    
    


    
![jpeg](output_12_209.jpg)
    


    
    
    


    
![jpeg](output_12_211.jpg)
    


    
    
    


    
![jpeg](output_12_213.jpg)
    


    
    
    


    
![jpeg](output_12_215.jpg)
    


    
    
    


    
![jpeg](output_12_217.jpg)
    


    
    
    


    
![jpeg](output_12_219.jpg)
    


    
    
    


    
![jpeg](output_12_221.jpg)
    


    
    
    


    
![jpeg](output_12_223.jpg)
    


    
    
    


    
![jpeg](output_12_225.jpg)
    


    
    
    


    
![jpeg](output_12_227.jpg)
    


    
    
    


    
![jpeg](output_12_229.jpg)
    


    
    
    


    
![jpeg](output_12_231.jpg)
    


    
    
    


    
![jpeg](output_12_233.jpg)
    


    
    
    


    
![jpeg](output_12_235.jpg)
    


    
    
    


    
![jpeg](output_12_237.jpg)
    


    
    
    


    
![jpeg](output_12_239.jpg)
    


    
    
    


    
![jpeg](output_12_241.jpg)
    


    
    
    


    
![jpeg](output_12_243.jpg)
    


    
    
    


    
![jpeg](output_12_245.jpg)
    


    
    
    


    
![jpeg](output_12_247.jpg)
    


    
    
    


    
![jpeg](output_12_249.jpg)
    


    
    
    


    
![jpeg](output_12_251.jpg)
    


    
    
    


    
![jpeg](output_12_253.jpg)
    


    
    
    


    
![jpeg](output_12_255.jpg)
    


    
    
    


    
![jpeg](output_12_257.jpg)
    


    
    
    


    
![jpeg](output_12_259.jpg)
    


    
    
    


    
![jpeg](output_12_261.jpg)
    


    
    
    


    
![jpeg](output_12_263.jpg)
    


    
    
    


    
![jpeg](output_12_265.jpg)
    


    
    
    


    
![jpeg](output_12_267.jpg)
    


    
    
    


    
![jpeg](output_12_269.jpg)
    


    
    
    


    
![jpeg](output_12_271.jpg)
    


    
    
    


    
![jpeg](output_12_273.jpg)
    


    
    
    


    
![jpeg](output_12_275.jpg)
    


    
    
    


    
![jpeg](output_12_277.jpg)
    


    
    
    


    
![jpeg](output_12_279.jpg)
    


    
    
    


    
![jpeg](output_12_281.jpg)
    


    
    
    


    
![jpeg](output_12_283.jpg)
    


    
    
    


    
![jpeg](output_12_285.jpg)
    


    
    
    


    
![jpeg](output_12_287.jpg)
    


    
    
    


    
![jpeg](output_12_289.jpg)
    


    
    
    


    
![jpeg](output_12_291.jpg)
    


    
    
    


    
![jpeg](output_12_293.jpg)
    


    
    
    


    
![jpeg](output_12_295.jpg)
    


    
    
    


    
![jpeg](output_12_297.jpg)
    


    
    
    


    
![jpeg](output_12_299.jpg)
    


    
    
    


    
![jpeg](output_12_301.jpg)
    


    
    
    


    
![jpeg](output_12_303.jpg)
    


    
    
    


    
![jpeg](output_12_305.jpg)
    


    
    
    


    
![jpeg](output_12_307.jpg)
    


    
    
    


    
![jpeg](output_12_309.jpg)
    


    
    
    


    
![jpeg](output_12_311.jpg)
    


    
    
    


    
![jpeg](output_12_313.jpg)
    


    
    
    


    
![jpeg](output_12_315.jpg)
    


    
    
    


    
![jpeg](output_12_317.jpg)
    


    
    
    


    
![jpeg](output_12_319.jpg)
    


    
    
    


    
![jpeg](output_12_321.jpg)
    


    
    
    


    
![jpeg](output_12_323.jpg)
    


    
    
    


    
![jpeg](output_12_325.jpg)
    


    
    
    


    
![jpeg](output_12_327.jpg)
    


    
    
    


    
![jpeg](output_12_329.jpg)
    


    
    
    


    
![jpeg](output_12_331.jpg)
    


    
    
    


    
![jpeg](output_12_333.jpg)
    


    
    
    


    
![jpeg](output_12_335.jpg)
    


    
    
    


    
![jpeg](output_12_337.jpg)
    


    
    
    


    
![jpeg](output_12_339.jpg)
    


    
    
    


    
![jpeg](output_12_341.jpg)
    


    
    
    


    
![jpeg](output_12_343.jpg)
    


    
    
    


    
![jpeg](output_12_345.jpg)
    


    
    
    


    
![jpeg](output_12_347.jpg)
    


    
    
    


    
![jpeg](output_12_349.jpg)
    


    
    
    


    
![jpeg](output_12_351.jpg)
    


    
    
    


    
![jpeg](output_12_353.jpg)
    


    
    
    


    
![jpeg](output_12_355.jpg)
    


    
    
    


    
![jpeg](output_12_357.jpg)
    


    
    
    


    
![jpeg](output_12_359.jpg)
    


    
    
    


    
![jpeg](output_12_361.jpg)
    


    
    
    


    
![jpeg](output_12_363.jpg)
    


    
    
    


    
![jpeg](output_12_365.jpg)
    


    
    
    


    
![jpeg](output_12_367.jpg)
    


    
    
    


    
![jpeg](output_12_369.jpg)
    


    
    
    


    
![jpeg](output_12_371.jpg)
    


    
    
    


    
![jpeg](output_12_373.jpg)
    


    
    
    


    
![jpeg](output_12_375.jpg)
    


    
    
    


    
![jpeg](output_12_377.jpg)
    


    
    
    


    
![jpeg](output_12_379.jpg)
    


    
    
    


    
![jpeg](output_12_381.jpg)
    


    
    
    


    
![jpeg](output_12_383.jpg)
    


    
    
    


    
![jpeg](output_12_385.jpg)
    


    
    
    


    
![jpeg](output_12_387.jpg)
    


    
    
    


    
![jpeg](output_12_389.jpg)
    


    
    
    


    
![jpeg](output_12_391.jpg)
    


    
    
    


    
![jpeg](output_12_393.jpg)
    


    
    
    


    
![jpeg](output_12_395.jpg)
    


    
    
    


    
![jpeg](output_12_397.jpg)
    


    
    
    


    
![jpeg](output_12_399.jpg)
    


    
    
    


    
![jpeg](output_12_401.jpg)
    


    
    
    


    
![jpeg](output_12_403.jpg)
    


    
    
    


    
![jpeg](output_12_405.jpg)
    


    
    
    


    
![jpeg](output_12_407.jpg)
    


    
    
    


    
![jpeg](output_12_409.jpg)
    


    
    
    


    
![jpeg](output_12_411.jpg)
    


    
    
    


    
![jpeg](output_12_413.jpg)
    


    
    
    


    
![jpeg](output_12_415.jpg)
    


    
    
    


    
![jpeg](output_12_417.jpg)
    


    
    
    


    
![jpeg](output_12_419.jpg)
    


    
    
    


    
![jpeg](output_12_421.jpg)
    


    
    
    


    
![jpeg](output_12_423.jpg)
    


    
    
    


    
![jpeg](output_12_425.jpg)
    


    
    
    


    
![jpeg](output_12_427.jpg)
    


    
    
    


    
![jpeg](output_12_429.jpg)
    


    
    
    


    
![jpeg](output_12_431.jpg)
    


    
    
    


    
![jpeg](output_12_433.jpg)
    


    
    
    


    
![jpeg](output_12_435.jpg)
    


    
    
    


    
![jpeg](output_12_437.jpg)
    


    
    
    


    
![jpeg](output_12_439.jpg)
    


    
    
    


    
![jpeg](output_12_441.jpg)
    


    
    
    


    
![jpeg](output_12_443.jpg)
    


    
    
    


    
![jpeg](output_12_445.jpg)
    


    
    
    


    
![jpeg](output_12_447.jpg)
    


    
    
    


    
![jpeg](output_12_449.jpg)
    


    
    
    


    
![jpeg](output_12_451.jpg)
    


    
    
    


    
![jpeg](output_12_453.jpg)
    


    
    
    


    
![jpeg](output_12_455.jpg)
    


    
    
    


    
![jpeg](output_12_457.jpg)
    


    
    
    


    
![jpeg](output_12_459.jpg)
    


    
    
    


    
![jpeg](output_12_461.jpg)
    


    
    
    


    
![jpeg](output_12_463.jpg)
    


    
    
    


    
![jpeg](output_12_465.jpg)
    


    
    
    


    
![jpeg](output_12_467.jpg)
    


    
    
    


    
![jpeg](output_12_469.jpg)
    


    
    
    


    
![jpeg](output_12_471.jpg)
    


    
    
    


    
![jpeg](output_12_473.jpg)
    


    
    
    


    
![jpeg](output_12_475.jpg)
    


    
    
    


    
![jpeg](output_12_477.jpg)
    


    
    
    


    
![jpeg](output_12_479.jpg)
    


    
    
    


    
![jpeg](output_12_481.jpg)
    


    
    
    


    
![jpeg](output_12_483.jpg)
    


    
    
    


    
![jpeg](output_12_485.jpg)
    


    
    
    


    
![jpeg](output_12_487.jpg)
    


    
    
    


    
![jpeg](output_12_489.jpg)
    


    
    
    


    
![jpeg](output_12_491.jpg)
    


    
    
    


    
![jpeg](output_12_493.jpg)
    


    
    
    


    
![jpeg](output_12_495.jpg)
    


    
    
    


    
![jpeg](output_12_497.jpg)
    


    
    
    


    
![jpeg](output_12_499.jpg)
    


    
    
    


    
![jpeg](output_12_501.jpg)
    


    
    
    


    
![jpeg](output_12_503.jpg)
    


    
    
    


    
![jpeg](output_12_505.jpg)
    


    
    
    


    
![jpeg](output_12_507.jpg)
    


    
    
    


    
![jpeg](output_12_509.jpg)
    


    
    
    


    
![jpeg](output_12_511.jpg)
    


    
    
    


    
![jpeg](output_12_513.jpg)
    


    
    
    


    
![jpeg](output_12_515.jpg)
    


    
    
    


    
![jpeg](output_12_517.jpg)
    


    
    
    


    
![jpeg](output_12_519.jpg)
    


    
    
    


    
![jpeg](output_12_521.jpg)
    


    
    
    


    
![jpeg](output_12_523.jpg)
    


    
    
    


    
![jpeg](output_12_525.jpg)
    


    
    
    


    
![jpeg](output_12_527.jpg)
    


    
    
    


    
![jpeg](output_12_529.jpg)
    


    
    
    


    
![jpeg](output_12_531.jpg)
    


    
    
    


    
![jpeg](output_12_533.jpg)
    


    
    
    


    
![jpeg](output_12_535.jpg)
    


    
    
    


    
![jpeg](output_12_537.jpg)
    


    
    
    


    
![jpeg](output_12_539.jpg)
    


    
    
    


    
![jpeg](output_12_541.jpg)
    


    
    
    


    
![jpeg](output_12_543.jpg)
    


    
    
    


    
![jpeg](output_12_545.jpg)
    


    
    
    


    
![jpeg](output_12_547.jpg)
    


    
    
    


    
![jpeg](output_12_549.jpg)
    


    
    
    


    
![jpeg](output_12_551.jpg)
    


    
    
    


    
![jpeg](output_12_553.jpg)
    


    
    
    


    
![jpeg](output_12_555.jpg)
    


    
    
    


    
![jpeg](output_12_557.jpg)
    


    
    
    


    
![jpeg](output_12_559.jpg)
    


    
    
    


    
![jpeg](output_12_561.jpg)
    


    
    
    


    
![jpeg](output_12_563.jpg)
    


    
    
    


    
![jpeg](output_12_565.jpg)
    


    
    
    


    
![jpeg](output_12_567.jpg)
    


    
    
    


    
![jpeg](output_12_569.jpg)
    


    
    
    


    
![jpeg](output_12_571.jpg)
    


    
    
    


    
![jpeg](output_12_573.jpg)
    


    
    
    


    
![jpeg](output_12_575.jpg)
    


    
    
    


    
![jpeg](output_12_577.jpg)
    


    
    
    


    
![jpeg](output_12_579.jpg)
    


    
    
    


    
![jpeg](output_12_581.jpg)
    


    
    
    


    
![jpeg](output_12_583.jpg)
    


    
    
    


    
![jpeg](output_12_585.jpg)
    


    
    
    


    
![jpeg](output_12_587.jpg)
    


    
    
    


    
![jpeg](output_12_589.jpg)
    


    
    
    


    
![jpeg](output_12_591.jpg)
    


    
    
    


    
![jpeg](output_12_593.jpg)
    


    
    
    


    
![jpeg](output_12_595.jpg)
    


    
    
    


    
![jpeg](output_12_597.jpg)
    


    
    
    


    
![jpeg](output_12_599.jpg)
    


    
    
    


    
![jpeg](output_12_601.jpg)
    


    
    
    


    
![jpeg](output_12_603.jpg)
    


    
    
    


    
![jpeg](output_12_605.jpg)
    


    
    
    


    
![jpeg](output_12_607.jpg)
    


    
    
    


    
![jpeg](output_12_609.jpg)
    


    
    
    


    
![jpeg](output_12_611.jpg)
    


    
    
    


    
![jpeg](output_12_613.jpg)
    


    
    
    


    
![jpeg](output_12_615.jpg)
    


    
    
    


    
![jpeg](output_12_617.jpg)
    


    
    
    


    
![jpeg](output_12_619.jpg)
    


    
    
    


    
![jpeg](output_12_621.jpg)
    


    
    
    


    
![jpeg](output_12_623.jpg)
    


    
    
    


    
![jpeg](output_12_625.jpg)
    


    
    
    


    
![jpeg](output_12_627.jpg)
    


    
    
    


    
![jpeg](output_12_629.jpg)
    


    
    
    


    
![jpeg](output_12_631.jpg)
    


    
    
    


    
![jpeg](output_12_633.jpg)
    


    
    
    


    
![jpeg](output_12_635.jpg)
    


    
    
    


    
![jpeg](output_12_637.jpg)
    


    
    
    


    
![jpeg](output_12_639.jpg)
    


    
    
    


    
![jpeg](output_12_641.jpg)
    


    
    
    


    
![jpeg](output_12_643.jpg)
    


    
    
    


    
![jpeg](output_12_645.jpg)
    


    
    
    


    
![jpeg](output_12_647.jpg)
    


    
    
    


    
![jpeg](output_12_649.jpg)
    


    
    
    


    
![jpeg](output_12_651.jpg)
    


    
    
    


    
![jpeg](output_12_653.jpg)
    


    
    
    


    
![jpeg](output_12_655.jpg)
    


    
    
    


    
![jpeg](output_12_657.jpg)
    


    
    
    


    
![jpeg](output_12_659.jpg)
    


    
    
    


    
![jpeg](output_12_661.jpg)
    


    
    
    


    
![jpeg](output_12_663.jpg)
    


    
    
    


    
![jpeg](output_12_665.jpg)
    


    
    
    


    
![jpeg](output_12_667.jpg)
    


    
    
    


    
![jpeg](output_12_669.jpg)
    


    
    
    


    
![jpeg](output_12_671.jpg)
    


    
    
    


    
![jpeg](output_12_673.jpg)
    


    
    
    


    
![jpeg](output_12_675.jpg)
    


    
    
    


    
![jpeg](output_12_677.jpg)
    


    
    
    


    
![jpeg](output_12_679.jpg)
    


    
    
    


    
![jpeg](output_12_681.jpg)
    


    
    
    


    
![jpeg](output_12_683.jpg)
    


    
    
    


    
![jpeg](output_12_685.jpg)
    


    
    
    


    
![jpeg](output_12_687.jpg)
    


    
    
    


    
![jpeg](output_12_689.jpg)
    


    
    
    


    
![jpeg](output_12_691.jpg)
    


    
    
    


    
![jpeg](output_12_693.jpg)
    


    
    
    


    
![jpeg](output_12_695.jpg)
    


    
    
    


    
![jpeg](output_12_697.jpg)
    


    
    
    


    
![jpeg](output_12_699.jpg)
    


    
    
    


    
![jpeg](output_12_701.jpg)
    


    
    
    


    
![jpeg](output_12_703.jpg)
    


    
    
    


    
![jpeg](output_12_705.jpg)
    


    
    
    


    
![jpeg](output_12_707.jpg)
    


    
    
    


    
![jpeg](output_12_709.jpg)
    


    
    
    


    
![jpeg](output_12_711.jpg)
    


    
    
    


    
![jpeg](output_12_713.jpg)
    


    
    
    


    
![jpeg](output_12_715.jpg)
    


    
    
    


    
![jpeg](output_12_717.jpg)
    


    
    
    


    
![jpeg](output_12_719.jpg)
    


    
    
    


    
![jpeg](output_12_721.jpg)
    


    
    
    


    
![jpeg](output_12_723.jpg)
    


    
    
    


    
![jpeg](output_12_725.jpg)
    


    
    
    


    
![jpeg](output_12_727.jpg)
    


    
    
    


    
![jpeg](output_12_729.jpg)
    


    
    
    


    
![jpeg](output_12_731.jpg)
    


    
    
    


    
![jpeg](output_12_733.jpg)
    


    
    
    


    
![jpeg](output_12_735.jpg)
    


    
    
    


    
![jpeg](output_12_737.jpg)
    


    
    
    


    
![jpeg](output_12_739.jpg)
    


    
    
    


    
![jpeg](output_12_741.jpg)
    


    
    
    


    
![jpeg](output_12_743.jpg)
    


    
    
    


    
![jpeg](output_12_745.jpg)
    


    
    
    


    
![jpeg](output_12_747.jpg)
    


    
    
    


    
![jpeg](output_12_749.jpg)
    


    
    
    


    
![jpeg](output_12_751.jpg)
    


    
    
    


    
![jpeg](output_12_753.jpg)
    


    
    
    


    
![jpeg](output_12_755.jpg)
    


    
    
    


    
![jpeg](output_12_757.jpg)
    


    
    
    


    
![jpeg](output_12_759.jpg)
    


    
    
    


    
![jpeg](output_12_761.jpg)
    


    
    
    


    
![jpeg](output_12_763.jpg)
    


    
    
    


    
![jpeg](output_12_765.jpg)
    


    
    
    


    
![jpeg](output_12_767.jpg)
    


    
    
    


    
![jpeg](output_12_769.jpg)
    


    
    
    


    
![jpeg](output_12_771.jpg)
    


    
    
    


    
![jpeg](output_12_773.jpg)
    


    
    
    


    
![jpeg](output_12_775.jpg)
    


    
    
    


    
![jpeg](output_12_777.jpg)
    


    
    
    


    
![jpeg](output_12_779.jpg)
    


    
    
    


    
![jpeg](output_12_781.jpg)
    


    
    
    


    
![jpeg](output_12_783.jpg)
    


    
    
    


    
![jpeg](output_12_785.jpg)
    


    
    
    


    
![jpeg](output_12_787.jpg)
    


    
    
    


    
![jpeg](output_12_789.jpg)
    


    
    
    


    
![jpeg](output_12_791.jpg)
    


    
    
    


    
![jpeg](output_12_793.jpg)
    


    
    
    


    
![jpeg](output_12_795.jpg)
    


    
    
    


    
![jpeg](output_12_797.jpg)
    


    
    
    


    
![jpeg](output_12_799.jpg)
    


    
    
    


    
![jpeg](output_12_801.jpg)
    


    
    
    


    
![jpeg](output_12_803.jpg)
    


    
    
    


    
![jpeg](output_12_805.jpg)
    


    
    
    


    
![jpeg](output_12_807.jpg)
    


    
    
    


    
![jpeg](output_12_809.jpg)
    


    
    
    


    
![jpeg](output_12_811.jpg)
    


    
    
    


    
![jpeg](output_12_813.jpg)
    


    
    
    


    
![jpeg](output_12_815.jpg)
    


    
    
    


    
![jpeg](output_12_817.jpg)
    


    
    
    


    
![jpeg](output_12_819.jpg)
    


    
    
    


    
![jpeg](output_12_821.jpg)
    


    
    
    


    
![jpeg](output_12_823.jpg)
    


    
    
    


    
![jpeg](output_12_825.jpg)
    


    
    
    


    
![jpeg](output_12_827.jpg)
    


    
    
    


    
![jpeg](output_12_829.jpg)
    


    
    
    


    
![jpeg](output_12_831.jpg)
    


    
    
    


    
![jpeg](output_12_833.jpg)
    


    
    
    


    
![jpeg](output_12_835.jpg)
    


    
    
    


    
![jpeg](output_12_837.jpg)
    


    
    
    


    
![jpeg](output_12_839.jpg)
    


    
    
    


    
![jpeg](output_12_841.jpg)
    


    
    
    


    
![jpeg](output_12_843.jpg)
    


    
    
    


    
![jpeg](output_12_845.jpg)
    


    
    
    


    
![jpeg](output_12_847.jpg)
    


    
    
    


    
![jpeg](output_12_849.jpg)
    


    
    
    


    
![jpeg](output_12_851.jpg)
    


    
    
    


    
![jpeg](output_12_853.jpg)
    


    
    
    


    
![jpeg](output_12_855.jpg)
    


    
    
    


    
![jpeg](output_12_857.jpg)
    


    
    
    


    
![jpeg](output_12_859.jpg)
    


    
    
    


    
![jpeg](output_12_861.jpg)
    


    
    
    


    
![jpeg](output_12_863.jpg)
    


    
    
    


    
![jpeg](output_12_865.jpg)
    


    
    
    


    
![jpeg](output_12_867.jpg)
    


    
    
    


    
![jpeg](output_12_869.jpg)
    


    
    
    


    
![jpeg](output_12_871.jpg)
    


    
    
    


    
![jpeg](output_12_873.jpg)
    


    
    
    


    
![jpeg](output_12_875.jpg)
    


    
    
    


    
![jpeg](output_12_877.jpg)
    


    
    
    


    
![jpeg](output_12_879.jpg)
    


    
    
    


    
![jpeg](output_12_881.jpg)
    


    
    
    


    
![jpeg](output_12_883.jpg)
    


    
    
    


    
![jpeg](output_12_885.jpg)
    


    
    
    


    
![jpeg](output_12_887.jpg)
    


    
    
    


    
![jpeg](output_12_889.jpg)
    


    
    
    


    
![jpeg](output_12_891.jpg)
    


    
    
    


    
![jpeg](output_12_893.jpg)
    


    
    
    


    
![jpeg](output_12_895.jpg)
    


    
    
    


    
![jpeg](output_12_897.jpg)
    


    
    
    


    
![jpeg](output_12_899.jpg)
    


    
    
    


    
![jpeg](output_12_901.jpg)
    


    
    
    


    
![jpeg](output_12_903.jpg)
    


    
    
    


    
![jpeg](output_12_905.jpg)
    


    
    
    


    
![jpeg](output_12_907.jpg)
    


    
    
    


    
![jpeg](output_12_909.jpg)
    


    
    
    


    
![jpeg](output_12_911.jpg)
    


    
    
    


    
![jpeg](output_12_913.jpg)
    


    
    
    


    
![jpeg](output_12_915.jpg)
    


    
    
    


    
![jpeg](output_12_917.jpg)
    


    
    
    


    
![jpeg](output_12_919.jpg)
    


    
    
    


    
![jpeg](output_12_921.jpg)
    


    
    
    


    
![jpeg](output_12_923.jpg)
    


    
    
    


    
![jpeg](output_12_925.jpg)
    


    
    
    


    
![jpeg](output_12_927.jpg)
    


    
    
    


    
![jpeg](output_12_929.jpg)
    


    
    
    


    
![jpeg](output_12_931.jpg)
    


    
    
    


    
![jpeg](output_12_933.jpg)
    


    
    
    


    
![jpeg](output_12_935.jpg)
    


    
    
    


    
![jpeg](output_12_937.jpg)
    


    
    
    


    
![jpeg](output_12_939.jpg)
    


    
    
    


    
![jpeg](output_12_941.jpg)
    


    
    
    


    
![jpeg](output_12_943.jpg)
    


    
    
    


    
![jpeg](output_12_945.jpg)
    


    
    
    


    
![jpeg](output_12_947.jpg)
    


    
    
    


    
![jpeg](output_12_949.jpg)
    


    
    
    


    
![jpeg](output_12_951.jpg)
    


    
    
    


    
![jpeg](output_12_953.jpg)
    


    
    
    


    
![jpeg](output_12_955.jpg)
    


    
    
    


    
![jpeg](output_12_957.jpg)
    


    
    
    


    
![jpeg](output_12_959.jpg)
    


    
    
    


    
![jpeg](output_12_961.jpg)
    


    
    
    


    
![jpeg](output_12_963.jpg)
    


    
    
    


    
![jpeg](output_12_965.jpg)
    


    
    
    


    
![jpeg](output_12_967.jpg)
    


    
    
    


    
![jpeg](output_12_969.jpg)
    


    
    
    


    
![jpeg](output_12_971.jpg)
    


    
    
    


    
![jpeg](output_12_973.jpg)
    


    
    
    


    
![jpeg](output_12_975.jpg)
    


    
    
    


    
![jpeg](output_12_977.jpg)
    


    
    
    


    
![jpeg](output_12_979.jpg)
    


    
    
    


    
![jpeg](output_12_981.jpg)
    


    
    
    


    
![jpeg](output_12_983.jpg)
    


    
    
    


    
![jpeg](output_12_985.jpg)
    


    
    
    


    
![jpeg](output_12_987.jpg)
    


    
    
    


    
![jpeg](output_12_989.jpg)
    


    
    
    


    
![jpeg](output_12_991.jpg)
    


    
    
    


    
![jpeg](output_12_993.jpg)
    


    
    
    


    
![jpeg](output_12_995.jpg)
    


    
    
    


    
![jpeg](output_12_997.jpg)
    


    
    
    


    
![jpeg](output_12_999.jpg)
    


    
    
    


    
![jpeg](output_12_1001.jpg)
    


    
    
    


    
![jpeg](output_12_1003.jpg)
    


    
    
    


    
![jpeg](output_12_1005.jpg)
    


    
    
    


    
![jpeg](output_12_1007.jpg)
    


    
    
    


    
![jpeg](output_12_1009.jpg)
    


    
    
    


    
![jpeg](output_12_1011.jpg)
    


    
    
    


    
![jpeg](output_12_1013.jpg)
    


    
    
    


    
![jpeg](output_12_1015.jpg)
    


    
    
    


    
![jpeg](output_12_1017.jpg)
    


    
    
    


    
![jpeg](output_12_1019.jpg)
    


    
    
    


    
![jpeg](output_12_1021.jpg)
    


    
    
    


    
![jpeg](output_12_1023.jpg)
    


    
    
    


    
![jpeg](output_12_1025.jpg)
    


    
    
    


    
![jpeg](output_12_1027.jpg)
    


    
    
    


    
![jpeg](output_12_1029.jpg)
    


    
    
    


    
![jpeg](output_12_1031.jpg)
    


    
    
    


    
![jpeg](output_12_1033.jpg)
    


    
    
    


    
![jpeg](output_12_1035.jpg)
    


    
    
    


    
![jpeg](output_12_1037.jpg)
    


    
    
    


    
![jpeg](output_12_1039.jpg)
    


    
    
    


    
![jpeg](output_12_1041.jpg)
    


    
    
    


    
![jpeg](output_12_1043.jpg)
    


    
    
    


    
![jpeg](output_12_1045.jpg)
    


    
    
    


    
![jpeg](output_12_1047.jpg)
    


    
    
    


    
![jpeg](output_12_1049.jpg)
    


    
    
    


    
![jpeg](output_12_1051.jpg)
    


    
    
    


    
![jpeg](output_12_1053.jpg)
    


    
    
    


    
![jpeg](output_12_1055.jpg)
    


    
    
    


    
![jpeg](output_12_1057.jpg)
    


    
    
    


    
![jpeg](output_12_1059.jpg)
    


    
    
    


    
![jpeg](output_12_1061.jpg)
    


    
    
    


    
![jpeg](output_12_1063.jpg)
    


    
    
    


    
![jpeg](output_12_1065.jpg)
    


    
    
    


    
![jpeg](output_12_1067.jpg)
    


    
    
    


    
![jpeg](output_12_1069.jpg)
    


    
    
    


    
![jpeg](output_12_1071.jpg)
    


    
    
    


    
![jpeg](output_12_1073.jpg)
    


    
    
    


    
![jpeg](output_12_1075.jpg)
    


    
    
    


    
![jpeg](output_12_1077.jpg)
    


    
    
    


    
![jpeg](output_12_1079.jpg)
    


    
    
    


    
![jpeg](output_12_1081.jpg)
    


    
    
    


    
![jpeg](output_12_1083.jpg)
    


    
    
    


    
![jpeg](output_12_1085.jpg)
    


    
    
    


    
![jpeg](output_12_1087.jpg)
    


    
    
    


    
![jpeg](output_12_1089.jpg)
    


    
    
    


    
![jpeg](output_12_1091.jpg)
    


    
    
    


    
![jpeg](output_12_1093.jpg)
    


    
    
    


    
![jpeg](output_12_1095.jpg)
    


    
    
    


    
![jpeg](output_12_1097.jpg)
    


    
    
    


    
![jpeg](output_12_1099.jpg)
    


    
    
    


    
![jpeg](output_12_1101.jpg)
    


    
    
    


    
![jpeg](output_12_1103.jpg)
    


    
    
    


    
![jpeg](output_12_1105.jpg)
    


    
    
    


    
![jpeg](output_12_1107.jpg)
    


    
    
    


    
![jpeg](output_12_1109.jpg)
    


    
    
    


    
![jpeg](output_12_1111.jpg)
    


    
    
    


    
![jpeg](output_12_1113.jpg)
    


    
    
    


    
![jpeg](output_12_1115.jpg)
    


    
    
    


    
![jpeg](output_12_1117.jpg)
    


    
    
    


    
![jpeg](output_12_1119.jpg)
    


    
    
    


    
![jpeg](output_12_1121.jpg)
    


    
    
    


    
![jpeg](output_12_1123.jpg)
    


    
    
    


    
![jpeg](output_12_1125.jpg)
    


    
    
    


    
![jpeg](output_12_1127.jpg)
    


    
    
    


    
![jpeg](output_12_1129.jpg)
    


    
    
    


    
![jpeg](output_12_1131.jpg)
    


    
    
    


    
![jpeg](output_12_1133.jpg)
    


    
    
    


    
![jpeg](output_12_1135.jpg)
    


    
    
    


    
![jpeg](output_12_1137.jpg)
    


    
    
    


    
![jpeg](output_12_1139.jpg)
    


    
    
    


    
![jpeg](output_12_1141.jpg)
    


    
    
    


    
![jpeg](output_12_1143.jpg)
    


    
    
    


    
![jpeg](output_12_1145.jpg)
    


    
    
    


    
![jpeg](output_12_1147.jpg)
    


    
    
    


    
![jpeg](output_12_1149.jpg)
    


    
    
    


    
![jpeg](output_12_1151.jpg)
    


    
    
    


    
![jpeg](output_12_1153.jpg)
    


    
    
    


    
![jpeg](output_12_1155.jpg)
    


    
    
    


    
![jpeg](output_12_1157.jpg)
    


    
    
    


    
![jpeg](output_12_1159.jpg)
    


    
    
    


    
![jpeg](output_12_1161.jpg)
    


    
    
    


    
![jpeg](output_12_1163.jpg)
    


    
    
    


    
![jpeg](output_12_1165.jpg)
    


    
    
    


    
![jpeg](output_12_1167.jpg)
    


    
    
    


    
![jpeg](output_12_1169.jpg)
    


    
    
    


    
![jpeg](output_12_1171.jpg)
    


    
    
    


    
![jpeg](output_12_1173.jpg)
    


    
    
    


    
![jpeg](output_12_1175.jpg)
    


    
    
    


    
![jpeg](output_12_1177.jpg)
    


    
    
    


    
![jpeg](output_12_1179.jpg)
    


    
    
    


    
![jpeg](output_12_1181.jpg)
    


    
    
    


    
![jpeg](output_12_1183.jpg)
    


    
    
    


    
![jpeg](output_12_1185.jpg)
    


    
    
    


    
![jpeg](output_12_1187.jpg)
    


    
    
    


    
![jpeg](output_12_1189.jpg)
    


    
    
    


    
![jpeg](output_12_1191.jpg)
    


    
    
    


    
![jpeg](output_12_1193.jpg)
    


    
    
    


    
![jpeg](output_12_1195.jpg)
    


    
    
    


    
![jpeg](output_12_1197.jpg)
    


    
    
    


    
![jpeg](output_12_1199.jpg)
    


    
    
    


    
![jpeg](output_12_1201.jpg)
    


    
    
    


    
![jpeg](output_12_1203.jpg)
    


    
    
    


    
![jpeg](output_12_1205.jpg)
    


    
    
    


    
![jpeg](output_12_1207.jpg)
    


    
    
    


    
![jpeg](output_12_1209.jpg)
    


    
    
    


    
![jpeg](output_12_1211.jpg)
    


    
    
    


    
![jpeg](output_12_1213.jpg)
    


    
    
    


    
![jpeg](output_12_1215.jpg)
    


    
    
    


    
![jpeg](output_12_1217.jpg)
    


    
    
    


    
![jpeg](output_12_1219.jpg)
    


    
    
    


    
![jpeg](output_12_1221.jpg)
    


    
    
    


    
![jpeg](output_12_1223.jpg)
    


    
    
    


    
![jpeg](output_12_1225.jpg)
    


    
    
    


    
![jpeg](output_12_1227.jpg)
    


    
    
    


    
![jpeg](output_12_1229.jpg)
    


    
    
    


    
![jpeg](output_12_1231.jpg)
    


    
    
    


    
![jpeg](output_12_1233.jpg)
    


    
    
    


    
![jpeg](output_12_1235.jpg)
    


    
    
    


    
![jpeg](output_12_1237.jpg)
    


    
    
    


    
![jpeg](output_12_1239.jpg)
    


    
    
    


    
![jpeg](output_12_1241.jpg)
    


    
    
    


    
![jpeg](output_12_1243.jpg)
    


    
    
    


    
![jpeg](output_12_1245.jpg)
    


    
    
    


    
![jpeg](output_12_1247.jpg)
    


    
    
    


    
![jpeg](output_12_1249.jpg)
    


    
    
    


    
![jpeg](output_12_1251.jpg)
    


    
    
    


    
![jpeg](output_12_1253.jpg)
    


    
    
    


    
![jpeg](output_12_1255.jpg)
    


    
    
    


    
![jpeg](output_12_1257.jpg)
    


    
    
    


    
![jpeg](output_12_1259.jpg)
    


    
    
    


    
![jpeg](output_12_1261.jpg)
    


    
    
    


    
![jpeg](output_12_1263.jpg)
    


    
    
    


    
![jpeg](output_12_1265.jpg)
    


    
    
    


    
![jpeg](output_12_1267.jpg)
    


    
    
    


    
![jpeg](output_12_1269.jpg)
    


    
    
    


    
![jpeg](output_12_1271.jpg)
    


    
    
    


    
![jpeg](output_12_1273.jpg)
    


    
    
    


    
![jpeg](output_12_1275.jpg)
    


    
    
    


    
![jpeg](output_12_1277.jpg)
    


    
    
    


    
![jpeg](output_12_1279.jpg)
    


    
    
    


    
![jpeg](output_12_1281.jpg)
    


    
    
    


    
![jpeg](output_12_1283.jpg)
    


    
    
    


    
![jpeg](output_12_1285.jpg)
    


    
    
    


    
![jpeg](output_12_1287.jpg)
    


    
    
    


    
![jpeg](output_12_1289.jpg)
    


    
    
    


    
![jpeg](output_12_1291.jpg)
    


    
    
    


    
![jpeg](output_12_1293.jpg)
    


    
    
    


    
![jpeg](output_12_1295.jpg)
    


    
    
    


    
![jpeg](output_12_1297.jpg)
    


    
    
    


    
![jpeg](output_12_1299.jpg)
    


    
    
    


    
![jpeg](output_12_1301.jpg)
    


    
    
    


    
![jpeg](output_12_1303.jpg)
    


    
    
    


    
![jpeg](output_12_1305.jpg)
    


    
    
    


    
![jpeg](output_12_1307.jpg)
    


    
    
    


    
![jpeg](output_12_1309.jpg)
    


    
    
    


    
![jpeg](output_12_1311.jpg)
    


    
    
    


    
![jpeg](output_12_1313.jpg)
    


    
    
    


    
![jpeg](output_12_1315.jpg)
    


    
    
    


    
![jpeg](output_12_1317.jpg)
    


    
    
    


    
![jpeg](output_12_1319.jpg)
    


    
    
    


    
![jpeg](output_12_1321.jpg)
    


    
    
    


    
![jpeg](output_12_1323.jpg)
    


    
    
    


    
![jpeg](output_12_1325.jpg)
    


    
    
    


    
![jpeg](output_12_1327.jpg)
    


    
    
    


    
![jpeg](output_12_1329.jpg)
    


    
    
    


    
![jpeg](output_12_1331.jpg)
    


    
    
    


    
![jpeg](output_12_1333.jpg)
    


    
    
    


    
![jpeg](output_12_1335.jpg)
    


    
    
    


    
![jpeg](output_12_1337.jpg)
    


    
    
    


    
![jpeg](output_12_1339.jpg)
    


    
    
    


    
![jpeg](output_12_1341.jpg)
    


    
    
    


    
![jpeg](output_12_1343.jpg)
    


    
    
    


    
![jpeg](output_12_1345.jpg)
    


    
    
    


    
![jpeg](output_12_1347.jpg)
    


    
    
    


    
![jpeg](output_12_1349.jpg)
    


    
    
    


    
![jpeg](output_12_1351.jpg)
    


    
    
    


    
![jpeg](output_12_1353.jpg)
    


    
    
    


    
![jpeg](output_12_1355.jpg)
    


    
    
    


    
![jpeg](output_12_1357.jpg)
    


    
    
    


    
![jpeg](output_12_1359.jpg)
    


    
    
    


    
![jpeg](output_12_1361.jpg)
    


    
    
    


    
![jpeg](output_12_1363.jpg)
    


    
    
    


    
![jpeg](output_12_1365.jpg)
    


    
    
    


    
![jpeg](output_12_1367.jpg)
    


    
    
    


    
![jpeg](output_12_1369.jpg)
    


    
    
    


    
![jpeg](output_12_1371.jpg)
    


    
    
    


    
![jpeg](output_12_1373.jpg)
    


    
    
    


    
![jpeg](output_12_1375.jpg)
    


    
    
    


    
![jpeg](output_12_1377.jpg)
    


    
    
    


    
![jpeg](output_12_1379.jpg)
    


    
    
    


    
![jpeg](output_12_1381.jpg)
    


    
    
    


    
![jpeg](output_12_1383.jpg)
    


    
    
    


    
![jpeg](output_12_1385.jpg)
    


    
    
    


    
![jpeg](output_12_1387.jpg)
    


    
    
    


    
![jpeg](output_12_1389.jpg)
    


    
    
    


    
![jpeg](output_12_1391.jpg)
    


    
    
    


    
![jpeg](output_12_1393.jpg)
    


    
    
    


    
![jpeg](output_12_1395.jpg)
    


    
    
    


    
![jpeg](output_12_1397.jpg)
    


    
    
    


    
![jpeg](output_12_1399.jpg)
    


    
    
    


    
![jpeg](output_12_1401.jpg)
    


    
    
    


    
![jpeg](output_12_1403.jpg)
    


    
    
    


    
![jpeg](output_12_1405.jpg)
    


    
    
    


    
![jpeg](output_12_1407.jpg)
    


    
    
    

# mAP


```python
import pandas as pd
import matplotlib.pyplot as plt
```


```python
df = pd.read_excel("/content/results.xlsx")
df
```





  <div id="df-172c4720-2f81-47fa-8b7c-201bfb608368" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>train/box_loss</th>
      <th>train/obj_loss</th>
      <th>train/cls_loss</th>
      <th>metrics/precision</th>
      <th>metrics/recall</th>
      <th>metrics/mAP_0.5</th>
      <th>metrics/mAP_0.5:0.95</th>
      <th>val/box_loss</th>
      <th>val/obj_loss</th>
      <th>val/cls_loss</th>
      <th>x/lr0</th>
      <th>x/lr1</th>
      <th>x/lr2</th>
      <th>Unnamed: 14</th>
      <th>Unnamed: 15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.094882</td>
      <td>0.046213</td>
      <td>0.020608</td>
      <td>0.58632</td>
      <td>0.013033</td>
      <td>0.009321</td>
      <td>0.00262</td>
      <td>0.082655</td>
      <td>0.033144</td>
      <td>0.018284</td>
      <td>0.070065</td>
      <td>0.003326</td>
      <td>0.003326</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.074194</td>
      <td>0.039991</td>
      <td>0.017092</td>
      <td>0.51875</td>
      <td>0.505210</td>
      <td>0.456460</td>
      <td>0.17980</td>
      <td>0.053927</td>
      <td>0.024527</td>
      <td>0.007748</td>
      <td>0.039933</td>
      <td>0.006528</td>
      <td>0.006528</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.059805</td>
      <td>0.034794</td>
      <td>0.009479</td>
      <td>0.80979</td>
      <td>0.701710</td>
      <td>0.773670</td>
      <td>0.35002</td>
      <td>0.045890</td>
      <td>0.018784</td>
      <td>0.004022</td>
      <td>0.009669</td>
      <td>0.009597</td>
      <td>0.009597</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.050337</td>
      <td>0.031468</td>
      <td>0.006934</td>
      <td>0.89460</td>
      <td>0.789780</td>
      <td>0.872800</td>
      <td>0.44025</td>
      <td>0.039826</td>
      <td>0.016241</td>
      <td>0.002734</td>
      <td>0.009406</td>
      <td>0.009406</td>
      <td>0.009406</td>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.045928</td>
      <td>0.029317</td>
      <td>0.005949</td>
      <td>0.89672</td>
      <td>0.830580</td>
      <td>0.901670</td>
      <td>0.49157</td>
      <td>0.035576</td>
      <td>0.015616</td>
      <td>0.002288</td>
      <td>0.009406</td>
      <td>0.009406</td>
      <td>0.009406</td>
      <td>NaN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.043631</td>
      <td>0.028093</td>
      <td>0.005220</td>
      <td>0.90517</td>
      <td>0.830950</td>
      <td>0.904940</td>
      <td>0.49962</td>
      <td>0.036438</td>
      <td>0.014911</td>
      <td>0.002534</td>
      <td>0.009208</td>
      <td>0.009208</td>
      <td>0.009208</td>
      <td>NaN</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.041930</td>
      <td>0.027479</td>
      <td>0.004662</td>
      <td>0.91349</td>
      <td>0.845850</td>
      <td>0.916070</td>
      <td>0.52202</td>
      <td>0.034640</td>
      <td>0.014604</td>
      <td>0.002068</td>
      <td>0.009010</td>
      <td>0.009010</td>
      <td>0.009010</td>
      <td>NaN</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.040753</td>
      <td>0.026828</td>
      <td>0.004321</td>
      <td>0.90972</td>
      <td>0.855380</td>
      <td>0.922290</td>
      <td>0.48847</td>
      <td>0.036713</td>
      <td>0.014360</td>
      <td>0.001948</td>
      <td>0.008812</td>
      <td>0.008812</td>
      <td>0.008812</td>
      <td>NaN</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.039758</td>
      <td>0.025897</td>
      <td>0.004107</td>
      <td>0.92836</td>
      <td>0.859010</td>
      <td>0.926860</td>
      <td>0.52239</td>
      <td>0.033641</td>
      <td>0.014014</td>
      <td>0.001736</td>
      <td>0.008614</td>
      <td>0.008614</td>
      <td>0.008614</td>
      <td>NaN</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7</td>
      <td>0.040759</td>
      <td>0.026609</td>
      <td>0.004366</td>
      <td>0.91488</td>
      <td>0.859510</td>
      <td>0.924980</td>
      <td>0.52716</td>
      <td>0.033532</td>
      <td>0.015213</td>
      <td>0.002075</td>
      <td>0.008812</td>
      <td>0.008812</td>
      <td>0.008812</td>
      <td>NaN</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8</td>
      <td>0.039057</td>
      <td>0.025794</td>
      <td>0.004082</td>
      <td>0.91774</td>
      <td>0.867460</td>
      <td>0.929480</td>
      <td>0.52760</td>
      <td>0.034511</td>
      <td>0.014057</td>
      <td>0.002034</td>
      <td>0.006850</td>
      <td>0.006850</td>
      <td>0.006850</td>
      <td>NaN</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9</td>
      <td>0.038278</td>
      <td>0.025539</td>
      <td>0.003729</td>
      <td>0.92733</td>
      <td>0.874270</td>
      <td>0.935070</td>
      <td>0.53508</td>
      <td>0.034527</td>
      <td>0.013905</td>
      <td>0.001541</td>
      <td>0.006400</td>
      <td>0.006400</td>
      <td>0.006400</td>
      <td>NaN</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10</td>
      <td>0.037662</td>
      <td>0.025123</td>
      <td>0.003586</td>
      <td>0.92976</td>
      <td>0.876100</td>
      <td>0.935660</td>
      <td>0.54333</td>
      <td>0.032905</td>
      <td>0.013436</td>
      <td>0.001555</td>
      <td>0.005950</td>
      <td>0.005950</td>
      <td>0.005950</td>
      <td>NaN</td>
      <td>13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>11</td>
      <td>0.036985</td>
      <td>0.024625</td>
      <td>0.003394</td>
      <td>0.92959</td>
      <td>0.883610</td>
      <td>0.936860</td>
      <td>0.56382</td>
      <td>0.031406</td>
      <td>0.013398</td>
      <td>0.001660</td>
      <td>0.005500</td>
      <td>0.005500</td>
      <td>0.005500</td>
      <td>NaN</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12</td>
      <td>0.036241</td>
      <td>0.024194</td>
      <td>0.003178</td>
      <td>0.93777</td>
      <td>0.884250</td>
      <td>0.940340</td>
      <td>0.57258</td>
      <td>0.030774</td>
      <td>0.013152</td>
      <td>0.001461</td>
      <td>0.005050</td>
      <td>0.005050</td>
      <td>0.005050</td>
      <td>NaN</td>
      <td>15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>13</td>
      <td>0.035658</td>
      <td>0.024110</td>
      <td>0.002874</td>
      <td>0.93499</td>
      <td>0.893090</td>
      <td>0.940440</td>
      <td>0.58681</td>
      <td>0.029535</td>
      <td>0.013002</td>
      <td>0.001573</td>
      <td>0.004600</td>
      <td>0.004600</td>
      <td>0.004600</td>
      <td>NaN</td>
      <td>16</td>
    </tr>
    <tr>
      <th>16</th>
      <td>14</td>
      <td>0.035617</td>
      <td>0.024145</td>
      <td>0.003064</td>
      <td>0.93556</td>
      <td>0.896270</td>
      <td>0.944870</td>
      <td>0.56168</td>
      <td>0.031351</td>
      <td>0.013006</td>
      <td>0.001516</td>
      <td>0.004150</td>
      <td>0.004150</td>
      <td>0.004150</td>
      <td>NaN</td>
      <td>17</td>
    </tr>
    <tr>
      <th>17</th>
      <td>15</td>
      <td>0.035224</td>
      <td>0.023553</td>
      <td>0.003013</td>
      <td>0.93413</td>
      <td>0.899250</td>
      <td>0.943930</td>
      <td>0.58223</td>
      <td>0.029982</td>
      <td>0.013013</td>
      <td>0.001430</td>
      <td>0.003700</td>
      <td>0.003700</td>
      <td>0.003700</td>
      <td>NaN</td>
      <td>18</td>
    </tr>
    <tr>
      <th>18</th>
      <td>16</td>
      <td>0.034864</td>
      <td>0.023473</td>
      <td>0.003034</td>
      <td>0.93592</td>
      <td>0.898060</td>
      <td>0.946220</td>
      <td>0.56042</td>
      <td>0.032169</td>
      <td>0.013198</td>
      <td>0.001362</td>
      <td>0.003250</td>
      <td>0.003250</td>
      <td>0.003250</td>
      <td>NaN</td>
      <td>19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>17</td>
      <td>0.034379</td>
      <td>0.023420</td>
      <td>0.002871</td>
      <td>0.93859</td>
      <td>0.897730</td>
      <td>0.946920</td>
      <td>0.59638</td>
      <td>0.028990</td>
      <td>0.012649</td>
      <td>0.001435</td>
      <td>0.002800</td>
      <td>0.002800</td>
      <td>0.002800</td>
      <td>NaN</td>
      <td>20</td>
    </tr>
    <tr>
      <th>20</th>
      <td>18</td>
      <td>0.034032</td>
      <td>0.023068</td>
      <td>0.002710</td>
      <td>0.93765</td>
      <td>0.899570</td>
      <td>0.948300</td>
      <td>0.60238</td>
      <td>0.028767</td>
      <td>0.012639</td>
      <td>0.001362</td>
      <td>0.002350</td>
      <td>0.002350</td>
      <td>0.002350</td>
      <td>NaN</td>
      <td>21</td>
    </tr>
    <tr>
      <th>21</th>
      <td>19</td>
      <td>0.033887</td>
      <td>0.022975</td>
      <td>0.002660</td>
      <td>0.93356</td>
      <td>0.902970</td>
      <td>0.949290</td>
      <td>0.60640</td>
      <td>0.028568</td>
      <td>0.012647</td>
      <td>0.001290</td>
      <td>0.001900</td>
      <td>0.001900</td>
      <td>0.001900</td>
      <td>NaN</td>
      <td>22</td>
    </tr>
    <tr>
      <th>22</th>
      <td>20</td>
      <td>0.033594</td>
      <td>0.022966</td>
      <td>0.002620</td>
      <td>0.94049</td>
      <td>0.905640</td>
      <td>0.948920</td>
      <td>0.59937</td>
      <td>0.029066</td>
      <td>0.012649</td>
      <td>0.001308</td>
      <td>0.001450</td>
      <td>0.001450</td>
      <td>0.001450</td>
      <td>NaN</td>
      <td>23</td>
    </tr>
    <tr>
      <th>23</th>
      <td>21</td>
      <td>0.033165</td>
      <td>0.022642</td>
      <td>0.002439</td>
      <td>0.94288</td>
      <td>0.905100</td>
      <td>0.951340</td>
      <td>0.61209</td>
      <td>0.028086</td>
      <td>0.012520</td>
      <td>0.001285</td>
      <td>0.001000</td>
      <td>0.001000</td>
      <td>0.001000</td>
      <td>NaN</td>
      <td>24</td>
    </tr>
    <tr>
      <th>24</th>
      <td>20</td>
      <td>0.033086</td>
      <td>0.022430</td>
      <td>0.002313</td>
      <td>0.93837</td>
      <td>0.907590</td>
      <td>0.951230</td>
      <td>0.60905</td>
      <td>0.028347</td>
      <td>0.012557</td>
      <td>0.001317</td>
      <td>0.001450</td>
      <td>0.001450</td>
      <td>0.001450</td>
      <td>NaN</td>
      <td>25</td>
    </tr>
    <tr>
      <th>25</th>
      <td>21</td>
      <td>0.033614</td>
      <td>0.022679</td>
      <td>0.002484</td>
      <td>0.93500</td>
      <td>0.906610</td>
      <td>0.948130</td>
      <td>0.59368</td>
      <td>0.029155</td>
      <td>0.012656</td>
      <td>0.001410</td>
      <td>0.003172</td>
      <td>0.003172</td>
      <td>0.003172</td>
      <td>NaN</td>
      <td>26</td>
    </tr>
    <tr>
      <th>26</th>
      <td>22</td>
      <td>0.033369</td>
      <td>0.022731</td>
      <td>0.002388</td>
      <td>0.94146</td>
      <td>0.900530</td>
      <td>0.951060</td>
      <td>0.60389</td>
      <td>0.028738</td>
      <td>0.012505</td>
      <td>0.001221</td>
      <td>0.002831</td>
      <td>0.002831</td>
      <td>0.002831</td>
      <td>NaN</td>
      <td>27</td>
    </tr>
    <tr>
      <th>27</th>
      <td>23</td>
      <td>0.033188</td>
      <td>0.022554</td>
      <td>0.002374</td>
      <td>0.94090</td>
      <td>0.899880</td>
      <td>0.949970</td>
      <td>0.60153</td>
      <td>0.028679</td>
      <td>0.012479</td>
      <td>0.001267</td>
      <td>0.002490</td>
      <td>0.002490</td>
      <td>0.002490</td>
      <td>NaN</td>
      <td>28</td>
    </tr>
    <tr>
      <th>28</th>
      <td>24</td>
      <td>0.032750</td>
      <td>0.022157</td>
      <td>0.002269</td>
      <td>0.94038</td>
      <td>0.905960</td>
      <td>0.949860</td>
      <td>0.61084</td>
      <td>0.028310</td>
      <td>0.012436</td>
      <td>0.001338</td>
      <td>0.002148</td>
      <td>0.002148</td>
      <td>0.002148</td>
      <td>NaN</td>
      <td>29</td>
    </tr>
    <tr>
      <th>29</th>
      <td>25</td>
      <td>0.032430</td>
      <td>0.021924</td>
      <td>0.002111</td>
      <td>0.94837</td>
      <td>0.903280</td>
      <td>0.951750</td>
      <td>0.60919</td>
      <td>0.028329</td>
      <td>0.012392</td>
      <td>0.001183</td>
      <td>0.001807</td>
      <td>0.001807</td>
      <td>0.001807</td>
      <td>NaN</td>
      <td>30</td>
    </tr>
    <tr>
      <th>30</th>
      <td>26</td>
      <td>0.032073</td>
      <td>0.021941</td>
      <td>0.001944</td>
      <td>0.94553</td>
      <td>0.904890</td>
      <td>0.950500</td>
      <td>0.61178</td>
      <td>0.028141</td>
      <td>0.012432</td>
      <td>0.001301</td>
      <td>0.001466</td>
      <td>0.001466</td>
      <td>0.001466</td>
      <td>NaN</td>
      <td>31</td>
    </tr>
    <tr>
      <th>31</th>
      <td>27</td>
      <td>0.031944</td>
      <td>0.021936</td>
      <td>0.001943</td>
      <td>0.93905</td>
      <td>0.913020</td>
      <td>0.951910</td>
      <td>0.61193</td>
      <td>0.028135</td>
      <td>0.012380</td>
      <td>0.001285</td>
      <td>0.001124</td>
      <td>0.001124</td>
      <td>0.001124</td>
      <td>NaN</td>
      <td>32</td>
    </tr>
    <tr>
      <th>32</th>
      <td>28</td>
      <td>0.031732</td>
      <td>0.021445</td>
      <td>0.001881</td>
      <td>0.94233</td>
      <td>0.908290</td>
      <td>0.950710</td>
      <td>0.61550</td>
      <td>0.027872</td>
      <td>0.012350</td>
      <td>0.001304</td>
      <td>0.000783</td>
      <td>0.000783</td>
      <td>0.000783</td>
      <td>NaN</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-172c4720-2f81-47fa-8b7c-201bfb608368')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-172c4720-2f81-47fa-8b7c-201bfb608368 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-172c4720-2f81-47fa-8b7c-201bfb608368');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-42354a08-9225-4664-adff-cf696c85936e">
  <button class="colab-df-quickchart" onclick="quickchart('df-42354a08-9225-4664-adff-cf696c85936e')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-42354a08-9225-4664-adff-cf696c85936e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
plt.figure(figsize=(12,5))
plt.plot(range(1, 34), df.iloc[:,6], color = 'green', marker = 'o', linestyle = 'solid')
plt.xticks(list(range(1, 34)))
plt.yticks([0.0, 0.4, 0.6, 0.8, 0.9, 0.95])
plt.xlabel("epochs")
plt.ylabel("mAP")
plt.show()
```


    
![png](output_16_0.png)
    



```python
plt.figure(figsize=(10,5))
plt.plot(range(5, 34), df.iloc[4:,6], color = 'green', marker = 'o', linestyle = 'solid')
plt.xticks(list(range(5, 34)))
plt.xlabel("epochs")
plt.ylabel("mAP")
plt.show()
```


    
![png](output_17_0.png)
    



```python
df[df.iloc[:, 6] == max(df.iloc[:, 6])].iloc[:, 6]
```




    31    0.95191
    Name:      metrics/mAP_0.5, dtype: float64




```python
plt.figure(figsize=(10,5))
plt.plot(range(5, 34), df.iloc[4:,6], color = 'green', marker = 'o', linestyle = 'solid')
plt.axhline(max(df.iloc[:, 6]), color = 'red', linestyle = '--', alpha = 0.6)
plt.axvline(32, linestyle = '--', alpha = 0.6)
plt.xticks(list(range(5, 34)))
plt.xlabel("epochs")
plt.ylabel("mAP")
plt.show()
```


    
![png](output_19_0.png)
    


# 인터넷 이미지 예측


```python
!python detect.py --weights /content/drive/MyDrive/yolov5s_results2/weights/best.pt --img 416 --conf 0.5 --source /content/11.jpg
```

    [34m[1mdetect: [0mweights=['/content/drive/MyDrive/yolov5s_results2/weights/best.pt'], source=/content/11.jpg, data=data/coco128.yaml, imgsz=[416, 416], conf_thres=0.5, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    image 1/1 /content/11.jpg: 288x416 5 helmets, Done. (0.050s)
    Speed: 0.6ms pre-process, 49.5ms inference, 1.3ms NMS per image at shape (1, 3, 416, 416)
    Results saved to [1mruns/detect/exp5[0m
    


```python
from IPython.display import Image, display

display(Image(filename='/content/yolov5/runs/detect/exp5/11.jpg'))
```


    
![jpeg](output_22_0.jpg)
    



```python
!python detect.py --weights /content/drive/MyDrive/yolov5s_results2/weights/best.pt --img 416 --conf 0.5 --source /content/22.jpg
```

    [34m[1mdetect: [0mweights=['/content/drive/MyDrive/yolov5s_results2/weights/best.pt'], source=/content/22.jpg, data=data/coco128.yaml, imgsz=[416, 416], conf_thres=0.5, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    image 1/1 /content/22.jpg: 320x416 3 helmets, Done. (0.055s)
    Speed: 0.7ms pre-process, 54.6ms inference, 2.4ms NMS per image at shape (1, 3, 416, 416)
    Results saved to [1mruns/detect/exp4[0m
    


```python
from IPython.display import Image, display

display(Image(filename='/content/yolov5/runs/detect/exp4/22.jpg'))
```


    
![jpeg](output_24_0.jpg)
    



```python
!python detect.py --weights /content/drive/MyDrive/yolov5s_results2/weights/best.pt --img 416 --conf 0.5 --source /content/88.jpg
```

    [34m[1mdetect: [0mweights=['/content/drive/MyDrive/yolov5s_results2/weights/best.pt'], source=/content/88.jpg, data=data/coco128.yaml, imgsz=[416, 416], conf_thres=0.5, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    image 1/1 /content/88.jpg: 320x416 13 heads, Done. (0.163s)
    Speed: 2.6ms pre-process, 162.7ms inference, 1.3ms NMS per image at shape (1, 3, 416, 416)
    Results saved to [1mruns/detect/exp6[0m
    


```python
from IPython.display import Image, display

display(Image(filename='/content/yolov5/runs/detect/exp6/88.jpg'))
```


    
![jpeg](output_26_0.jpg)
    



```python
!python detect.py --weights /content/drive/MyDrive/yolov5s_results2/weights/best.pt --img 416 --conf 0.5 --source /content/33.jpg
```

    [34m[1mdetect: [0mweights=['/content/drive/MyDrive/yolov5s_results2/weights/best.pt'], source=/content/33.jpg, data=data/coco128.yaml, imgsz=[416, 416], conf_thres=0.5, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    image 1/1 /content/33.jpg: 320x416 2 heads, 1 helmet, Done. (0.054s)
    Speed: 0.5ms pre-process, 53.7ms inference, 1.5ms NMS per image at shape (1, 3, 416, 416)
    Results saved to [1mruns/detect/exp3[0m
    


```python
from IPython.display import Image, display

display(Image(filename='/content/yolov5/runs/detect/exp3/33.jpg'))
```


    
![jpeg](output_28_0.jpg)
    



```python
!python detect.py --weights /content/drive/MyDrive/yolov5s_results2/weights/best.pt --img 416 --conf 0.5 --source /content/77.jpg
```

    [34m[1mdetect: [0mweights=['/content/drive/MyDrive/yolov5s_results2/weights/best.pt'], source=/content/77.jpg, data=data/coco128.yaml, imgsz=[416, 416], conf_thres=0.5, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    image 1/1 /content/77.jpg: 320x416 2 heads, 6 helmets, Done. (0.158s)
    Speed: 3.0ms pre-process, 157.7ms inference, 1.3ms NMS per image at shape (1, 3, 416, 416)
    Results saved to [1mruns/detect/exp5[0m
    


```python
from IPython.display import Image, display

display(Image(filename='/content/yolov5/runs/detect/exp5/77.jpg'))
```


    
![jpeg](output_30_0.jpg)
    



```python
!python detect.py --weights /content/drive/MyDrive/yolov5/yolov5s_results2/weights/best.pt --img 416 --conf 0.5 --source /content/55.jpg
```

    [34m[1mdetect: [0mweights=['/content/drive/MyDrive/yolov5/yolov5s_results2/weights/best.pt'], source=/content/55.jpg, data=data/coco128.yaml, imgsz=[416, 416], conf_thres=0.5, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
    YOLOv5 🚀 v6.1-306-gfbe67e4 Python-3.10.12 torch-2.1.0+cu118 CPU
    
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    image 1/1 /content/55.jpg: 416x320 2 heads, 7 helmets, Done. (0.161s)
    Speed: 3.6ms pre-process, 161.3ms inference, 32.4ms NMS per image at shape (1, 3, 416, 416)
    Results saved to [1mruns/detect/exp[0m
    


```python
display(Image(filename='/content/yolov5/runs/detect/exp/55.jpg'))
```


    
![jpeg](output_32_0.jpg)
    



```python

```
