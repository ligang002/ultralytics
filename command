

参数说明：
    --yaml:     模型的yaml文件
    --weight:   预训练权重
    --cfg:      超参数
    --data:     数据集格式
    --epochs:   训练次数
    --imgsz:    图片大小
    --unamp:    关闭混合精度


yolov8默认：
    # 默认yolov8n.yaml
    python ./train.py --yaml ultralytics/cfg/models/v8/yolov8s.yaml --weight yolov8s.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640

修改激活函数：
    # yolov8-act.yaml
    python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8s-act.yaml --weight yolov8s.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640


注意力机制：

    # yolov8-attention-SE.yaml
    python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8s-attention-SE.yaml --weight yolov8s.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640

    # yolov8-attention-CBAM.yaml
    python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8s-attention-CBAM.yaml --weight yolov8s.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640

    # yolov8-attention-ECA.yaml
    python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8s-attention-ECA.yaml --weight yolov8s.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640

    # yolov8-attention-CA.yaml
    python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8s-attention-CA.yaml --weight yolov8s.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640

    # yolov8-attention-A2Attention.yaml(必须关掉AMP，否则会出现Nan，应该是因为计算得到的数超过了16位，所以出现Nan)
    python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8s-attention-A2Attention.yaml --weight yolov8s.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640 --unamp

    # yolov8-attention-BAM.yaml（BAM中包含有 BN 层， 当有BN时，batch 不能为1，所以使用BAM时，需要将task.py 中的
       m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])，改为
       m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(2, ch, s, s))])）
    python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8s-attention-BAM.yaml --weight yolov8s.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640



python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8s-attention-SE.yaml --cfg ultralytics/cfg/hyp_chepai.yaml --data ultralytics/cfg/datasets/VOC_self.yaml



