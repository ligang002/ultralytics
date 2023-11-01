

# 默认yolov8n.yaml
python ./train.py --yaml ultralytics/cfg/models/v8/yolov8n.yaml --weight yolov8n.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640

# yolov8-act.yaml
python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8-act.yaml --weight yolov8n.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640

# yolov8-attention-SE.yaml
python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8-attention-SE.yaml --weight yolov8n.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640

# yolov8-attention-CBAM.yaml
python ./train.py --yaml ultralytics/cfg/models/v8/det_self/yolov8-attention-CBAM.yaml --weight yolov8n.pt --cfg hyp.yaml --data ultralytics/cfg/datasets/coco128.yaml --epochs 100 --imgsz 640


