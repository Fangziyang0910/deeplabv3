# 运行环境

```
pip install -r requirements.txt
```

# 运行文件

- `train.ipynb`：训练deeplabv3模型的notebook文件
- `eval_time.ipynb`：计算测试集平均预测时间
- `calc_iou_ap.ipynb`：计算测试集PA和IOU指标
- `/utils/preprocess_data.py`：数据预处理文件
- `/model/`：deeplabv3模型定义

- `/data/`：数据集
- `/pretrained_models/resnet/`：resnet预训练模型
- `/training_logs/model_1/`：训练阶段的模型参数，只放了epoch=100、200、288（最终的模型参数）的模型参数，还有训练各阶段损失曲线图
- `/training_logs/model_eval_seq/`：对测试集语义分割

