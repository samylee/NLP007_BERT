# NLP007_BERT
NLP007: bert using pytorch

## 使用说明
### 要求
> Python == 3.6.13 \
> PyTorch == 1.10.1  
### 数据集下载
[corpus.small(提取码8888)](https://pan.baidu.com/s/1sEc-rBHw_R3ZlRr-6xiI-w)
### 训练
```shell script
python train.py  
loss: 0.56
```
### 已训练好的模型下载
[epoch_100_loss_2.56.pt(提取码8888)](https://pan.baidu.com/s/14syhDAbclE2ZeCHnyM_Mpw)
### 测试
```shell script
python predict.py  
```
```
input:  
input_sentence = 'Robbie album sold on memory card\tSinger Robbie Williams\' greatest hits album is to be sold on a memory card for mobile phones.'  
sentence1_mask_id, sentence2_mask_id = 2, 5 # sold, album  

output:  
is_next: 1 	socre: 1.0  
mask1_word: sold  
mask2_word: album  
```
## 参考
https://github.com/codertimo/BERT-pytorch   
https://blog.csdn.net/samylee  
