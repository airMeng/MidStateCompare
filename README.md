# MidStateCompare
This is for comparing mid state dict between gpu traing and cpu training.  
setting is in setting.yaml:  
```yaml
model_name: resnext101_32x8d     # model name, also the dir where the state dict is saved.  
refer_gpu_path: 'gpu'            # path to gpu refer state
with_mkldnn: false               # TBD  
save_gpu_refer: false            # if true, save the gpu training state as reference. if false, start comparison  
epoch: all                       # in which epoch conduct the comparison. all or int(0,etc)  
iteration: all                   # in which iteration conduct the comparison. all or int(0,etc)  
keys: all                        # which key to compare m1.linear2.bias,etc  
save_all_dict: true              # whether to save cpu state dict  
save_path: 'cpu'                 # path to save cpu state dict    
```

First time to use it, save gpu state dict as reference. remember to set **save_gpu_refer: true**
```python
from compare_wrapper import compare_state
model.cuda()
y,x=y.cuda(),x.cuda()
...
# if epoch can't be passed, the epoch default is 0, which means we only compare epoch 0.
with compare_state(model,iter,epoch=):
  y=model(x)
```

to compare cpu state.set **save_gpu_refer: false**
```python
from compare_wrapper import compare_state
model.cpu()
y,x=y.cpu(),x.cpu()
with compare_state(model,iter,epoch=):
  y=model(x)
```
remember cpu() and cuda() switch.

the output will be like:
```python
INFO:root:epoch = 0 iter = 5 state_dict error is acceptable
ERROR:root:state tensor error for m2.linear2.bias between cpu and gpu is larger than threshold
WARNING:root:epoch = 0 iter = 6 state_dict error is unacceptable
```
