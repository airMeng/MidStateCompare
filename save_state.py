import contextlib
import torch
import os
from yaml import Loader
from compare_dict import compare_state
import logging
logging.basicConfig(level=logging.DEBUG)

def args_parser(path):
    with open(path) as f:
        setting=Loader(f)
    args=setting.get_data()
    return args


@contextlib.contextmanager
def save_state(model,epoch,iter):
    yield
    args=args_parser('setting.yaml')
    online_args=args['online_comparison']
    compare_epoch=online_args['epoch']
    compare_iter=online_args['iteration']
    compare_keys=online_args['keys']
    save_path=online_args['save_path']
    refer_gpu_path=args['refer_gpu_path']
    #refer_gpu_dict=torch.load(os.path.join(refer_gpu_path,str(epoch),str(iter),'state_dict.checkpoint'))
    path=os.path.join(save_path,str(epoch),str(iter))
    if not os.path.exists(path):
        os.makedirs(path)
    """
    if epoch==compare_epoch or compare_epoch=='all':
        if iter==compare_iter or compare_iter=='all':
            compare_result=compare_state(model.state_dict(),refer_gpu_dict,dic_keys=compare_keys)
            if compare_result:
                logging.info('epoch = {} iter = {} state_dict error is acceptable'.format(epoch, iter))
            else:
                logging.warning('epoch = {} iter = {} state_dict error is unacceptable'.format(epoch, iter))
    """
    if online_args['save_all_dict']:
        file_name=os.path.join(path,'cpu.checkpoint')
        torch.save(model.state_dict(), file_name)