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
def save_state(model,iter,**kwargs):
    args = args_parser('setting.yaml')
    save_gpu_refer=args['save_gpu_refer']
    if 'epoch' not in kwargs.keys():
        epoch=0
    else:
        epoch=kwargs['epoch']
    model_name=args['model_name']
    compare_epoch = args['epoch']
    compare_iter = args['iteration']
    compare_keys = args['keys']
    save_path = os.path.join(model_name,args['save_path'])
    refer_gpu_path = os.path.join(model_name,args['refer_gpu_path'], str(epoch),str(iter))
    if not os.path.exists(refer_gpu_path):
        os.makedirs(refer_gpu_path)
    yield
    if save_gpu_refer:
        file_name = os.path.join(refer_gpu_path,'gpu.checkpoint')
        torch.save(model.state_dict(), file_name)
    elif iter <=3 and epoch==0:
        model.load_state_dict(torch.load(os.path.join(refer_gpu_path, 'gpu.checkpoint')))
    else:
        refer_gpu_dict=torch.load(os.path.join(refer_gpu_path,'gpu.checkpoint'))
        path=os.path.join(save_path,str(epoch),str(iter))
        if not os.path.exists(path):
            os.makedirs(path)
        if epoch==compare_epoch or compare_epoch=='all':
            if iter==compare_iter or compare_iter=='all':

                compare_result=compare_state(model.state_dict(),refer_gpu_dict,dic_keys=compare_keys)
                if compare_result:
                    logging.info('epoch = {} iter = {} state_dict error is acceptable'.format(epoch, iter))
                else:
                    logging.warning('epoch = {} iter = {} state_dict error is unacceptable'.format(epoch, iter))

                if args['save_all_dict']:
                    file_name=os.path.join(path,'cpu.checkpoint')
                    torch.save(model.state_dict(), file_name)
