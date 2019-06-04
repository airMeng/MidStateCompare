import collections
import logging
import torch



def compare(dicts_cpu,dicts_gpu):
    epoch_cpu=len(dicts_cpu)
    epoch_gpu=len(dicts_gpu)
    if epoch_cpu<epoch_gpu:
        logging.info('the epoch of cpu training is {} less than gpu epoch {}'.format(epoch_cpu,epoch_gpu))
        logging.info('compare all epochs in cpu training')
    elif epoch_cpu>epoch_gpu:
        logging.info('the epoch of cpu training is {} more than gpu epoch {}'.format(epoch_cpu, epoch_gpu))
        logging.info('compare first {} epochs in cpu training'.format(epoch_gpu))

    epoch_compare=min(epoch_cpu,epoch_gpu)
    for i in range(epoch_compare):
        compare_state(dicts_cpu[i],dicts_gpu[i])


def compare_state(dicts_cpu,dicts_gpu,error_threshold=1e-2,dic_keys='all'):
    if dic_keys == 'all':
        if dicts_gpu.keys()==dicts_cpu.keys():
            dic_keys=list(dicts_gpu.keys())
        else:
            dic_keys = list(set(list(dicts_gpu.keys)).intersection(set(list(dicts_cpu.keys))))
            keys_cpu = list(set(list(dicts_cpu.keys)).difference(set(list(dicts_gpu.keys))))
            keys_gpu = list(set(list(dicts_gpu.keys)).difference(set(list(dicts_cpu.keys))))
            logging.warning('unknown cpu training state keys {}'.format(keys_cpu))
            logging.warning('missing gpu training state keys {}'.format(keys_gpu))
    else:
        dic_keys = [dic_keys]

    err_msg=[]
    for key in dic_keys:
        try:
            tensor_cpu=dicts_cpu[key]
            tensor_gpu=dicts_gpu[key]
        except KeyError:
            logging.error('key {} is not in state_dict.keys()'.format(key))
            raise
        if tensor_cpu.shape != tensor_gpu.shape:
            logging.error('mismatched shape for {} state dict in cpu and gpu \n\
cpu tensor shape is {} while gpu tensor is {}'.format(key,tensor_cpu.shape,tensor_gpu.shape))
            err_msg.append('mismatched shape for {} state dict in cpu and gpu \n\
cpu tensor shape is {} while gpu tensor is {}'.format(key,tensor_cpu.shape,tensor_gpu.shape))
            continue
        else:
            error=torch.norm(tensor_cpu.float()-tensor_gpu.cpu().float())/tensor_cpu.numel()
            if error>=error_threshold:
                logging.error('state tensor error for {} between cpu and gpu is larger than threshold'.format(key))
                err_msg.append('state tensor error for {} between cpu and gpu is larger than threshold'.format(key))
    if len(err_msg)==0:
        return True
    else:
        return False


def main():
    dicts_cpu=torch.load('123.checkpoint')
    dicts_gpu = torch.load('0/0/state_dict.checkpoint')
    print(compare_state(dicts_cpu, dicts_gpu))


if __name__ == '__main__':
    main()





