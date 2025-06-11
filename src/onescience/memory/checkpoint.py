from contextlib import contextmanager
from torch.utils.checkpoint import checkpoint
from functools import partial
import torch

class part(torch.nn.Module):
    '''
    2025 01 09 wangxian
    这个类主要是为了避免pytorch的类别检查，为了兼容原生pytorch而设计
    '''
    def __init__(self,ch,ori):
        super().__init__()
        self.ch = ch
        self.ori = ori
    def __call__(self,*args,**kwags):

        f = partial(self.ch,self.ori,use_reentrant=True)
        return f(*args,**kwags)

@contextmanager
def replace_function(module, replace_layers_list, ddp_flag = False):
    '''
    2025 01 09 wangxian
    这个函数可以使得任意pytorch模型中的模块被替换为checkpoint函数，从而实现checkpoint的功能
    以下给出一个案例，替代模型训练中的forward过程
    
    example:
    with replace_function(my_model, ['layer1','layer2','layer3','layer4'],dist.world_size > 1):
        outpred_surface, outpred_upper_air = my_model(invar)
    
    
    模型前向被替换为上下文的包裹函数
    其中replace_layers_list是需要被替换的nn.module子类，注册在模型中，ddp_flag代表是否使用分布式训练，默认是false
    '''
    original_function_list = []
    #part_ins = part()
    for replace_layer in replace_layers_list:
        original_function = getattr(module, replace_layer) if ddp_flag == False else getattr(module.module, replace_layer)
        original_function_list.append(original_function)
        new_func = part(checkpoint,original_function)
        setattr(module, replace_layer, new_func) if ddp_flag == False else setattr(module.module, replace_layer, new_func)
    try:
        yield
    finally:
        for i,replace_layer in enumerate(replace_layers_list):
            setattr(module, replace_layer, original_function_list[i]) if ddp_flag == False else setattr(module.module, replace_layer, original_function_list[i])