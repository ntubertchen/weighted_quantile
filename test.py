import random
from quantile_v2 import c_quantile
from weighted_quantile import quantile
from  itertools import permutations
import pickle
import numpy as np
def add_sample(a,test_sample,out=None,overwrite_input=False,keepdims=False):
    w = np.random.random_sample((a.shape))
    interpolation_list = ['lower','higher','midpoint','nearest','linear']
    # q_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,np.random.rand(10)]
    q_list = list(np.random.rand(30))
    _q = []
    for i in range(3):
        _q.append(random.choices(q_list, k=1))
    for i in range(3):
        _q.append(random.choices(q_list, k=1))
    axis_list = range(a.ndim)
    _axis = []
    for i in range(3):
        _axis.append(tuple(random.sample(axis_list, k=1)))
    for i in range(3):
        _axis.append(random.sample(axis_list, k=1)[0])
    # print (_axis, axis_list)
    for interpolation in interpolation_list:
        for q in _q:
            for axis in _axis:
                d ={'a':a,
                    'q':q,
                    'w':w,
                    'axis':axis,
                    'out':out,
                    'overwrite_input':overwrite_input,
                    'interpolation':interpolation,
                    'keepdims':keepdims}
                test_sample.append(d)

def check_equal(param_list,error_samples):
    f = True
    for param_dict in param_list:
        a = param_dict['a']
        q = param_dict['q']
        w = param_dict['w']
        axis = param_dict['axis']
        out = param_dict['out']
        overwrite_input = param_dict['overwrite_input']
        interpolation = param_dict['interpolation']
        keepdims = param_dict['keepdims']

        result_a = c_quantile(a, q, w, axis=axis, out=out, overwrite_input=overwrite_input, interpolation=interpolation, keepdims=keepdims)
        result_b = quantile(a, q, axis=axis, out=out, overwrite_input=overwrite_input, interpolation=interpolation, keepdims=keepdims)

        if not np.allclose(result_a,result_b,equal_nan=True):
            print (a,q,w, axis, interpolation)
            error_samples.append(param_dict)
            print("Error occurs!")
            print("result_a",result_a)
            print("result_b",result_b)
            print (a, q, w)
            f = False
            break
    if f:
        print("Pass!")


if __name__=="__main__":
    test_sample = []
    add_sample(np.random.rand(3,3), test_sample)
    error_samples = []
    check_equal(test_sample,error_samples)
