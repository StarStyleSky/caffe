import caffe
import numpy as np
import scipy.io as sio
iter = 44
sol = "/data2/wujial/A-FCN/caffe/models/kernel_attention_module_2/solver.prototxt"
solver = caffe.SGDSolver(sol)
param_keys = solver.net.params.keys()
param_mat = dict()
w_mat = np.zeros((365,len(param_keys),4))
b_mat = np.zeros((365,len(param_keys),4))
for it in xrange(1,365):
  iter = it
  model = '/data2/wujial/A-FCN/caffe/models/kernel_attention_module_2/kernel_attention_iter_'+ str(iter)+ '.caffemodel'
  solver.net.copy_from(model)
  f = open('param_status_' + str(iter)+ '.txt','a+')
  for i in xrange(len(param_keys)):
    param = solver.net.params[param_keys[i]]
    p_len = len(param)
    for j in xrange(p_len):
      pa = param[j].data
      mini = pa.min()
      maxi = pa.max()
      mean = pa.mean()
      var = pa.var()
      if j == 0 :
         #print>>f,'%s params status w : %f, %f, %f, %f'%(param_keys[i],mini,maxi,mean,var)
         w_mat[it,i,0] = mini
         w_mat[it,i,1] = maxi
         w_mat[it,i,2] = mean
         w_mat[it,i,3] = var
      else:
         #print>>f,'%s params status b : %f, %f, %f, %f'%(param_keys[i],mini,maxi,mean,var)
         b_mat[it,i,0] = mini
         b_mat[it,i,1] = maxi
         b_mat[it,i,2] = mean
         b_mat[it,i,3] = var
  f.close()
param_mat['w_mat'] = w_mat
param_mat['b_mat'] = b_mat
saveto = 'params.mat'
sio.savemat(saveto,param_mat)