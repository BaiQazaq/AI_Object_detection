# FOR invidia GPU
# import torch
# import os

# print("tourch.cuda.is_available():", torch.cuda.is_available())
# print("tourch.cuda.device count():", torch.cuda.device_count())
# print("os.enviroment['CUDA_VISIBLE_DEVICES']:", os.environ.get('CUDA_VISIBLE_DEVICES'))

# if torch.cuda.is_available():
#     x = torch.cuda.FloatTensor(1)
#     y = torch.cuda.FloatStorage(1)
# else:
#     print("CUDA is not available. No GPU detected")



# import torch
# x = torch.rand(5, 3)
# print(x)

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")