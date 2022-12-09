import torch
import glob
import numpy as np

files = glob.glob(r"D:\pythonwork\Servercrash_detection\error\*")
#np.savetxt('my_file.txt', torch.Tensor([3,4,5,6]).numpy())

    #np.savetxt("{}.pt".format(path.split("\\")[-1].split(".")[0]),torch.load(path).cpu().numpy())

state = torch.load(files[2]).cpu().numpy()
# np.savetxt('state.txt', torch.load(files[2]).cpu().numpy())    
print(np.zeros(50).shape)