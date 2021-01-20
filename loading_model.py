model = torch.load("model")
model.eval()

import torch
torch.cuda.empty_cache()
dic={}

with open("vocab.nl",'r') as fp: 
  lines = fp.read().splitlines()
k=0
for line in lines:
  dic[k]=line
  k=k+1

f=open("predicted_output_f",'a')

for i in range(20000):
  s=src_test[i:i+1]
  t=trg_test[i:i+1]
  out=model(s,t[:, :-1])
  li=out.shape
  for i in range(li[0]):
    for j in range(li[1]):
      d=int(torch.argmax(out[i][j]))
      if(str(dic[d])!= "</S>" and str(dic[d])!= "uno" and str(dic[d])!= "<S>"):
        f.write(str(dic[d])+" ")
    f.write("\n")
