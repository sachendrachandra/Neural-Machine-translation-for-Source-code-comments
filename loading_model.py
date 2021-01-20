import torch
import sys
model = torch.load(sys.argv[2])
model.eval()

torch.cuda.empty_cache()
dic={}

with open(sys.argv[3],'r') as fp: 
  lines = fp.read().splitlines()
k=0
for line in lines:
  dic[k]=line
  k=k+1

f=open(sys.argv[4],'a')

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
