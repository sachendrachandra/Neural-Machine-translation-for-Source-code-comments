  import torch
  from torch.utils.data import DataLoader
  
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def data_process(vocab,file,maxl):
  dic={}

  with open(vocab,'r') as fp: 
    lines = fp.read().splitlines()
  k=0
  for line in lines:
    dic[line]=k
    k=k+1
  print(dic)

  UNK_ID = 2

  a_file = open(file, "r")
  max=maxl
  list_of_lists = []
  for line in a_file:
    line = "<S> " + line + "</S>"
    stripped_line = line.strip()
    line_list = stripped_line.split()
    line_list = [dic.get(w, UNK_ID) for w in line_list]
    # print(line_list)
    if(len(line_list)<max):
      # max = len(line_list)
      for i in range(max-len(line_list)):
        line_list.append(2)
    else:
      line_list=line_list[0:max]
      line_list[max-1] = dic["</S>"]
    list_of_lists.append(line_list)
  
  return list_of_lists
  
  
if __name__=='__main__':

  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  src_train = torch.tensor(data_process(sys.argv[1],sys.argv[3],100)).to(device=device)
  trg_train = torch.tensor(data_process(sys.argv[2],sys.argv[4],20)).to(device=device)

  src_test = torch.tensor(data_process(sys.argv[1],sys.argv[5],100)).to(device=device)
  trg_test = torch.tensor(data_process(sys.argv[2],sys.argv[6],20)).to(device=device)

  src_valid = torch.tensor(data_process(sys.argv[1],sys.argv[7],100)).to(device=device)
  trg_valid = torch.tensor(data_process(sys.argv[2],sys.argv[8],20)).to(device=device)

  train=[]
  test=[]
  valid=[]
  for i in range(len(src_train)):
    t=(src_train.data[i],trg_train.data[i])
    train.append(t)
  for i in range(len(src_test)):
    t=(src_test.data[i],trg_test.data[i])
    test.append(t)
  for i in range(len(src_valid)):
    t = (src_valid.data[i],trg_valid.data[i])
    valid.append(t)

  train_iter = DataLoader(train, batch_size=256,
                        shuffle=True)
  valid_iter = DataLoader(valid, batch_size=256,
                        shuffle=True)
  test_iter = DataLoader(test, batch_size=256,
                       shuffle=True)
