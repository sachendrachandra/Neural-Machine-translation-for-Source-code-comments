# data processing and training details
#_____________________________________________________________________________________________________
#Transformer Code

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import time

class SelfAttention(nn.Module):
  def __init__(self,embed_size,heads):
    super(SelfAttention,self).__init__()
    self.embed_size=embed_size
    self.heads=heads
    self.head_dim= embed_size // heads

    assert (self.head_dim*heads == embed_size), "Embedding size need to be divisible by head"

    self.values=nn.Linear(self.head_dim,self.head_dim,bias=False)
    self.keys=nn.Linear(self.head_dim,self.head_dim,bias=False)
    self.queries=nn.Linear(self.head_dim,self.head_dim,bias=False)
    self.fc_out=nn.Linear(heads*self.head_dim,embed_size)

  def forward(self,values,keys,query,mask):
    N=query.shape[0]
    value_len, key_len, query_len = values.shape[1],keys.shape[1],query.shape[1]

    #Split embeddings into self.heads pieces
    values = values.reshape(N,value_len,self.heads,self.head_dim)
    keys = keys.reshape(N,key_len,self.heads,self.head_dim)
    queries = query.reshape(N,query_len,self.heads,self.head_dim)

    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)

    energy= torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
    #queries shape:(N,query_len,heads,head_dim)
    #keys shape:(N,key_len,heads,head_dim)
    #energy shape:(N,heads,query_len,key_len)

    if mask is not None:
      energy = energy.masked_fill(mask == 0, float("-1e28"))

    attention = torch.softmax(energy / self.embed_size **(1/2),dim=3)

    out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(
        N,query_len,self.heads*self.head_dim
    ) 

    # attention shape:(N,heads,query_len,key_len)
    # values shape:(N,value_len,heads,heads_dim)
    # after einsum shape:(N,query_len,heads,head_dim)

    out=self.fc_out(out)
    return out

class TransformerBlock(nn.Module):
  def __init__(self,embed_size,heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
        nn.Linear(embed_size, forward_expansion*embed_size),
        nn.ReLU(),
        nn.Linear(forward_expansion*embed_size, embed_size)
    )
    self.dropout=nn.Dropout(dropout)

  def forward(self, value, key, query, mask):
    attention = self.attention(value,key,query,mask)

    x = self.dropout(self.norm1(attention + query))
    forward = self.feed_forward(x)
    out = self.dropout(self.norm2(forward + x))

    return out

class Encoder(nn.Module):
  def __init__(self, src_vocab_size, embed_size, num_layers,
      heads,
      device,
      forward_expansion,
      dropout,
      max_length):
    
    super(Encoder, self).__init__()
    self.embed_size = embed_size
    self.device = device
    self.word_embeddings = nn.Embedding(src_vocab_size, embed_size)
    self.position_embeddings = nn.Embedding(src_vocab_size, embed_size)

    self.layers = nn.ModuleList(
        [
         TransformerBlock(
             embed_size,
             heads,
             dropout = dropout,
             forward_expansion = forward_expansion 
             )
        for _ in range(num_layers)]
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    N, seq_length = x.shape
    positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

    out = self.dropout(self.word_embeddings(x) + self.position_embeddings(positions))

    for layer in self.layers:
      out = layer(out, out, out, mask)
    return out 

class DecoderBlock(nn.Module):
  def __init__(self, embed_size, heads, forward_expansion, dropout, device):
    super(DecoderBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm = nn.LayerNorm(embed_size)
    self.transformer_block = TransformerBlock(
        embed_size, heads, dropout, forward_expansion
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, value, key, src_mask, trg_mask):
    attention = self.attention(x, x, x, trg_mask)
    query = self.dropout(self.norm(attention + x))
    out = self.transformer_block(value, key, query, src_mask)

    return out

class Decoder(nn.Module):
  def __init__(
      self,
      trg_vocab_size,
      embed_size,
      num_layers,
      heads,
      forward_expansion,
      dropout,
      device,
      max_length,):
    super(Decoder, self).__init__()
    self.device = device
    self.word_embedding = nn.Embedding(trg_vocab_size,embed_size)
    self.position_embedding = nn.Embedding(max_length,embed_size)

    self.layers = nn.ModuleList(
        [DecoderBlock(embed_size,heads,forward_expansion,dropout, device)
        for _ in range(num_layers)]
    )

    self.fc_out = nn.Linear(embed_size, trg_vocab_size)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, enc_out, src_mask, trg_mask):
    N, seq_length = x.shape
    positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)
    x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

    for layer in self.layers:
      x = layer(x, enc_out, enc_out, src_mask, trg_mask)
    
    out = self.fc_out(x)

    return out

class Transformer(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, forward_expansion=4, embed_size = 256, num_layers = 2, heads = 8, dropout=0, device="cuda", max_length=100):
    super(Transformer, self).__init__()
    self.encoder = Encoder(
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion, 
        dropout,
        max_length       
    )

    self.decoder = Decoder(
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    )

    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx
    self.device = device

  def make_src_mask(self, src):
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    return src_mask.to(self.device)
  
  def make_trg_mask(self, trg):

    N, trg_len =trg.shape
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        N, 1, trg_len, trg_len
    )

    return trg_mask.to(self.device)

  def forward(self, src, trg):
    src_mask = self.make_src_mask(src)
    trg_mask = self.make_trg_mask(trg)
    enc_src = self.encoder(src, src_mask)
    out = self.decoder(trg, enc_src, src_mask, trg_mask)
    return out
#________________________________________________________________________________________________________________

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

#_______________________________________________________________________________________________________________________________
# Training and Evaluation methods

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


criterion = nn.CrossEntropyLoss(ignore_index=2)

def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            print(output.shape)
            trg = trg[1:].view(-1)
            print(trg.shape)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#___________________________________________________________________________________________________________________
# Training and saving the transformer model

src_pad_idx = 2
trg_pad_idx = 2
src_vocab_size=30000
trg_vocab_size=30000
model=Transformer(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx).to(device=device)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

N_EPOCHS = 6
CLIP = 1
t=1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iter, criterion)
print("test loss: ",test_loss)

torch.save(model,sys.argv[9])
