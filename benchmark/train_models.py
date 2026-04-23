"""
CausalRec-Bench — Train All Models
Trains Standard MF, Causal MF,
Standard LightGCN, Causal LightGCN
Runtime: approximately 20-30 minutes
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import os, sys, time
sys.path.insert(0, '.')
from models.fast_mf import FastMF

print("=" * 60)
print("CausalRec-Bench — Training All Models")
print("=" * 60)
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# ─── LOAD DATA ────────────────────────────
print("Loading data...")
users = pd.read_csv('data/users.csv')
items = pd.read_csv('data/items.csv')
train_df = pd.read_csv('data/train.csv')

print(f"  Users: {len(users):,}")
print(f"  Items: {len(items):,}")
print(f"  Train: {len(train_df):,}")
print()

# ─── MAPS ─────────────────────────────────
user_map = {uid:idx for idx,uid in enumerate(train_df['user_id'].unique())}
item_map = {iid:idx for idx,iid in enumerate(items['item_id'].unique())}
reverse_item_map = {idx:iid for iid,idx in item_map.items()}
n_users = len(user_map)
n_items = len(item_map)

print(f"Maps: {n_users:,} users, {n_items:,} items")
print()

# Genuine training data
train_genuine = train_df[
    train_df['click_cause']=='genuine_preference'
].copy()

all_count = len(train_df[train_df['clicked']==True])
gen_count = len(train_genuine[train_genuine['clicked']==True])
print(f"Standard training clicks: {all_count:,}")
print(f"Genuine training clicks:  {gen_count:,}")
print(f"Biased clicks removed:    {all_count-gen_count:,} ({(all_count-gen_count)/all_count:.1%})")
print()

# ─── TRAIN FAST MF ────────────────────────
print("=" * 50)
print("Training Standard FastMF...")
print("=" * 50)
t0=time.time()

mf_std = FastMF(n_users, n_items, n_factors=32,
                learning_rate=0.005, reg=0.01,
                n_epochs=20, random_seed=42)
mf_std.fit(train_df, user_map, item_map, verbose=True)
mf_std.save('pretrained_models/fmf_std')
print(f"Saved to pretrained_models/fmf_std")
print(f"Time: {time.time()-t0:.0f}s")
print()

print("=" * 50)
print("Training Causal FastMF...")
print("=" * 50)
t0=time.time()

mf_caus = FastMF(n_users, n_items, n_factors=32,
                 learning_rate=0.005, reg=0.01,
                 n_epochs=20, random_seed=42)
mf_caus.fit(train_genuine, user_map, item_map, verbose=True)
mf_caus.save('pretrained_models/fmf_caus')
print(f"Saved to pretrained_models/fmf_caus")
print(f"Time: {time.time()-t0:.0f}s")
print()

# ─── LIGHTGCN ─────────────────────────────
class LightGCN(nn.Module):
    def __init__(self,n_users,n_items,emb_dim=64,n_layers=3,dropout=0.1):
        super().__init__()
        self.n_users=n_users; self.n_items=n_items
        self.n_layers=n_layers; self.dropout=dropout
        self.user_emb=nn.Embedding(n_users,emb_dim)
        self.item_emb=nn.Embedding(n_items,emb_dim)
        nn.init.normal_(self.user_emb.weight,std=0.01)
        nn.init.normal_(self.item_emb.weight,std=0.01)

    def forward(self,adj):
        emb=torch.cat([self.user_emb.weight,self.item_emb.weight],dim=0)
        layers=[emb]
        for _ in range(self.n_layers):
            emb=torch.sparse.mm(adj,emb)
            if self.training: emb=F.dropout(emb,p=self.dropout)
            layers.append(emb)
        final=torch.mean(torch.stack(layers,dim=0),dim=0)
        return final[:self.n_users],final[self.n_users:]

    def bpr_loss(self,u,pos,neg,adj):
        ue,ie=self.forward(adj)
        u_=ue[u]; p_=ie[pos]; n_=ie[neg]
        ps=(u_*p_).sum(dim=1); ns=(u_*n_).sum(dim=1)
        loss=-torch.log(torch.sigmoid(ps-ns)+1e-8).mean()
        reg=(u_.norm(2).pow(2)+p_.norm(2).pow(2)+n_.norm(2).pow(2))/len(u)
        return loss+0.001*reg

class InteractionDataset(Dataset):
    def __init__(self,interactions,user_map,item_map):
        self.all_items=np.array(list(item_map.values()))
        pos=interactions[interactions['clicked']==True][['user_id','item_id']].values
        self.pairs=[]; self.user_pos={}
        for uid,iid in pos:
            if uid in user_map and iid in item_map:
                u=user_map[uid]; i=item_map[iid]
                self.pairs.append((u,i))
                if u not in self.user_pos: self.user_pos[u]=set()
                self.user_pos[u].add(i)

    def __len__(self): return len(self.pairs)

    def __getitem__(self,idx):
        u,pos=self.pairs[idx]
        neg=np.random.choice(self.all_items)
        while neg in self.user_pos.get(u,set()):
            neg=np.random.choice(self.all_items)
        return torch.LongTensor([u]),torch.LongTensor([pos]),torch.LongTensor([neg])

def build_adj(interactions,user_map,item_map,n_users,n_items):
    pos=interactions[interactions['clicked']==True][['user_id','item_id']].values
    rows,cols=[],[]
    for uid,iid in pos:
        if uid in user_map and iid in item_map:
            rows.append(user_map[uid]); cols.append(item_map[iid])
    R=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n_users,n_items))
    up=sp.hstack([sp.csr_matrix((n_users,n_users)),R])
    lo=sp.hstack([R.T,sp.csr_matrix((n_items,n_items))])
    full=sp.vstack([up,lo])
    rs=np.array(full.sum(1)).flatten(); rs[rs==0]=1
    d=sp.diags(np.power(rs,-0.5))
    norm=d.dot(full).dot(d).tocoo()
    idx=torch.LongTensor(np.vstack([norm.row,norm.col]))
    val=torch.FloatTensor(norm.data)
    return torch.sparse_coo_tensor(idx,val,torch.Size(norm.shape)),R

def train_lgcn(model,adj,interactions,user_map,item_map,
               n_epochs=50,batch_size=2048,lr=0.001,name="LightGCN"):
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    ds=InteractionDataset(interactions,user_map,item_map)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0)
    print(f"  Training on {len(ds):,} interactions...")
    model.train()
    for epoch in range(n_epochs):
        total=0; n=0
        for u,pos,neg in loader:
            u=u.squeeze().to(device)
            pos=pos.squeeze().to(device)
            neg=neg.squeeze().to(device)
            opt.zero_grad()
            loss=model.bpr_loss(u,pos,neg,adj)
            loss.backward(); opt.step()
            total+=loss.item(); n+=1
        if (epoch+1)%10==0:
            print(f"  Epoch {epoch+1}/{n_epochs} Loss:{total/n:.4f}")
    print("  Done")
    return model

# Build graphs
print("=" * 50)
print("Building LightGCN graphs...")
print("=" * 50)
t0=time.time()

adj_std,R_std=build_adj(train_df,user_map,item_map,n_users,n_items)
adj_std=adj_std.to(device)
adj_caus,R_caus=build_adj(train_genuine,user_map,item_map,n_users,n_items)
adj_caus=adj_caus.to(device)

print(f"  Standard graph edges: {R_std.nnz:,}")
print(f"  Causal graph edges:   {R_caus.nnz:,}")
print(f"  Biased edges removed: {R_std.nnz-R_caus.nnz:,}")
print(f"  Time: {time.time()-t0:.0f}s")
print()

# Train Standard LightGCN
print("=" * 50)
print("Training Standard LightGCN...")
print("=" * 50)
t0=time.time()
torch.manual_seed(42)
lgcn_std=LightGCN(n_users,n_items,64,3,0.1).to(device)
lgcn_std=train_lgcn(lgcn_std,adj_std,train_df,
                    user_map,item_map,
                    n_epochs=50,batch_size=2048,
                    lr=0.001,name="Standard LightGCN")
torch.save(lgcn_std.state_dict(),'pretrained_models/lgcn_std.pt')
std_final_loss = None
print(f"Saved to pretrained_models/lgcn_std.pt")
print(f"Time: {time.time()-t0:.0f}s")
print()

# Train Causal LightGCN
print("=" * 50)
print("Training Causal LightGCN...")
print("=" * 50)
t0=time.time()
torch.manual_seed(42)
lgcn_caus=LightGCN(n_users,n_items,64,3,0.1).to(device)
lgcn_caus=train_lgcn(lgcn_caus,adj_caus,train_genuine,
                     user_map,item_map,
                     n_epochs=50,batch_size=2048,
                     lr=0.001,name="Causal LightGCN")
torch.save(lgcn_caus.state_dict(),'pretrained_models/lgcn_caus.pt')
print(f"Saved to pretrained_models/lgcn_caus.pt")
print(f"Time: {time.time()-t0:.0f}s")
print()

# Verify all saved
print("=" * 60)
print("ALL MODELS TRAINED AND SAVED")
print("=" * 60)
print()
for f in [
    'pretrained_models/fmf_std_U.npy',
    'pretrained_models/fmf_std_V.npy',
    'pretrained_models/fmf_caus_U.npy',
    'pretrained_models/fmf_caus_V.npy',
    'pretrained_models/lgcn_std.pt',
    'pretrained_models/lgcn_caus.pt',
]:
    if os.path.exists(f):
        size=os.path.getsize(f)/1024/1024
        print(f"  OK  {f} ({size:.1f} MB)")
    else:
        print(f"  MISSING  {f}")

print()
print("Next: python benchmark/run_evaluation.py")
