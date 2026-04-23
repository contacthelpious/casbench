"""
CausalRec-Bench — Full Evaluation
Evaluates all 6 models on all scenarios
and saves verified results.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import os, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from evaluation.metrics import evaluate_model

print("=" * 65)
print("CausalRec-Bench — Full Evaluation")
print("=" * 65)
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# ─── LOAD DATA ────────────────────────────
print("Loading data...")
users = pd.read_csv('data/users.csv')
items = pd.read_csv('data/items.csv')
train_df = pd.read_csv('data/train.csv')
cold_start = pd.read_csv('data/cold_start.csv')
winter = pd.read_csv('data/winter_cold.csv')
summer = pd.read_csv('data/summer_cold.csv')
ecom_cold = pd.read_csv('data/ecom_cold.csv')
stream_cold = pd.read_csv('data/stream_cold.csv')
level1 = pd.read_csv('data/level1_simple.csv')
level3 = pd.read_csv('data/level3_hard.csv')
print(f"  Users: {len(users):,} | Items: {len(items):,}")
print(f"  Train: {len(train_df):,}")
print()

# ─── MAPS ─────────────────────────────────
user_map = {uid:idx for idx,uid in enumerate(train_df['user_id'].unique())}
item_map = {iid:idx for idx,iid in enumerate(items['item_id'].unique())}
reverse_item_map = {idx:iid for iid,idx in item_map.items()}
n_users = len(user_map)
n_items = len(item_map)

# ─── POPULARITY ───────────────────────────
item_pop = train_df[train_df['clicked']==True]['item_id'].value_counts()
ecom_items = items[items['domain']=='ecommerce']
stream_items = items[items['domain']=='streaming']
ecom_pop = train_df[(train_df['clicked']==True)&(train_df['domain']=='ecommerce')]['item_id'].value_counts()
stream_pop = train_df[(train_df['clicked']==True)&(train_df['domain']=='streaming')]['item_id'].value_counts()

def popularity(uid,uinfo,idf,k=10): return item_pop.head(k).index.tolist()
def pop_ecom(uid,uinfo,idf,k=10): return ecom_pop.head(k).index.tolist()
def pop_stream(uid,uinfo,idf,k=10): return stream_pop.head(k).index.tolist()

# ─── CAUSAL RULES ─────────────────────────
ECOM_CATEGORIES = ['electronics','books','clothing','food','home','health','outdoor']
STREAM_CATEGORIES = ['action','drama','comedy','documentary','thriller','animation','romance']
ECOM_AGE_CATEGORY = {
    'young': {'electronics':0.40,'books':0.15,'clothing':0.25,'food':0.08,'home':0.05,'health':0.04,'outdoor':0.03},
    'middle':{'electronics':0.20,'books':0.20,'clothing':0.15,'food':0.18,'home':0.15,'health':0.08,'outdoor':0.04},
    'senior':{'electronics':0.08,'books':0.35,'clothing':0.10,'food':0.18,'home':0.12,'health':0.15,'outdoor':0.02}
}
ECOM_GENDER_MODIFIER = {
    'male':  {'electronics':+0.10,'books':0.0,'clothing':-0.08,'food':0.0,'home':0.0,'health':-0.02,'outdoor':+0.05},
    'female':{'electronics':-0.08,'books':0.0,'clothing':+0.12,'food':0.0,'home':+0.02,'health':+0.05,'outdoor':-0.02}
}
ECOM_LIFE_STAGE_MODIFIER = {
    'student':     {'electronics':+0.08,'books':+0.15,'clothing':+0.05,'food':-0.05,'home':-0.08,'health':-0.05,'outdoor':0.0},
    'single_adult':{'electronics':+0.05,'books':0.0,'clothing':+0.10,'food':+0.05,'home':-0.05,'health':0.0,'outdoor':+0.05},
    'parent':      {'electronics':0.0,'books':+0.05,'clothing':0.0,'food':+0.10,'home':+0.15,'health':+0.05,'outdoor':0.0},
    'retired':     {'electronics':-0.05,'books':+0.10,'clothing':0.0,'food':+0.05,'home':+0.05,'health':+0.15,'outdoor':-0.05}
}
ECOM_SEASON_MODIFIER = {
    'summer':{'electronics':+0.02,'books':-0.08,'clothing':+0.05,'food':+0.05,'home':-0.05,'health':0.0,'outdoor':+0.15},
    'autumn':{'electronics':+0.05,'books':+0.05,'clothing':+0.10,'food':0.0,'home':+0.05,'health':0.0,'outdoor':-0.05},
    'winter':{'electronics':+0.05,'books':+0.15,'clothing':+0.08,'food':+0.05,'home':+0.10,'health':+0.05,'outdoor':-0.15},
    'spring':{'electronics':0.0,'books':0.0,'clothing':+0.10,'food':0.0,'home':0.0,'health':+0.05,'outdoor':+0.10}
}
ECOM_INCOME_PRICE = {
    'low':   {'budget':0.65,'mid_range':0.30,'premium':0.05},
    'medium':{'budget':0.20,'mid_range':0.55,'premium':0.25},
    'high':  {'budget':0.05,'mid_range':0.25,'premium':0.70}
}
STREAM_AGE_CATEGORY = {
    'young': {'action':0.35,'drama':0.10,'comedy':0.20,'documentary':0.05,'thriller':0.15,'animation':0.10,'romance':0.05},
    'middle':{'action':0.20,'drama':0.25,'comedy':0.18,'documentary':0.12,'thriller':0.12,'animation':0.05,'romance':0.08},
    'senior':{'action':0.10,'drama':0.35,'comedy':0.15,'documentary':0.25,'thriller':0.08,'animation':0.02,'romance':0.05}
}
STREAM_GENDER_MODIFIER = {
    'male':  {'action':+0.12,'drama':-0.05,'comedy':0.0,'documentary':+0.05,'thriller':+0.08,'animation':0.0,'romance':-0.10},
    'female':{'action':-0.08,'drama':+0.10,'comedy':+0.05,'documentary':0.0,'thriller':-0.03,'animation':+0.02,'romance':+0.12}
}
STREAM_LIFE_STAGE_MODIFIER = {
    'student':     {'action':+0.08,'drama':-0.05,'comedy':+0.10,'documentary':-0.05,'thriller':+0.05,'animation':+0.08,'romance':+0.05},
    'single_adult':{'action':+0.05,'drama':+0.05,'comedy':+0.08,'documentary':0.0,'thriller':+0.05,'animation':0.0,'romance':+0.08},
    'parent':      {'action':-0.05,'drama':+0.10,'comedy':+0.05,'documentary':+0.08,'thriller':-0.05,'animation':+0.15,'romance':0.0},
    'retired':     {'action':-0.10,'drama':+0.15,'comedy':+0.05,'documentary':+0.18,'thriller':-0.05,'animation':-0.05,'romance':+0.05}
}
STREAM_SEASON_MODIFIER = {
    'summer':{'action':+0.05,'drama':-0.05,'comedy':+0.08,'documentary':-0.05,'thriller':0.0,'animation':+0.05,'romance':+0.05},
    'autumn':{'action':0.0,'drama':+0.08,'comedy':0.0,'documentary':+0.05,'thriller':+0.08,'animation':0.0,'romance':+0.05},
    'winter':{'action':0.0,'drama':+0.12,'comedy':+0.05,'documentary':+0.10,'thriller':+0.10,'animation':+0.05,'romance':+0.08},
    'spring':{'action':+0.05,'drama':0.0,'comedy':+0.08,'documentary':0.0,'thriller':0.0,'animation':+0.05,'romance':+0.05}
}
STREAM_INCOME_LENGTH = {
    'low':   {'short':0.60,'medium':0.30,'long':0.10},
    'medium':{'short':0.25,'medium':0.50,'long':0.25},
    'high':  {'short':0.10,'medium':0.35,'long':0.55}
}

def calculate_genuine_preference(user, domain):
    if domain == 'ecommerce':
        cats=ECOM_CATEGORIES; ap=ECOM_AGE_CATEGORY
        gp=ECOM_GENDER_MODIFIER; lp=ECOM_LIFE_STAGE_MODIFIER
        sp=ECOM_SEASON_MODIFIER; ip=ECOM_INCOME_PRICE
    else:
        cats=STREAM_CATEGORIES; ap=STREAM_AGE_CATEGORY
        gp=STREAM_GENDER_MODIFIER; lp=STREAM_LIFE_STAGE_MODIFIER
        sp=STREAM_SEASON_MODIFIER; ip=STREAM_INCOME_LENGTH
    prefs={c:ap[user['age_group']][c] for c in cats}
    for c in cats:
        prefs[c]+=gp[user['gender']][c]
        prefs[c]+=lp[user['life_stage']][c]
        prefs[c]+=sp[user['season']][c]
        prefs[c]=max(0.01,prefs[c])
    total=sum(prefs.values())
    for c in cats: prefs[c]/=total
    return prefs, ip[user['income']]

genuine_ui = train_df[
    train_df['click_cause']=='genuine_preference'
].groupby(['user_id','item_id']).size().reset_index(name='cnt')

def causal_ub(uid, uinfo, idf, k=10):
    ug=genuine_ui[genuine_ui['user_id']==uid]
    domain=uinfo.get('domain_pref','ecommerce')
    if domain=='both': domain='ecommerce'
    if len(ug)==0:
        cp,sp=calculate_genuine_preference(uinfo,domain)
        tc=sorted(cp.items(),key=lambda x:x[1],reverse=True)[:2]
        tp=sorted(sp.items(),key=lambda x:x[1],reverse=True)[:2]
        tcn=[c for c,_ in tc]; tpn=[p for p,_ in tp]
        di=idf[idf['domain']==domain] if domain in ['ecommerce','streaming'] else idf
        m=di[di['category'].isin(tcn)&di['price_tier'].isin(tpn)]['item_id'].tolist()
        if len(m)>=k: return m[:k]
        m2=di[di['category'].isin(tcn)]['item_id'].tolist()
        m=list(set(m+m2))[:k]
        if len(m)<k:
            pop=[i for i in item_pop.index if i not in m][:k-len(m)]
            m=m+pop
        return m[:k]
    already=set(ug['item_id'].tolist())
    sim=genuine_ui[
        genuine_ui['item_id'].isin(already)&
        (genuine_ui['user_id']!=uid)
    ]['user_id'].value_counts().head(20).index.tolist()
    if not sim:
        cp,_=calculate_genuine_preference(uinfo,'ecommerce')
        tc=max(cp,key=cp.get)
        return idf[idf['category']==tc]['item_id'].head(k).tolist()
    recs=genuine_ui[
        genuine_ui['user_id'].isin(sim)&
        ~genuine_ui['item_id'].isin(already)
    ]['item_id'].value_counts().head(k).index.tolist()
    if len(recs)<k:
        pop=[i for i in item_pop.index if i not in recs][:k-len(recs)]
        recs=recs+pop
    return recs[:k]

# ─── LOAD FASTMF ──────────────────────────
print("Loading FastMF models...")
from models.fast_mf import FastMF
mf_std = FastMF(n_users, n_items)
mf_std.U = np.load('pretrained_models/fmf_std_U.npy')
mf_std.V = np.load('pretrained_models/fmf_std_V.npy')
print("  Standard MF loaded")

mf_caus = FastMF(n_users, n_items)
mf_caus.U = np.load('pretrained_models/fmf_caus_U.npy')
mf_caus.V = np.load('pretrained_models/fmf_caus_V.npy')
print("  Causal MF loaded")
print()

def mf_std_rec(uid,uinfo,idf,k=10):
    if uid not in user_map: return item_pop.head(k).index.tolist()
    return [reverse_item_map[i] for i in mf_std.recommend(user_map[uid],k) if i in reverse_item_map][:k]

def mf_caus_rec(uid,uinfo,idf,k=10):
    if uid not in user_map: return causal_ub(uid,uinfo,idf,k)
    return [reverse_item_map[i] for i in mf_caus.recommend(user_map[uid],k) if i in reverse_item_map][:k]

# ─── LOAD LIGHTGCN ────────────────────────
print("Loading LightGCN models...")

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
    return torch.sparse_coo_tensor(idx,val,torch.Size(norm.shape))

print("  Building graphs...")
adj_std=build_adj(train_df,user_map,item_map,n_users,n_items).to(device)
train_genuine=train_df[train_df['click_cause']=='genuine_preference'].copy()
adj_caus=build_adj(train_genuine,user_map,item_map,n_users,n_items).to(device)

lgcn_std=LightGCN(n_users,n_items,64,3,0.1).to(device)
lgcn_std.load_state_dict(torch.load('pretrained_models/lgcn_std.pt',map_location=device))
lgcn_std.eval()
print("  Standard LightGCN loaded")

lgcn_caus=LightGCN(n_users,n_items,64,3,0.1).to(device)
lgcn_caus.load_state_dict(torch.load('pretrained_models/lgcn_caus.pt',map_location=device))
lgcn_caus.eval()
print("  Causal LightGCN loaded")
print()

def lgcn_rec(model,adj,uid,uinfo,idf,k=10):
    model.eval()
    is_cold=(uinfo['new_user']=='cold_start' or uid not in user_map)
    if is_cold: return causal_ub(uid,uinfo,idf,k)
    with torch.no_grad():
        ue,ie=model.forward(adj)
    uv=ue[user_map[uid]].cpu().numpy()
    iv=ie.cpu().numpy()
    scores=iv.dot(uv)
    top=np.argsort(scores)[::-1]
    recs=[]
    for idx in top:
        if idx in reverse_item_map: recs.append(reverse_item_map[idx])
        if len(recs)>=k: break
    return recs

def lgcn_std_rec(uid,uinfo,idf,k=10): return lgcn_rec(lgcn_std,adj_std,uid,uinfo,idf,k)
def lgcn_caus_rec(uid,uinfo,idf,k=10): return lgcn_rec(lgcn_caus,adj_caus,uid,uinfo,idf,k)

# ─── ALL MODELS ───────────────────────────
all_models = [
    ('Popularity', popularity),
    ('Standard MF', mf_std_rec),
    ('Causal MF', mf_caus_rec),
    ('Standard LightGCN', lgcn_std_rec),
    ('Causal LightGCN', lgcn_caus_rec),
    ('Causal Upper Bound', causal_ub),
]

# ─── SCENARIOS ────────────────────────────
scenarios = [
    ('Cold-Start', cold_start, items, 3000),
    ('Winter Cold-Start', winter, items, 1000),
    ('Summer Cold-Start', summer, items, 1000),
    ('E-commerce Domain', ecom_cold, ecom_items, 2000),
    ('Streaming Domain', stream_cold, stream_items, 2000),
    ('Level 1 - Simple', level1, items, 2000),
    ('Level 3 - Hard', level3, items, 2000),
]

# ─── RUN EVALUATION ───────────────────────
print("=" * 65)
print("RUNNING FULL EVALUATION")
print("6 models x 7 scenarios")
print("=" * 65)
print()

all_results = []
t_start = time.time()

for sc_name,sc_data,sc_items,max_u in scenarios:
    print(f"Evaluating: {sc_name}...")
    t0=time.time()
    for mname,mfunc in all_models:
        r=evaluate_model(mname,mfunc,sc_data,users,sc_items,k=10,max_users=max_u)
        r['scenario']=sc_name
        all_results.append(r)
    print(f"  Done ({time.time()-t0:.0f}s)")

results_df=pd.DataFrame(all_results)
print(f"\nTotal time: {time.time()-t_start:.0f}s")

# ─── DISPLAY RESULTS ──────────────────────
print()
print("=" * 65)
print("RESULTS — ALL 6 MODELS")
print("=" * 65)

model_order=['Popularity','Standard MF','Causal MF',
             'Standard LightGCN','Causal LightGCN',
             'Causal Upper Bound']

for sc in results_df['scenario'].unique():
    s=results_df[results_df['scenario']==sc]
    print(f"\n{sc}:")
    print(f"  {'Model':<22} {'P@10':>7} {'N@10':>7} {'CP@10':>7} {'GenP':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    best=s['category_p@10'].max()
    for model in model_order:
        row=s[s['model']==model]
        if len(row)==0: continue
        r=row.iloc[0]
        marker=' *' if r['category_p@10']==best else ''
        print(f"  {r['model']:<22} {r['precision@10']:>7.4f} {r['ndcg@10']:>7.4f} {r['category_p@10']:>7.4f} {r['genuine_p@10']:>7.4f}{marker}")

# ─── KEY FINDINGS ─────────────────────────
print()
print("=" * 65)
print("KEY FINDINGS")
print("=" * 65)
print()

cold_std=results_df[(results_df['scenario']=='Cold-Start')&(results_df['model']=='Standard MF')]['category_p@10'].values[0]
cold_caus=results_df[(results_df['scenario']=='Cold-Start')&(results_df['model']=='Causal MF')]['category_p@10'].values[0]
imp1=(cold_caus-cold_std)/cold_std*100
print(f"Finding 1 — Causal MF vs Standard MF (Cold-Start CP@10):")
print(f"  Standard MF:  {cold_std:.4f}")
print(f"  Causal MF:    {cold_caus:.4f}")
print(f"  Improvement:  {imp1:+.1f}%")
print()

l3_std=results_df[(results_df['scenario']=='Level 3 - Hard')&(results_df['model']=='Standard LightGCN')]['category_p@10'].values[0]
l3_caus=results_df[(results_df['scenario']=='Level 3 - Hard')&(results_df['model']=='Causal LightGCN')]['category_p@10'].values[0]
imp2=(l3_caus-l3_std)/l3_std*100
print(f"Finding 2 — Causal LightGCN vs Standard LightGCN (Level 3 CP@10):")
print(f"  Standard LightGCN: {l3_std:.4f}")
print(f"  Causal LightGCN:   {l3_caus:.4f}")
print(f"  Improvement:       {imp2:+.1f}%")
print()

stream_pop=results_df[(results_df['scenario']=='Streaming Domain')&(results_df['model']=='Popularity')]['category_p@10'].values[0]
ecom_pop=results_df[(results_df['scenario']=='E-commerce Domain')&(results_df['model']=='Popularity')]['category_p@10'].values[0]
print(f"Finding 3 — Domain gap (Popularity):")
print(f"  E-commerce: {ecom_pop:.4f}")
print(f"  Streaming:  {stream_pop:.4f}")
print(f"  Gap:        {ecom_pop-stream_pop:+.4f}")
print()

# Save results
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/final_results.csv', index=False)
print("=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)
print()
print("Results saved to results/final_results.csv")
print()
print("Next: python VERIFY_RESULTS.py")
