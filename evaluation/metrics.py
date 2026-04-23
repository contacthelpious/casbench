import numpy as np
import pandas as pd

def precision_at_k(rec, rel, k=10):
    return len(set(rec[:k])&set(rel))/k if k else 0

def recall_at_k(rec, rel, k=10):
    h=len(set(rec[:k])&set(rel))
    return h/len(rel) if rel else 0

def ndcg_at_k(rec, rel, k=10):
    dcg=sum(1/np.log2(i+2) for i,x in enumerate(rec[:k]) if x in rel)
    idcg=sum(1/np.log2(i+2) for i in range(min(len(rel),k)))
    return dcg/idcg if idcg else 0

def genuine_precision_at_k(rec, gen, k=10):
    return len(set(rec[:k])&set(gen))/k if k else 0

def category_precision_at_k(rec, gc, items_df, k=10):
    rc=items_df[items_df['item_id'].isin(rec[:k])]['category'].tolist()
    return sum(1 for c in rc if c in gc)/k if k else 0

def hit_rate_at_k(rec, rel, k=10):
    return int(len(set(rec[:k])&set(rel))>0)

def mrr_at_k(rec, rel, k=10):
    for i,x in enumerate(rec[:k]):
        if x in rel: return 1.0/(i+1)
    return 0.0

def evaluate_model(
    model_name, model_func, test_data,
    users_df, items_df, k=10, max_users=None
):
    uids = test_data['user_id'].unique()
    if max_users and len(uids) > max_users:
        np.random.seed(42)
        uids = np.random.choice(uids, max_users, replace=False)
    P,R,N,G,C=[],[],[],[],[]
    for uid in uids:
        ud = test_data[test_data['user_id']==uid]
        rel = ud[ud['clicked']==True]['item_id'].tolist()
        gen = ud[ud['click_cause']=='genuine_preference']['item_id'].tolist()
        gcat = ud[ud['click_cause']=='genuine_preference']['category'].tolist()
        if not rel: continue
        ui = users_df[users_df['user_id']==uid].iloc[0]
        recs = model_func(uid, ui, items_df, k)
        P.append(precision_at_k(recs,rel,k))
        R.append(recall_at_k(recs,rel,k))
        N.append(ndcg_at_k(recs,rel,k))
        G.append(genuine_precision_at_k(recs,gen,k))
        if gcat:
            C.append(category_precision_at_k(recs,gcat,items_df,k))
    return {
        'model': model_name,
        'precision@10': np.mean(P),
        'recall@10': np.mean(R),
        'ndcg@10': np.mean(N),
        'genuine_p@10': np.mean(G),
        'category_p@10': np.mean(C) if C else 0
    }
