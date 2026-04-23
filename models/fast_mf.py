import numpy as np

class FastMF:
    def __init__(self, n_users, n_items, n_factors=32,
                 learning_rate=0.005, reg=0.01,
                 n_epochs=20, random_seed=42):
        self.n_factors=n_factors
        self.lr=learning_rate
        self.reg=reg
        self.n_epochs=n_epochs
        np.random.seed(random_seed)
        self.U=np.random.normal(0,0.1,(n_users,n_factors)).astype(np.float32)
        self.V=np.random.normal(0,0.1,(n_items,n_factors)).astype(np.float32)

    def fit(self, interactions, user_map, item_map, verbose=True):
        pos=interactions[interactions['clicked']==True][['user_id','item_id']].values
        mask=[r[0] in user_map and r[1] in item_map for r in pos]
        pos=pos[mask]
        u_idx=np.array([user_map[r[0]] for r in pos])
        i_idx=np.array([item_map[r[1]] for r in pos])
        n=len(u_idx)
        item_list=np.array(list(item_map.values()))
        if verbose: print(f"  Training on {n:,} interactions...")
        for epoch in range(self.n_epochs):
            perm=np.random.permutation(n)
            u_idx=u_idx[perm]; i_idx=i_idx[perm]
            total_loss=0
            for s in range(0,n,4096):
                e=min(s+4096,n)
                bu=u_idx[s:e]; bi=i_idx[s:e]
                uv=self.U[bu]; iv=self.V[bi]
                ps=np.sum(uv*iv,axis=1); err=1.0-ps
                self.U[bu]+=self.lr*(err[:,None]*iv-self.reg*uv)
                self.V[bi]+=self.lr*(err[:,None]*uv-self.reg*iv)
                total_loss+=np.sum(err**2)
                ni=np.random.choice(item_list,size=len(bu))
                nv=self.V[ni]; ne=0.0-np.sum(self.U[bu]*nv,axis=1)
                self.V[ni]+=self.lr*(ne[:,None]*self.U[bu]-self.reg*nv)
            if verbose and (epoch+1)%5==0:
                print(f"  Epoch {epoch+1}/{self.n_epochs} Loss:{total_loss/n:.4f}")
        if verbose: print("  Done")

    def recommend(self, u_idx, k=10):
        scores=self.V.dot(self.U[u_idx])
        return np.argsort(scores)[::-1][:k].tolist()

    def save(self, path_prefix):
        np.save(f'{path_prefix}_U.npy', self.U)
        np.save(f'{path_prefix}_V.npy', self.V)

    def load(self, path_prefix):
        self.U=np.load(f'{path_prefix}_U.npy')
        self.V=np.load(f'{path_prefix}_V.npy')
