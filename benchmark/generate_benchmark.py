"""
CausalRec-Bench — Full Benchmark Generation
Generates all data from scratch.
Runtime: approximately 20-30 minutes
"""

import pandas as pd
import numpy as np
import os
import time

print("=" * 60)
print("CausalRec-Bench — Benchmark Generation")
print("=" * 60)
print()

np.random.seed(42)
os.makedirs('data', exist_ok=True)

# ─── CONFIG ───────────────────────────────
N_USERS = 50000
N_ITEMS_PER_DOMAIN = 2000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10

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
POSITION_BIAS = {1:0.25,2:0.18,3:0.13,4:0.09,5:0.06,6:0.04,7:0.03,8:0.02,9:0.01,10:0.01}

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

# ─── STEP 1: GENERATE USERS ───────────────
print("Step 1: Generating users...")
t0=time.time()

users = pd.DataFrame({
    'user_id': range(1, N_USERS+1),
    'age_group': np.random.choice(['young','middle','senior'], N_USERS, p=[0.35,0.45,0.20]),
    'gender': np.random.choice(['male','female'], N_USERS, p=[0.48,0.52]),
    'income': np.random.choice(['low','medium','high'], N_USERS, p=[0.30,0.45,0.25]),
    'life_stage': np.random.choice(['student','single_adult','parent','retired'], N_USERS, p=[0.20,0.25,0.35,0.20]),
    'location': np.random.choice(['urban','suburban','rural'], N_USERS, p=[0.45,0.35,0.20]),
    'new_user': np.random.choice(['cold_start','warm'], N_USERS, p=[0.20,0.80]),
    'season': np.random.choice(['summer','autumn','winter','spring'], N_USERS, p=[0.25,0.25,0.25,0.25]),
    'domain_pref': np.random.choice(['ecommerce','streaming','both'], N_USERS, p=[0.35,0.30,0.35]),
})

users.to_csv('data/users.csv', index=False)
print(f"  {len(users):,} users generated ({time.time()-t0:.0f}s)")
print(f"  Cold-start: {(users['new_user']=='cold_start').sum():,} ({(users['new_user']=='cold_start').mean():.1%})")

# ─── STEP 2: GENERATE ITEMS ───────────────
print()
print("Step 2: Generating items...")
t0=time.time()

n = N_ITEMS_PER_DOMAIN
ecom_items = pd.DataFrame({
    'item_id': range(1, n+1),
    'domain': 'ecommerce',
    'category': np.random.choice(ECOM_CATEGORIES, n, p=[0.20,0.15,0.20,0.15,0.15,0.10,0.05]),
    'price_tier': np.random.choice(['budget','mid_range','premium'], n, p=[0.40,0.40,0.20]),
    'popularity': np.random.choice(['low','medium','high'], n, p=[0.60,0.30,0.10]),
    'promotion': np.random.choice(['promoted','not_promoted'], n, p=[0.20,0.80]),
    'item_age': np.random.choice(['new','established'], n, p=[0.15,0.85]),
    'seasonal_relevance': np.random.choice(['summer','autumn','winter','spring','all_seasons'], n, p=[0.10,0.10,0.10,0.10,0.60]),
    'avg_position': np.random.choice(range(1,11), n),
})

stream_items = pd.DataFrame({
    'item_id': range(n+1, n*2+1),
    'domain': 'streaming',
    'category': np.random.choice(STREAM_CATEGORIES, n, p=[0.20,0.20,0.15,0.15,0.15,0.08,0.07]),
    'price_tier': np.random.choice(['short','medium','long'], n, p=[0.30,0.45,0.25]),
    'popularity': np.random.choice(['low','medium','high'], n, p=[0.60,0.30,0.10]),
    'promotion': np.random.choice(['promoted','not_promoted'], n, p=[0.15,0.85]),
    'item_age': np.random.choice(['new','established'], n, p=[0.20,0.80]),
    'seasonal_relevance': np.random.choice(['summer','autumn','winter','spring','all_seasons'], n, p=[0.10,0.10,0.10,0.10,0.60]),
    'avg_position': np.random.choice(range(1,11), n),
})

all_items = pd.concat([ecom_items, stream_items], ignore_index=True)
all_items.to_csv('data/items.csv', index=False)
ecom_items.to_csv('data/items_ecommerce.csv', index=False)
stream_items.to_csv('data/items_streaming.csv', index=False)
print(f"  {len(all_items):,} items generated ({time.time()-t0:.0f}s)")

# ─── STEP 3: SIMULATE INTERACTIONS ────────
print()
print("Step 3: Simulating interactions...")
print("  This takes 15-20 minutes...")
t0=time.time()

all_interactions = []
items_dict = all_items.set_index('item_id').to_dict('index')
ecom_item_ids = list(ecom_items['item_id'])
stream_item_ids = list(stream_items['item_id'])

ITEMS_PER_USER = 60
MIN_GENUINE = 3
BATCH_SIZE = 1000

for batch_start in range(0, N_USERS, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, N_USERS)
    batch_users = users.iloc[batch_start:batch_end]

    for _, user in batch_users.iterrows():
        dp = user['domain_pref']
        if dp == 'ecommerce':
            domains_items = [('ecommerce', ecom_item_ids, ITEMS_PER_USER)]
        elif dp == 'streaming':
            domains_items = [('streaming', stream_item_ids, ITEMS_PER_USER)]
        else:
            domains_items = [
                ('ecommerce', ecom_item_ids, ITEMS_PER_USER//2),
                ('streaming', stream_item_ids, ITEMS_PER_USER//2)
            ]

        for domain, domain_item_ids, n_items in domains_items:
            cp, sp = calculate_genuine_preference(user, domain)
            cats = list(cp.keys())

            genuine_count = 0
            attempts = 0

            while genuine_count < MIN_GENUINE and attempts < 3:
                sampled = np.random.choice(domain_item_ids,
                    size=min(n_items, len(domain_item_ids)),
                    replace=False)
                attempts += 1

                for item_id in sampled:
                    item = items_dict[item_id]

                    # Exposure probability
                    base_exp = 0.35
                    if item['promotion'] == 'promoted': base_exp += 0.40
                    if item['popularity'] == 'high': base_exp += 0.30
                    elif item['popularity'] == 'medium': base_exp += 0.10
                    if item['item_age'] == 'new': base_exp -= 0.20
                    base_exp = max(0.05, min(0.95, base_exp))

                    if np.random.random() > base_exp:
                        continue

                    # Genuine match score
                    cat_match = cp.get(item['category'], 0.01)
                    price_match = sp.get(item['price_tier'], 0.01)
                    genuine_match = cat_match * 0.6 + price_match * 0.4

                    # Click probability with confounders
                    click_prob = genuine_match * 0.8
                    if item['promotion'] == 'promoted': click_prob += 0.15
                    if item['popularity'] == 'high': click_prob += 0.12
                    pos = int(item['avg_position'])
                    click_prob += POSITION_BIAS.get(pos, 0.01)
                    if user['new_user'] == 'cold_start' and item['promotion'] == 'promoted':
                        click_prob += 0.10
                    click_prob = max(0.01, min(0.95, click_prob))

                    clicked = np.random.random() < click_prob

                    # Causal label
                    if clicked:
                        if genuine_match > 0.30:
                            cause = 'genuine_preference'
                            genuine_count += 1
                        elif item['promotion'] == 'promoted':
                            cause = 'promotion_bias'
                        elif item['popularity'] == 'high':
                            cause = 'popularity_bias'
                        elif POSITION_BIAS.get(pos, 0) > 0.10:
                            cause = 'position_bias'
                        else:
                            cause = 'mixed'
                    else:
                        cause = 'no_click'

                    purchased = False
                    if clicked:
                        purchased = np.random.random() < (cat_match * 0.5 + price_match * 0.5) * 0.6

                    all_interactions.append({
                        'user_id': user['user_id'],
                        'item_id': item_id,
                        'domain': domain,
                        'age_group': user['age_group'],
                        'gender': user['gender'],
                        'income': user['income'],
                        'life_stage': user['life_stage'],
                        'season': user['season'],
                        'new_user': user['new_user'],
                        'category': item['category'],
                        'price_tier': item['price_tier'],
                        'popularity': item['popularity'],
                        'promotion': item['promotion'],
                        'item_age': item['item_age'],
                        'avg_position': item['avg_position'],
                        'seasonal_relevance': item['seasonal_relevance'],
                        'clicked': clicked,
                        'purchased': purchased,
                        'genuine_match': round(genuine_match, 4),
                        'click_cause': cause,
                        'position_bias': POSITION_BIAS.get(pos, 0.01)
                    })

    if (batch_start + BATCH_SIZE) % 5000 == 0 or batch_end == N_USERS:
        elapsed = time.time()-t0
        pct = batch_end/N_USERS
        eta = elapsed/pct*(1-pct) if pct > 0 else 0
        print(f"  Users {batch_end:,}/{N_USERS:,} | "
              f"Interactions: {len(all_interactions):,} | "
              f"ETA: {eta/60:.0f}min")

interactions_df = pd.DataFrame(all_interactions)
interactions_df = interactions_df.drop_duplicates(
    subset=['user_id','item_id'], keep='first'
)
interactions_df.to_csv('data/interactions.csv', index=False)

clicked = interactions_df[interactions_df['clicked']==True]
genuine = interactions_df[interactions_df['click_cause']=='genuine_preference']
print(f"  Total interactions: {len(interactions_df):,}")
print(f"  Total clicks: {len(clicked):,}")
print(f"  Genuine clicks: {len(genuine):,} ({len(genuine)/len(clicked):.1%})")
print(f"  Non-genuine: {1-len(genuine)/len(clicked):.1%}")
print(f"  Time: {(time.time()-t0)/60:.1f} minutes")

# ─── STEP 4: CREATE SPLITS ────────────────
print()
print("Step 4: Creating evaluation splits...")
t0=time.time()

all_user_ids = interactions_df['user_id'].unique()
np.random.shuffle(all_user_ids)
n = len(all_user_ids)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_users = all_user_ids[:train_end]
val_users = all_user_ids[train_end:val_end]
test_users = all_user_ids[val_end:]

train_df = interactions_df[interactions_df['user_id'].isin(train_users)]
val_df = interactions_df[interactions_df['user_id'].isin(val_users)]
test_df = interactions_df[interactions_df['user_id'].isin(test_users)]

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print(f"  Train: {len(train_df):,} ({len(train_users):,} users)")
print(f"  Val:   {len(val_df):,} ({len(val_users):,} users)")
print(f"  Test:  {len(test_df):,} ({len(test_users):,} users)")

# Cold-start
cold = interactions_df[interactions_df['new_user']=='cold_start']
cold.to_csv('data/cold_start.csv', index=False)
print(f"  Cold-start: {len(cold):,}")

# Difficulty levels
level1 = interactions_df[
    (interactions_df['promotion']=='not_promoted') &
    (interactions_df['popularity']=='low') &
    (interactions_df['avg_position']>=5)
]
level2 = interactions_df[interactions_df['promotion']=='not_promoted']
level3 = interactions_df.copy()

level1.to_csv('data/level1_simple.csv', index=False)
level2.to_csv('data/level2_medium.csv', index=False)
level3.to_csv('data/level3_hard.csv', index=False)
print(f"  Level1: {len(level1):,} | Level2: {len(level2):,} | Level3: {len(level3):,}")

# Seasonal splits
for season in ['winter','summer','autumn','spring']:
    s = interactions_df[
        (interactions_df['season']==season) &
        (interactions_df['new_user']=='cold_start')
    ]
    s.to_csv(f'data/{season}_cold.csv', index=False)
    print(f"  {season}_cold: {len(s):,}")

# Domain splits
for domain, short in [('ecommerce','ecom'),('streaming','stream')]:
    d_cold = interactions_df[
        (interactions_df['domain']==domain) &
        (interactions_df['new_user']=='cold_start')
    ]
    d_only = interactions_df[interactions_df['domain']==domain]
    d_cold.to_csv(f'data/{short}_cold.csv', index=False)
    d_only.to_csv(f'data/{short}_only.csv', index=False)
    print(f"  {short}_cold: {len(d_cold):,} | {short}_only: {len(d_only):,}")

# Position splits
high_pos = interactions_df[interactions_df['avg_position']<=3]
low_pos = interactions_df[interactions_df['avg_position']>=8]
high_pos.to_csv('data/high_position.csv', index=False)
low_pos.to_csv('data/low_position.csv', index=False)
print(f"  High position: {len(high_pos):,} | Low position: {len(low_pos):,}")

# Cold start profiles
cold_profiles = users[users['new_user']=='cold_start']
cold_profiles.to_csv('data/cold_start_profiles.csv', index=False)

print()
print("=" * 60)
print("BENCHMARK GENERATION COMPLETE")
print("=" * 60)
print(f"  Users:        {len(users):,}")
print(f"  Items:        {len(all_items):,}")
print(f"  Interactions: {len(interactions_df):,}")
print(f"  Splits saved: 18")
print()
print("Next: python benchmark/train_models.py")
