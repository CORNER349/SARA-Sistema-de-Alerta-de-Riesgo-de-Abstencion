import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from prep import *
fr=load(); N=nat(fr)
MID={2009,2015,2021,2027}
for y in YRS:
    d=fr[y]; d["r"]=d.abst-N[y]
    d["mkey"]=d.EDOCVE.astype(str).str.zfill(2)+"-"+d.MPIOCVE.astype(str).str.zfill(3)
    d["nsr"]=d.NS/d.LN.clip(lower=1)
def munr(y):
    d=fr[y]; g=d.groupby("mkey").agg(nv=("NV","sum"),b=("base","sum"))
    return (g.nv/g.b-N[y]).rename(f"mr")
M={y:munr(y) for y in YRS}
def build(t,tgt=True):
    i=YRS.index(t) if t in YRS else len(YRS); 
    p=YRS[i-1]; q=YRS[i-2]
    base=fr[t] if t in YRS else fr[2024]
    d=base[["key","mkey","EDOCVE","MPIOCVE","LN","base","NS","nsr","jov","old","fem","TIPOSEC"]].copy()
    if t in YRS: d["r"]=fr[t].r.values; d["abst"]=fr[t].abst.values
    for nm,y in (("1",p),("2",q)):
        d=d.merge(fr[y][["key","r"]].rename(columns={"r":"rs"+nm}),on="key",how="left")
        d["rm"+nm]=d.mkey.map(M[y])
        d["has"+nm]=d["rs"+nm].notna()*1
        d["rs"+nm]=d["rs"+nm].fillna(d["rm"+nm])
    d["rm1"]=d.rm1.fillna(0); d["rm2"]=d.rm2.fillna(0)
    d["rs1"]=d.rs1.fillna(d.rm1); d["rs2"]=d.rs2.fillna(d.rm2)
    d["mid"]=int(t in MID); d["t"]=t
    return d
def X(d):
    return pd.DataFrame({"rs1":d.rs1,"rs2":d.rs2,"rm1":d.rm1,"rm2":d.rm2,"jov":d.jov,"old":d.old,"fem":d.fem,
        "mid":d.mid,"mid_jov":d.mid*d.jov,"mid_rm1":d.mid*d.rm1,"mid_rs1":d.mid*d.rs1,"has1":d.has1,"has2":d.has2,
        "rural":(d.TIPOSEC=="R")*1,"mixta":(d.TIPOSEC=="M")*1})
TR={t:build(t) for t in [2015,2018,2021,2024]}
def ok(d): return (d.nsr<0.5)&(d.base>=30)
def fit(years):
    D=pd.concat([TR[t] for t in years]); D=D[ok(D)]
    m1=Ridge(alpha=1.0).fit(X(D),D.r,sample_weight=D.base)
    m2=HistGradientBoostingRegressor(max_depth=4,max_iter=250,learning_rate=0.06,min_samples_leaf=100, early_stopping=False, random_state=0).fit(X(D),D.r,sample_weight=D.base)
    return m1,m2
def agg(d,c):  # municipio ponderado por base
    g=d.assign(w=d.base*d[c]).groupby("mkey").agg(w=("w","sum"),b=("base","sum"))
    return g.w/g.b, g.b
def cls(a): return pd.cut(a,[-1,0.40,0.50,2],labels=["V","A","R"])
if __name__=="__main__":
  for test,trn in [(2021,[2015,2018]),(2024,[2015,2018,2021])]:
    m1,m2=fit(trn); d=TR[test].copy()
    d["p_ridge"]=m1.predict(X(d)); d["p_gb"]=m2.predict(X(d)); d["p_pers"]=d.rs1; d["p_mun1"]=d.rm1; d["p_mun12"]=(d.rm1+d.rm2)/2
    print(f"\n== test {test} n={len(d)}")
    for c in ["p_pers","p_mun1","p_mun12","p_ridge","p_gb"]:
        pa,b=agg(d.assign(x=d[c]),"x"); aa,_=agg(d,"r")
        e=(pa-aa).abs()
        sec=d[ok(d)]; es=(sec[c]-sec.r).abs()
        print(f"{c:8s} sec MAE(pond) {(es*sec.base).sum()/sec.base.sum()*100:.2f}pp | mun MAE pond {(e*b).sum()/b.sum()*100:.2f}pp  corr {np.corrcoef(pa,aa)[0,1]:.3f}  acuerdo semáforo {((cls(pa+N[test])==cls(aa+N[test]))*b).sum()/b.sum():.3f}")
