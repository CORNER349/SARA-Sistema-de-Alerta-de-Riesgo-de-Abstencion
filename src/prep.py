import pandas as pd, numpy as np
A="./"
YRS=[2009,2012,2015,2018,2021,2024]
NUM=["LN","SV","NV","NS"]+[f"{c}_{g}" for g in ["18_29","30_44","45_59","60mas","H","M"] for c in ["LN","NV","NS"]]
def load():
    out={}
    for y in YRS:
        d=pd.read_csv(f"{A}agg_{y}.csv")
        d["key"]=d.EDOCVE.astype(str).str.zfill(2)+"-"+d.SECCION.astype(str).str.zfill(4)
        meta=d.sort_values("LN",ascending=False).drop_duplicates("key")[["key","EDOCVE","EDONOM","MPIOCVE","MPIONOM","TIPOSEC","DEF","DEL"]]
        s=d.groupby("key")[NUM].sum().reset_index().merge(meta,on="key")
        s["base"]=(s.LN-s.NS).clip(lower=1)
        s["abst"]=s.NV/s.base
        s["jov"]=s.LN_18_29/s.LN.clip(lower=1)
        s["old"]=s.LN_60mas/s.LN.clip(lower=1)
        s["fem"]=s.LN_M/s.LN.clip(lower=1)
        out[y]=s
    return out
def nat(fr):
    return {y:fr[y].NV.sum()/fr[y].base.sum() for y in YRS}
