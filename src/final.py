import numpy as np, pandas as pd
from model2 import *
import os; os.makedirs("out", exist_ok=True)
NAT27=0.482            # abstención nacional supuesta en 2027 (nivel de 2021, última intermedia)
DELTA=0.06             # intervención: baja 6 pp la abstención de 18-29 años en zonas rojas
m1,m2=fit([2015,2018,2021,2024])
d=build(2027)
d["rhat"]=(m1.predict(X(d))+m2.predict(X(d))+(d.rm1+d.rm2)/2)/3
d["rhat"]-=(d.rhat*d.base).sum()/d.base.sum()      # centrar: el promedio nacional lo fija NAT27
d["abst27"]=(NAT27+d.rhat).clip(0.02,0.98)
d=d.merge(fr[2024][["key","MPIONOM","EDONOM","DEF","LN_18_29"]],on="key")
d["nv27"]=d.abst27*d.base
def semaforo(a): return pd.cut(a,[-1,0.45,0.55,2],labels=["Verde","Amarillo","Rojo"])
def tabla(keys):
    g=d.groupby(keys).agg(EDONOM=("EDONOM","first"),LN=("LN","sum"),base=("base","sum"),nv=("nv27","sum"),
        jov=("LN_18_29","sum"),MPIONOM=("MPIONOM","first"),DEFn=("DEF","first"))
    t=g.reset_index()
    t["abst_sin"]=t.nv/t.base; t["semaforo_sin"]=semaforo(t.abst_sin)
    return t
def escenario(t,delta,share_col="jov"):
    rojo=t.semaforo_sin=="Rojo"
    t=t.copy(); ganan=np.where(rojo,delta*t.jov,0)       # votantes adicionales
    t["votos_extra"]=ganan; t["abst_con"]=(t.nv-ganan)/t.base; t["semaforo_con"]=semaforo(t.abst_con)
    return t
mun=escenario(tabla(["EDOCVE","mkey"]),DELTA)
mun["abstencionistas_esp"]=mun.nv
mun=mun.sort_values("abstencionistas_esp",ascending=False)
mun.to_csv("out/semaforo_municipal_2027.csv",index=False,encoding="utf-8-sig")
defd=escenario(tabla(["EDOCVE","DEF"]),DELTA)
defd.to_csv("out/semaforo_distrito_federal_2027.csv",index=False,encoding="utf-8-sig")
print("municipios",len(mun)); 
for nm,t in (("municipio",mun),("distrito federal",defd)):
    print("\n",nm)
    print("semáforo SIN :",t.groupby("semaforo_sin",observed=False).agg(n=("LN","size"),LN=("LN","sum")).assign(pct_LN=lambda x:(x.LN/x.LN.sum()).round(3)).to_dict("index"))
    print("semáforo CON :",t.groupby("semaforo_con",observed=False).agg(n=("LN","size"),LN=("LN","sum")).assign(pct_LN=lambda x:(x.LN/x.LN.sum()).round(3)).to_dict("index"))
    print("abst nacional sin %.4f con %.4f | votos extra %.0f (%.2f%% de LN)"%(t.nv.sum()/t.base.sum(),(t.nv.sum()-t.votos_extra.sum())/t.base.sum(),t.votos_extra.sum(),100*t.votos_extra.sum()/t.LN.sum()))
print(mun[["EDONOM","MPIONOM","LN","abst_sin","semaforo_sin","abst_con","votos_extra"]].head(10).round(3).to_string())
d[["key","EDOCVE","EDONOM","MPIOCVE","MPIONOM","DEF","LN","jov","old","fem","TIPOSEC","abst27"]].round(4).to_csv("out/riesgo_seccion_2027.csv",index=False,encoding="utf-8-sig")
