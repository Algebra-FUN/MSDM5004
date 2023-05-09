#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from functools import partial

pio.templates.default = "plotly_white"

# %%
data1 = pd.read_csv("data/1 QAR/data.csv").set_index("RUNTIME")
data1

# %%
def Sflogplot(data,windspeed="VWRZ",d=1/4,bound=[0.1,2],fig=go.Figure()):
    f = np.fft.rfftfreq(len(data),d)
    S = np.abs(np.fft.rfft(data[windspeed]))
    fig.add_trace(go.Scatter(x=np.log(f[f>0]), y=np.log(S[f>0]), name=windspeed))
    band = (f>=bound[0]) & (f<=bound[1])
    f_ = f[band]
    S_ = S[band]
    a,b = np.polyfit(np.log(f_),np.log(S_),1)
    fig.add_trace(go.Scatter(x=np.log(f_), y=a*np.log(f_)+b, name=f"log(S)={a:.2f}log(f)+{b:.2f}"))
    fig.update_layout(title=f"log(S)-log(f) plot of {windspeed}", xaxis_title="log(f)", yaxis_title="log(S)")
    return f, S

# %%
fig = go.Figure()
f, S = Sflogplot(data1,windspeed="VWRZ",fig=fig)
fig.show()

# %%
def EDR_MLE_func(data,bound=[0.1,2],C_k=0.65,d=1/4,windspeed='VWRZ',airspeed='AIRSPEED'):
    f = np.fft.rfftfreq(len(data),d)
    band = (f>=bound[0]) & (f<=bound[1])
    S = np.abs(np.fft.rfft(data[windspeed],norm="forward"))
    U = np.mean(data[airspeed])
    S = S[band]
    f = f[band]
    return (2*np.pi/U)**(1/3)*np.mean(S*f**(5/3)/C_k)**.5

def EDR_MLE(data,w=10,d=1/4,**kwargs):
    return list(map(partial(EDR_MLE_func,d=d,**kwargs),data.rolling(int(w/d),center=True,min_periods=1)))

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=data1.index,y=data1.EDR,name="Truth"))
for w in [5,10,20]:
    edr_pred = EDR_MLE(data1,w=w)
    fig.add_trace(go.Scatter(x=data1.index,y=edr_pred,name=f"Estimated(w={w})"))
fig.update_layout(title=f"EDR estimation by MLE (f∈[0.1,2])",xaxis_title="Time",yaxis_title="EDR")
fig.show()

# %%
def EDR_NLR_func(data,bound=[0.1,2],C=1.05,d=1/4,windspeed='VWRZ',airspeed='AIRSPEED'):
    f = np.fft.rfftfreq(len(data),d)
    band = (f>=bound[0]) & (f<=bound[1])
    S = np.abs(np.fft.rfft(data[windspeed],norm="forward"))
    S[~band] = 0
    ws = np.fft.irfft(S,norm="forward")
    sigma = np.std(ws)
    U = np.mean(data[airspeed])
    return sigma*U**(-1/3)/np.sqrt(C*(bound[0]**(-2/3)-bound[1]**(-2/3)))

def EDR_NLR(data,w=10,d=1/4,**kwargs):
    return list(map(partial(EDR_NLR_func,d=d,**kwargs),data.rolling(int(w/d),center=True,min_periods=1)))

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=data1.index,y=data1.EDR,name="Truth"))
for w in [10,12,20]:
    edr_pred = EDR_NLR(data1,w=w)
    fig.add_trace(go.Scatter(x=data1.index,y=edr_pred,name=f"Estimated(w={w})"))
fig.update_layout(title=f"EDR estimation by NLR (f∈[0.1,2])",xaxis_title="Time",yaxis_title="EDR")
fig.show()

# %%
def get_transverse_velocity(data,aircraft_v="ui,vi,wi",wind_v="uw,vw,ww"):
    aircraft_v = aircraft_v.split(",")
    wind_v = wind_v.split(",")
    wind_speed2 = (data[wind_v]**2).sum(axis=1)
    aircraft_speed2 = (data[aircraft_v]**2).sum(axis=1)
    awdot = data.apply(lambda x: np.dot(x[aircraft_v],x[wind_v]),axis=1)
    return (wind_speed2-awdot**2/aircraft_speed2)**.5

# %%
data2 = pd.read_csv("data/2 PROBE/data.csv")
data2['tw'] = get_transverse_velocity(data2)

# %%
fig = go.Figure()
f, S = Sflogplot(data2,d=1,windspeed="tw",fig=fig)
fig.show()

# %%
fig = go.Figure()
bound = [0.1,0.5]
for w in [10]:
    edr_pred = EDR_MLE(data2,w=w,windspeed='tw',airspeed='tas',d=1,bound=bound)
    fig.add_trace(go.Scatter(x=data2.index,y=edr_pred,name=f"Estimated(w={w})"))
fig.update_layout(title=f"EDR estimation by MLE (f∈{bound})",xaxis_title="Time",yaxis_title="EDR")
fig.show()

# %%
fig = go.Figure()
bound = [0.1,0.5]
for w in [10]:
    edr_pred = EDR_NLR(data2,w=w,windspeed='tw',airspeed='tas',d=1,bound=bound)
    fig.add_trace(go.Scatter(x=data2.index,y=edr_pred,name=f"Estimated(w={w})"))
fig.update_layout(title=f"EDR estimation by NLR (f∈{bound})",xaxis_title="Time",yaxis_title="EDR")
fig.show()

# %%
