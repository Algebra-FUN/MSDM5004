import numpy as np
from functools import partial
import scipy
import plotly.graph_objects as go

# %%
def Sflogplot(data,windspeed="VWRZ",d=1/4,bound=[0.1,2],fig=go.Figure(),rolling=False):
    f = np.fft.rfftfreq(len(data),d)
    if not rolling:
        S = np.abs(np.fft.rfft(data[windspeed]))
    else:
        S = np.abs(np.fft.rfft(data[windspeed].rolling(int(rolling/d),center=True,min_periods=1).mean()))
    fig.add_trace(go.Scatter(x=f[f>0], y=S[f>0], name=windspeed))
    band = (f>=bound[0]) & (f<=bound[1])
    f_ = f[band]
    S_ = S[band]
    a, b, r, p, std = scipy.stats.linregress(np.log(f_),np.log(S_))
    r2 = r**2
    if not rolling:
        title = f"ln(S)-ln(f) plot of {windspeed} (fit goodness: r^2={r2:.2f}, f∈{bound})"
    else:
        title = f"ln(S)-ln(f) plot of {windspeed} (window: w={rolling}, fit goodness: r^2={r2:.2f}, f∈{bound})"
    fig.add_trace(go.Scatter(x=f_, y=np.exp(a*np.log(f_)+b), name=f"ln(S)={a:.2f}ln(f)+{b:.2f}"))
    fig.update_layout(title=title, xaxis_title="f", yaxis_title="S")
    fig.update_layout(legend=dict(yanchor="top",y=1,xanchor="right",x=1))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return f, S

def EDR_ML_func(data,bound=[0.1,1],C_k=0.65,d=1/4,windspeed='VWRZ',airspeed='AIRSPEED'):
    f = np.fft.rfftfreq(len(data),d)
    band = (f>=bound[0]) & (f<=bound[1])
    S = np.abs(np.fft.rfft(data[windspeed],norm="forward"))
    U = np.mean(data[airspeed])
    S = S[band]
    f = f[band]
    return (2*np.pi/U)**(1/3)*np.mean(S*f**(5/3)/C_k)**.5

def EDR_ML(data,w=10,d=1/4,**kwargs):
    return np.array(list(map(partial(EDR_ML_func,d=d,**kwargs),data.rolling(int(w/d),center=True,min_periods=1))))

def EDR_NLR_func(data,bound=[0.1,1],C=1.05,d=1/4,windspeed='VWRZ',airspeed='AIRSPEED'):
    f = np.fft.rfftfreq(len(data),d)
    band = (f>=bound[0]) & (f<=bound[1]) | (f == 0)
    S = np.abs(np.fft.rfft(data[windspeed],norm="forward"))
    S[~band] = 0
    ws = np.fft.irfft(S,norm="forward")
    sigma = np.std(ws)
    U = np.mean(data[airspeed])
    return sigma*U**(-1/3)/np.sqrt(C*(bound[0]**(-2/3)-bound[1]**(-2/3)))

def EDR_NLR(data,w=10,d=1/4,**kwargs):
    return np.array(list(map(partial(EDR_NLR_func,d=d,**kwargs),data.rolling(int(w/d),center=True,min_periods=1))))

# def get_transverse_velocity(data,aircraft_v="ui,vi,wi",wind_v="uw,vw,ww"):
#     aircraft_v = aircraft_v.split(",")
#     wind_v = wind_v.split(",")
#     wind_speed2 = (data[wind_v]**2).sum(axis=1)
#     aircraft_speed2 = (data[aircraft_v]**2).sum(axis=1)
#     awdot = data.apply(lambda x: np.dot(x[aircraft_v],x[wind_v]),axis=1)
#     return (wind_speed2-awdot**2/aircraft_speed2)**.5