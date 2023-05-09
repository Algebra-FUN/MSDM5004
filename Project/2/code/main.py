#%%
from EDRCompute import *
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

# %%
data1 = pd.read_csv("../data/1 QAR/data.csv").set_index("RUNTIME")
data1

# %%
fig = go.Figure()
f, S = Sflogplot(data1,windspeed="VWRZ",fig=fig)
fig.update_layout(width=800, height=600)
fig.show()
fig.write_image("../img/qar_loglog.pdf")

# %% 
for w in [10,20,50,120]:
    fig = go.Figure()
    f, S = Sflogplot(data1,windspeed="VWRZ",fig=fig,bound=[0.1,1],rolling=w)
    fig.update_layout(width=600, height=450)
    fig.write_image(f"../img/qar_loglog(w={w}).pdf")

# %%
bound = [0.1,1]
for w in [5,10,20]:
    fig = go.Figure()
    edr_gt = data1.EDR
    fig.add_trace(go.Scatter(x=data1.index,y=edr_gt,name="Truth"))
    edr_pred = EDR_ML(data1,w=w,bound=bound)
    mse = np.mean((edr_gt-edr_pred)**2)
    fig.add_trace(go.Scatter(x=data1.index,y=edr_pred,name=f"Estimated(w={w})"))
    title =f"EDR estimation by MLE (f∈{bound}, w={w}, mse={mse:.4f})"
    fig.update_layout(title=title,xaxis_title="Time",yaxis_title="EDR")
    fig.update_layout(width=800, height=500)
    fig.update_layout(legend=dict(yanchor="top",y=1,xanchor="right",x=1))
    fig.write_image(f"../img/qar_edr(w={w}).pdf")

# %%
bound = [0.1,1]
fig = go.Figure()
fig.add_trace(go.Scatter(x=data1.index,y=data1.EDR,name="Truth"))
for w in [5,10,20]:
    edr_pred = EDR_ML(data1,w=w,bound=bound)
    fig.add_trace(go.Scatter(x=data1.index,y=edr_pred,name=f"Estimated(w={w})"))
fig.update_layout(title=f"EDR estimation by MLE (f∈{bound})",xaxis_title="Time",yaxis_title="EDR")
fig.show()
# %%
bound = [0.1,1]
for w in [5,10,20]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data1.index,y=data1.EDR,name="Truth"))
    edr_pred = EDR_NLR(data1,w=w,bound=bound)
    mse = np.mean((data1.EDR-edr_pred)**2)
    title =f"EDR estimation by NLR (f∈{bound}, w={w}, mse={mse:.4f})"
    fig.add_trace(go.Scatter(x=data1.index,y=edr_pred,name=f"Estimated(w={w})"))
    fig.update_layout(title=title,xaxis_title="Time",yaxis_title="EDR")
    fig.update_layout(width=800, height=500)
    fig.update_layout(legend=dict(yanchor="top",y=1,xanchor="right",x=1))
    fig.write_image(f"../img/qar_edr_nlr(w={w}).pdf")

# %%
fig = go.Figure()
bound = [0.1,1]
fig.add_trace(go.Scatter(x=data1.index,y=data1.EDR,name="Truth"))
for w in [10,12,20]:
    edr_pred = EDR_NLR(data1,w=w,bound=bound)
    fig.add_trace(go.Scatter(x=data1.index,y=edr_pred,name=f"Estimated(w={w})"))
fig.update_layout(title=f"EDR estimation by NLR (f∈{bound})",xaxis_title="Time",yaxis_title="EDR")
fig.show()

# %%
data2 = pd.read_csv("../data/2 PROBE/data.csv")
data2['tw'] = get_transverse_velocity(data2)

# %%
fig = go.Figure()
f, S = Sflogplot(data2,d=1,windspeed="tw",fig=fig)
fig.update_layout(width=800, height=500)
fig.write_image("../img/probe_loglog.pdf")
fig.show()

# %% 
bound = [0.1,1]
for w in [5,10,20]:
    fig = go.Figure()
    edr_pred = EDR_ML(data2,w=w,windspeed='tw',airspeed='tas',d=1,bound=bound)
    fig.add_trace(go.Scatter(x=data2.index,y=edr_pred,name=f"Estimated(ML)"))
    edr_pred = EDR_NLR(data2,w=w,windspeed='tw',airspeed='tas',d=1,bound=bound)
    fig.add_trace(go.Scatter(x=data2.index,y=edr_pred,name=f"Estimated(NLR)"))
    title =f"EDR estimation by MLE (f∈{bound}, w={w})"
    fig.update_layout(title=title,xaxis_title="Time",yaxis_title="EDR")
    fig.update_layout(width=800, height=500)
    fig.update_layout(legend=dict(yanchor="top",y=1,xanchor="left",x=0))
    fig.write_image(f"../img/probe_edr_t(w={w}).pdf")

# %% 
bound = [0.1,1]
for w in [5,10,20]:
    fig = go.Figure()
    edr_pred = EDR_ML(data2,w=w,windspeed='ww',airspeed='tas',d=1,bound=bound)
    fig.add_trace(go.Scatter(x=data2.index,y=edr_pred,name=f"Estimated(ML)"))
    edr_pred = EDR_NLR(data2,w=w,windspeed='ww',airspeed='tas',d=1,bound=bound)
    fig.add_trace(go.Scatter(x=data2.index,y=edr_pred,name=f"Estimated(NLR)"))
    title =f"EDR estimation by MLE (f∈{bound}, w={w})"
    fig.update_layout(title=title,xaxis_title="Time",yaxis_title="EDR")
    fig.update_layout(width=800, height=500)
    fig.update_layout(legend=dict(yanchor="top",y=1,xanchor="left",x=0))
    fig.write_image(f"../img/probe_edr(w={w}).pdf")

# %% Plot Transverse Velocity
fig = go.Figure()
fig.add_trace(go.Scatter(x=data2.index,y=data2.tw))
fig.update_layout(title="Transverse Velocity",xaxis_title="Time",yaxis_title="Altitude")
fig.update_layout(width=800, height=500)
fig.write_image("../img/tw.pdf")

# %%
fig = go.Figure()
bound = [0.1,0.5]
for w in [10]:
    edr_pred = EDR_ML(data2,w=w,windspeed='tw',airspeed='tas',d=1,bound=bound)
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