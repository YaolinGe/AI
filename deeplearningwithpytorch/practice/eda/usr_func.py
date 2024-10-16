import numpy as np 
import pandas as pd 
import plotly 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

def reconstruct_strain(df_strain): 
    strain0 = []
    strain1 = []
    ind_0 = np.where(df_strain.iloc[:, 1] == 0)[0]
    ind_1 = np.where(df_strain.iloc[:, 1] == 1)[0]
    data_0 = df_strain.iloc[ind_0, :]
    data_1 = df_strain.iloc[ind_1, :]
    N_cols = data_0.shape[1] - 2
    dt_0 = np.diff(data_0.iloc[:, 0]) / N_cols
    dt_1 = np.diff(data_1.iloc[:, 0]) / N_cols
    dt_0 = np.append(dt_0, dt_0[-1])
    dt_1 = np.append(dt_1, dt_1[-1])
    for i in range(len(data_0)): 
        for j in range(data_0.shape[1] - 2): 
            strain0.append([data_0.iloc[i, 0] + j*dt_0[i], data_0.iloc[i, j+2]])
    for i in range(len(data_1)):
        for j in range(data_1.shape[1] - 2): 
            strain1.append([data_1.iloc[i, 0] + j*dt_1[i], data_1.iloc[i, j+2]])
    strain0 = pd.DataFrame(strain0, columns=['timestamp', 'value'])
    strain1 = pd.DataFrame(strain1, columns=['timestamp', 'value'])
    strain0['timestamp'] = strain0['timestamp'] - strain0['timestamp'].iloc[0]
    strain1['timestamp'] = strain1['timestamp'] - strain1['timestamp'].iloc[0]
    strain0['timestamp'] = strain0['timestamp'] / 1000
    strain1['timestamp'] = strain1['timestamp'] / 1000
    return strain0, strain1

def read_file(filepath): 
    df_accelerometer = pd.read_csv(filepath + "_box1.csv", header=None, sep=";", names=['timestamp', 'x2g', 'y2g', 'z2g', 'x50g', 'y50g'])
    df_accelerometer['timestamp'] = (df_accelerometer['timestamp'] - df_accelerometer['timestamp'][0]) / 1000
    df_strain = pd.read_csv(filepath + "_box2.csv", header=None, sep=";")
    df_strain0, df_strain1 = reconstruct_strain(df_strain)
    return df_accelerometer, df_strain0, df_strain1

def interpolate_data(t, t_old, data_old):
    values_interpolated = np.interp(t, t_old, data_old)
    df = pd.DataFrame({'timestamp': t, 'value': values_interpolated})
    return df

def synchronize_data(df_accelerometer, df_strain0, df_strain1):
    t_min = min(df_accelerometer['timestamp'].iloc[0], df_strain0['timestamp'].iloc[0], df_strain1['timestamp'].iloc[0])
    t_max = max(df_accelerometer['timestamp'].iloc[-1], df_strain0['timestamp'].iloc[-1], df_strain1['timestamp'].iloc[-1])
    N_max = max(len(df_accelerometer), len(df_strain0), len(df_strain1))
    t = np.linspace(t_min, t_max, N_max)

    df_accelerometer_x2g_interp = interpolate_data(t, df_accelerometer['timestamp'], df_accelerometer['x2g'])
    df_accelerometer_y2g_interp = interpolate_data(t, df_accelerometer['timestamp'], df_accelerometer['y2g'])
    df_accelerometer_z2g_interp = interpolate_data(t, df_accelerometer['timestamp'], df_accelerometer['z2g'])
    df_accelerometer_x50g_interp = interpolate_data(t, df_accelerometer['timestamp'], df_accelerometer['x50g'])
    df_accelerometer_y50g_interp = interpolate_data(t, df_accelerometer['timestamp'], df_accelerometer['y50g'])
    df_strain0_interp = interpolate_data(t, df_strain0['timestamp'], df_strain0['value'])
    df_strain1_interp = interpolate_data(t, df_strain1['timestamp'], df_strain1['value'])

    df_sync = pd.DataFrame({'timestamp': t, 'x2g': df_accelerometer_x2g_interp['value'], 'y2g': df_accelerometer_y2g_interp['value'], 'z2g': df_accelerometer_z2g_interp['value'], 'x50g': df_accelerometer_x50g_interp['value'], 'y50g': df_accelerometer_y50g_interp['value'], 'strain0': df_strain0_interp['value'], 'strain1': df_strain1_interp['value']})
    return df_sync

def make_subplot_data(df_sync, filepath, figpath, line_color='black', show:bool=False, save:bool=False):
    fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    
    # Add traces
    fig.add_trace(go.Scatter(x=df_sync['timestamp'], y=df_sync['x2g'], mode='lines', name='x2g', line=dict(color=line_color, width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_sync['timestamp'], y=df_sync['y2g'], mode='lines', name='y2g', line=dict(color=line_color, width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_sync['timestamp'], y=df_sync['z2g'], mode='lines', name='z2g', line=dict(color=line_color, width=1), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_sync['timestamp'], y=df_sync['x50g'], mode='lines', name='x50g', line=dict(color=line_color, width=1), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_sync['timestamp'], y=df_sync['y50g'], mode='lines', name='y50g', line=dict(color=line_color, width=1), showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=df_sync['timestamp'], y=df_sync['strain0'], mode='lines', name='strain0', line=dict(color=line_color, width=1), showlegend=False), row=6, col=1)
    fig.add_trace(go.Scatter(x=df_sync['timestamp'], y=df_sync['strain1'], mode='lines', name='strain1', line=dict(color=line_color, width=1), showlegend=False), row=7, col=1)

    fig.update_layout(
        height=800, 
        width=1200, 
        title_text=filepath,
        annotations=[
            dict(text="x2g", x=0.5, y=0.99, xref="paper", yref="paper", showarrow=False),
            dict(text="y2g", x=0.5, y=0.89, xref="paper", yref="paper", showarrow=False),
            dict(text="z2g", x=0.5, y=0.75, xref="paper", yref="paper", showarrow=False),
            dict(text="x50g", x=0.5, y=0.6, xref="paper", yref="paper", showarrow=False),
            dict(text="y50g", x=0.5, y=0.45, xref="paper", yref="paper", showarrow=False),
            dict(text="strain0", x=0.5, y=0.25, xref="paper", yref="paper", showarrow=False),
            dict(text="strain1", x=0.5, y=0.11, xref="paper", yref="paper", showarrow=False)
        ]
    )

    file = filepath.split("\\")[-1]
    if show:
        fig.show()
    if save: 
        fig.write_html(figpath + f"\\{file}.html", auto_open=False)

def preprocess_data(df): 
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame()
    for col in df.columns:
        if col == 'timestamp': 
            df_scaled[col] = df[col].values
            continue
        else: 
            value = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
            df_scaled[col] = value
    return df_scaled

