import numpy as np 
import pandas as pd 
import plotly 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import torch 
from tqdm import tqdm

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

def make_subplots_1(df, line_color='black', title:str=None):
    fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    traces = [
        ('x2g', 1), ('y2g', 2), ('z2g', 3), ('x50g', 4),
        ('y50g', 5), ('strain0', 6), ('strain1', 7)
    ]
    
    for name, row in traces:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df[name],
                mode='lines',
                name=name,
                line=dict(color=line_color, width=1),
                showlegend=False
            ),
            row=row,
            col=1
        )

    fig.update_layout(
        height=800,
        width=1200,
        title_text=title,
        annotations=[
            dict(text=name, x=0.5, y=1 - (i * 0.14), xref="paper", yref="paper", showarrow=False)
            for i, (name, _) in enumerate(traces)
        ]
    )
    
    return fig

def make_subplots_2(df, line_color='black', fig=None, title:str=None, mode='lines'):
    if fig is None:
        fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    traces = [
        ('x2g', 1), ('y2g', 2), ('z2g', 3), ('x50g', 4),
        ('y50g', 5), ('strain0', 6), ('strain1', 7)
    ]
    
    for name, row in traces:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df[name],
                mode=mode,
                name=name,
                line=dict(color=line_color, width=1) if mode == 'lines' else None,
                marker=dict(color=line_color) if mode != 'lines' else None,
                showlegend=False
            ),
            row=row,
            col=1
        )

    fig.update_layout(
        height=800,
        width=1200,
        title_text=title,
        annotations=[
            dict(text=name, x=0.5, y=1 - (i * 0.14), xref="paper", yref="paper", showarrow=False)
            for i, (name, _) in enumerate(traces)
        ]
    )
    
    return fig

def make_subplots_with_zones(df, line_color='black', fig=None, title:str=None, mode='lines', zones=None):
    """
    Creates subplots and marks anomaly zones with shaded regions.
    
    Parameters:
    - df: DataFrame containing the time series data with 'timestamp' as x-axis.
    - line_color: Color for the time series line.
    - fig: Optional Plotly figure to add traces to.
    - title: Title for the plot.
    - mode: Mode for the plot ('lines' or 'markers').
    - zones: List of (zone_start, zone_end) tuples that indicate anomaly zones.
    
    Returns:
    - fig: Plotly figure object with time series data and anomaly zones marked.
    """
    if fig is None:
        fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    traces = [
        ('x2g', 1), ('y2g', 2), ('z2g', 3), ('x50g', 4),
        ('y50g', 5), ('strain0', 6), ('strain1', 7)
    ]
    
    # Add traces for the time series data
    for name, row in traces:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df[name],
                mode=mode,
                name=name,
                line=dict(color=line_color, width=1) if mode == 'lines' else None,
                marker=dict(color=line_color) if mode != 'lines' else None,
                showlegend=False
            ),
            row=row,
            col=1
        )
    
    # Mark anomaly zones by shading the regions
    if zones:
        for zone_start, zone_end in zones:
            for _, row in traces:
                fig.add_shape(
                    type="rect",
                    xref="x",  # X-axis refers to the shared 'timestamp'
                    yref="paper",  # Y-axis spans all rows
                    x0=df['timestamp'].iloc[zone_start],  # Start of zone
                    x1=df['timestamp'].iloc[zone_end],    # End of zone
                    y0=0,  # From the bottom of the subplot
                    y1=1,  # To the top of the subplot (row-wise)
                    fillcolor="rgba(255, 0, 0, 0.2)",  # Red with transparency
                    line=dict(width=0),
                    row=row,
                    col=1
                )
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text=title,
        annotations=[
            dict(text=name, x=0.5, y=1 - (i * 0.14), xref="paper", yref="paper", showarrow=False)
            for i, (name, _) in enumerate(traces)
        ]
    )
    
    return fig

def preprocess_data(df): 
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame()
    for col in df.processed:
        if col == 'timestamp': 
            df_scaled[col] = df[col].values
            continue
        else: 
            value = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
            df_scaled[col] = value
    return df_scaled

def split_train_val_test_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1): 
    N = len(df)
    N_train = int(N * train_ratio)
    N_val = int(N * val_ratio)
    N_test = N - N_train - N_val
    df_train = df.iloc[:N_train, :]
    df_val = df.iloc[N_train:N_train+N_val, :]
    df_test = df.iloc[N_train+N_val:, :]
    return df_train, df_val, df_test

def create_sequences(data, sequence_length): 
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length]
        sequences.append(sequence)
    sequences = np.array(sequences)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
    tensor_dataset = torch.utils.data.TensorDataset(sequences_tensor)
    return tensor_dataset

def train_the_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=100): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs)): 
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader): 
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad(): 
            for i, data in enumerate(val_loader): 
                inputs = data[0]
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss}, Val     Loss: {val_loss}")
    test_loss = 0.0
    with torch.no_grad(): 
        for i, data in enumerate(test_loader): 
            inputs = data[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()
        test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {test_loss}")
    return model, train_losses, val_losses