from flask import Flask, request, render_template
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoded, _ = self.decoder(hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2))
        return decoded

def generate_sine_wave(seq_len, input_dim, batch_size):
    t = np.linspace(0, 2 * np.pi, seq_len)
    sine_wave = np.sin(t)
    input_signal = np.stack([sine_wave + np.random.normal(0, 0.1, seq_len) for _ in range(input_dim)], axis=1)
    input_signal = np.tile(input_signal, (batch_size, 1, 1))
    return torch.tensor(input_signal, dtype=torch.float32)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        seq_len = int(request.form['seq_len'])
        input_dim = int(request.form['input_dim'])
        hidden_dim = int(request.form['hidden_dim'])
        num_layers = int(request.form['num_layers'])
        batch_size = int(request.form['batch_size'])

        input_tensor = generate_sine_wave(seq_len, input_dim, batch_size)
        model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        output_tensor = model(input_tensor)
        latent_features = model.encoder(input_tensor).squeeze().detach().numpy()

        plt.figure(figsize=(12, 6))
        plt.plot(np.linspace(0, 2 * np.pi, seq_len), input_tensor[0,:,0], label='Original Signal')
        plt.plot(np.linspace(0, 2 * np.pi, seq_len), output_tensor.detach().numpy()[0,:,0], label='Reconstructed Signal')
        plt.legend()
        plt.title('Original and Reconstructed Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', plot_url=plot_url, latent_features=latent_features)

    return render_template('index.html', plot_url=None, latent_features=None)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2', port=8123)
