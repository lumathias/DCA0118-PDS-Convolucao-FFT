import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve


# Sobrepor e somar os resultados das convoluções
def overlap_add(x_pads, hn_pad, N, M):
    # Convolução circular
    aux = []
    for pad in x_pads:
        xn_dft = fft(pad, N)  # DFT de x[n]
        hn_dft = fft(hn_pad, N)  # DFT de  h[n]
        conv = ifft(xn_dft * hn_dft)  # Multiplicar no domínio da frequência
        aux.append(conv)
    # Sobrepor e somar
    yn = aux[0]
    for i in range(1, len(aux)):
        overlap = aux[i][:(M - 1)] + aux[i - 1][-(M - 1):]
        yn = np.concatenate((yn, overlap, aux[i][(M - 1):]))
    return yn

# Settings
N = 512  # Num de amostras da fft
M = 120  # Amostras
L = N-M+1  # Comprimento
tamx = 16*L  # Comprimento do sinal de entrada x[n]=15.3*L **Arredondado p/ 16 pois l38 n aceitava

# Gerar x[n] e h[n] aleatórios
h = np.random.randn(M)
x = np.random.randn(tamx)

# Preencher h(n) com L-1 zeros (L-1 = N–M)
hn_pad = np.pad(h, (0, (L-1)), 'constant')

# Arranjar x em L amostras
pad_size = round(tamx/L)
xn = np.split(x, pad_size)

# Preencher os blocos de x[n] com zeros
x_pads = []
for pad in xn:
    x_pad = np.pad(pad, (0, N - len(pad)), 'constant')
    x_pads.append(x_pad)

# Convolução overlap_Add
y = overlap_add(x_pads, hn_pad, N, M)

# Convolução no tempo de cada bloco
y_tempo = []
for pad in xn:
    result = convolve(pad, hn_pad, 'full')
    y_tempo.append(result)

# Convoluções no tempo concatenadas
y_t = np.concatenate(y_tempo)

# Comparar os resultados
y_dft = y[:len(y_t)]
print(y_dft)

# Graphs plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 1, 1)

plt.plot(y_t, label='Convolução no Tempo')
plt.plot(y, label='Convolução Circular')

plt.ylabel('Amplitude')
plt.xlabel('Amostras')
plt.title('Convolução Circular vs Convolução no Tempo')

plt.legend()
plt.tight_layout()
plt.show()
