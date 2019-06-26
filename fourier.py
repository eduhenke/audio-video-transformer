import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sounddevice as sd


def record(fs=44100, duration=0.01):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording

step = 0.1
fs = 44100
t_min, t_max = 0, step
f_max, f_step = 800, 40

t = np.linspace(t_min, t_max, step*fs)
rotation = lambda f, t : np.exp(-2*np.pi*1j*f*t)
wind = lambda f, rec: np.array([a*b for a, b in zip(rec, rotation(f, t))])
get_center = lambda w: sum(w) / len(w)

X, Y = [], []
fig = plt.figure()
line, = plt.plot(X, Y, '-')
axes = plt.axes()
axes.set_xlim(left=0, right=f_max)
axes.set_ylim(bottom=-1, top=1)

def update(frame):
    recording = record(fs, step) * 10
    recording = [p[0] for p in recording]

    V = max(np.linalg.norm(recording), 1)

    f = np.arange(0.01, f_max, f_step)
    
    X = f
    Y = np.array([abs(get_center(wind(freq, recording))) for freq in f])
    line.set_data(X, Y)
    max_i = np.argmax(Y)
    max_conf = Y[max_i]
    print(max_i * f_step * max_conf)

    return line,

animation = FuncAnimation(fig, update, interval=20)

plt.show()




# X, Y = [], []
# fig = plt.figure()
# line, = plt.plot(X, Y, '-')
# axes = plt.axes()
# axes.set_xlim(left=-1, right=1)
# axes.set_ylim(bottom=-1, top=1)

# f = 0
# point = plt.scatter(0, 0, color='red')
# def update(frame):
#     global f, point
#     winded = wind(f)
#     X = [c.real for c in winded]
#     Y = [c.imag for c in winded]
#     line.set_data(X, Y)
#     print(f)

#     center = get_center(winded)
#     point.remove()
#     point = plt.scatter(center.real, center.imag, color='red')

#     f += 0.01
#     return line,

# animation = FuncAnimation(fig, update, interval=20)

# plt.show()

