import numpy as np
import sounddevice as sd
import cv2
import matplotlib.pyplot as plt

def read_frame(cap):
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        yield frame

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


cap = cv2.VideoCapture(0)

for frame in read_frame(cap):
    recording = record()
    V = np.linalg.norm(recording)
    # recording = record(fs, step) * 10
    # recording = [p[0] for p in recording]

    V = max(np.linalg.norm(recording), 1)

    # f = np.arange(0.01, f_max, f_step)
    
    # X = f
    # Y = np.array([abs(get_center(wind(freq, recording))) for freq in f])
    # max_i = np.argmax(Y)
    # max_conf = Y[max_i]
    # print(f_step*max_i*max_conf, f_step*max_i, max_conf)

    # print("|" * int(V*10))

    B, G, R = cv2.split(frame)
    B = np.array([b*V*10 for b in B], dtype="uint8")
    G = np.array([g*V*5 for g in G], dtype="uint8")
    frame = cv2.merge([B, G, R])
    # frame = cv2.blur(frame, (max(int(10*V), 1), 1))

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
