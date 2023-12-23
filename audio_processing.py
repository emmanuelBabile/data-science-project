import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import datetime

# Function to process uploaded audio
def process_audio(audio):
    # Perform audio processing here
    st.audio(audio, format='audio/wav')

# ------------- shape calculation:
def get_audio_shape(data):
    if data is not None and len(data.shape) > 1:
        # Check if the audio has multiple channels (stereo)
        if data.shape[1] == 2:
            # Convert stereo to mono by taking the first channel
            data = np.array([data[i][0] for i in range(len(data))])

    return data.shape

# ------------- Plot based on recording time:
def get_time(data, fs):
    nsamples = len(data)
    ns = nsamples / fs
    return ns

def plot_by_time(data, fs):
    # Calculate the total duration
    ns = get_time(data, fs)
    Ts = str(datetime.timedelta(minutes=ns))

    # Create a Streamlit figure
    fig, ax = plt.subplots()

    # Create time values
    x = np.linspace(0, ns, len(data))

    # Plot the data against time
    ax.plot(x, data)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Plot by Time')
    ax.grid(True)

    # Display the total duration
    st.write(f"Total Duration: {Ts}")

    # Display the figure using Streamlit
    st.pyplot(fig)

# ------------- Plot of the first 20 seconds
def plot_first_20_seconds(data,dt, fs):
    ns = get_time(data, fs)
    Ts = str(datetime.timedelta(minutes=ns))

    # Calculate the time values for the extracted segment
    x = np.linspace(0, 20, len(dt))

    # Plot the data against time
    fig, ax = plt.subplots()
    ax.plot(x, dt)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Plot First 20 Seconds')
    ax.grid(True)

    # Display the total duration
    st.write(f"Total Duration: {Ts}")

    # Display the figure using Streamlit
    st.pyplot(fig)

# ------------- envelope detection:
def detect_envelope(env):

    # Create a Streamlit figure for the envelope plot
    fig, ax = plt.subplots()
    ax.plot(env)
    ax.hlines(170, 0, 80, colors='green')
    ax.set_xlabel('Time (chunks)')
    ax.set_ylabel('Envelope Value')
    ax.set_title('Envelope Plot')

    # Display the figure using Streamlit
    st.pyplot(fig)
    print("Envelope:", env)

# ------------- signal smoothing:
def smooth_signal(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# ---------- calculation of cycle time and between each respiratory cycle:
def calculate_respiration_cycles(sm_env, threshold):
    crossed = False
    xs = []
    for i, value in enumerate(sm_env):
        if value < threshold and not crossed:
            xs.append(i)
            crossed = True
        elif value >= threshold and crossed:
            crossed = False
            xs.append(i)
    return xs
