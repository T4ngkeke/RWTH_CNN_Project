import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import scipy.io as sio

# Load the CSV file using numpy
file_path = 'Data\mimic_perform_af_csv\mimic_perform_af_006_data.csv' # 5, 6, 17
# file_path = 'Data\mimic_perform_non_af_csv\mimic_perform_non_af_001_data.csv'
# file_path = 'ECG_Classification/training/train_ecg_00012.mat' # lengths: 10s-60s # interesting: 11, 12, 


if file_path.endswith('.csv'):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
elif file_path.endswith('.mat'):
    mat_data = sio.loadmat(file_path)
    ecg_data = mat_data['val'][0, :].flatten()
    
    sampling_rate = 300  # Hz
    time = np.arange(len(ecg_data)) / sampling_rate # generate time vector
    ppg_data = np.zeros(len(ecg_data)) # generate ppg data (pretend it is zeros)
    data = np.column_stack((time, ppg_data, ecg_data))

static_plot = True # static plot or animated plot

if static_plot:
    start_time = 0 
    duration = 1200 # number of seconds to show
    sample_rate = 125 # Hz

    # print(f"Anzahl der Daten {len(data[:,0])/sample_rate}") # 20 minute recording

    data_start = round(start_time*sample_rate)
    data_end = round(start_time*sample_rate + duration*sample_rate)

    # Extract the columns
    time = data[data_start:data_end, 0]  # Timestamp
    ppg = data[data_start:data_end, 1]   # PPG signal
    ecg = data[data_start:data_end, 2]   # ECG signal

    # standardize
    # ecg -= np.mean(ecg) # standardized to zero mean
    # ecg /= np.std(ecg) # standardized to unit variance
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # Plot the ECG signal
    ax1.plot(time, ecg, color='b')
    ax1.set_title('ECG Signal')
    ax1.set_ylabel('Amplitude')
    ax1.grid()

    # Plot the PPG signal
    ax2.plot(time, ppg, color='r')
    ax2.set_title('PPG Signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

else:
    # Extract the columns
    time = data[:, 0]  # Timestamp
    ppg = data[:, 1]   # PPG signal
    ecg = data[:, 2]   # ECG signal

    # Parameters for animation
    window_size = 10  # seconds
    play_speed = 0.2  # speed factor
    is_paused = False
    current_frame = 0

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    fig.suptitle(file_path)

    # Initial plot setup
    line_ecg, = ax1.plot([], [], color='b')
    line_ppg, = ax2.plot([], [], color='r')

    ax1.set_title('ECG Signal')
    ax1.set_ylabel('Amplitude')
    ax1.grid()

    ax2.set_title('PPG Signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid()

    # Animation function
    def init():
        ax1.set_xlim(0, window_size)
        ax2.set_xlim(0, window_size)
        ax1.set_ylim(np.min(ecg), np.max(ecg))
        ax2.set_ylim(np.min(ppg), np.max(ppg))
        return line_ecg, line_ppg

    def update(frame):
        global current_frame
        if not is_paused:
            current_frame = frame
            current_time = current_frame * play_speed
            start_idx = np.searchsorted(time, current_time)
            end_idx = np.searchsorted(time, current_time + window_size)

            line_ecg.set_data(time[start_idx:end_idx], ecg[start_idx:end_idx])
            line_ppg.set_data(time[start_idx:end_idx], ppg[start_idx:end_idx])

            ax1.set_xlim(current_time, current_time + window_size)
            ax2.set_xlim(current_time, current_time + window_size)
            #ax2.set_xlabel(f'Time (s): {current_time:.2f} - {current_time + window_size:.2f}')
            ax2.set_xticks(np.linspace(current_time, current_time + window_size, num=6))
            
        return line_ecg, line_ppg

    # Button and slider controls
    def toggle_pause(event=None):
        global is_paused
        is_paused = not is_paused

    def update_speed(val):
        global play_speed
        play_speed = val
        
    def move_forward():
        global current_frame
        current_frame += int(1 / play_speed)  # Move forward by 1 second
        update(current_frame)

    def move_backward():
        global current_frame
        current_frame -= int(1 / play_speed)  # Move backward by 1 second
        if current_frame < 0:
            current_frame = 0
        update(current_frame)

    # Adding buttons and slider
    ax_pause = plt.axes([0.81, 0.01, 0.1, 0.05])
    btn_pause = Button(ax_pause, 'Pause/Play')
    btn_pause.on_clicked(toggle_pause)

    ax_speed = plt.axes([0.1, 0.01, 0.65, 0.03])
    slider_speed = Slider(ax_speed, 'Speed', 0.1, 3.0, valinit=0.2)
    slider_speed.on_changed(update_speed)

    # Key press event handling
    def on_key(event):
        if event.key == ' ':
            toggle_pause()
        elif event.key == 'right':
            move_forward()
        elif event.key == 'left':
            move_backward()

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Animation
    ani = FuncAnimation(fig, update, frames=np.arange(0, len(time)), init_func=init, blit=True, interval=50)

    plt.tight_layout()
    plt.show()