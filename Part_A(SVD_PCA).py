import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Function to load audio files from a directory
def load_audio_files_from_directory(directory):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            audio_data, _ = librosa.load(file_path, sr=8000)
            audio_files.append(audio_data)
    return audio_files


# Main function to load audio files from all directories
def load_audio_data(root_directory):
    audio_data_matrix = []
    for digit in range(10):
        digit_directory = os.path.join(root_directory, str(digit))
        digit_audio_files = load_audio_files_from_directory(digit_directory)
        audio_data_matrix.append(digit_audio_files)
    return audio_data_matrix


# Function to find the max number of samples in the audio data matrix
def find_max_samples(audio_data_matrix):
    max_samples = 0
    for i in range(10):
        for j in range(100):
            if len(audio_data_matrix[i][j]) > max_samples:
                max_samples = len(audio_data_matrix[i][j])
    return max_samples


# Function to find the which audio file has the max amount of samples
def find_max_samples_position(audio_data_matrix):
    digit_max = 0
    audio_file_num = 0
    max_samples = 0
    for i in range(10):
        for j in range(100):
            if len(audio_data_matrix[i][j]) > max_samples:
                max_samples = len(audio_data_matrix[i][j])
                digit_max = i
                audio_file_num = j
    return digit_max, audio_file_num


# Function to append zeros to all the audio files for them to be the same size
def append_zeros_to_all_audio_files(audio_data_matrix):
    for i in range(10):
        for j in range(100):
            if len(audio_data_matrix[i][j]) < max_samples:
                audio_data_matrix[i][j] = np.append(audio_data_matrix[i][j],
                                                    [0] * (max_samples - len(audio_data_matrix[i][j])))
    return audio_data_matrix


# Function to remove amplitudes that are not within a certain threshold
def remove_low_amplitude_segments(audio_data_matrix, threshold=0.02):
    trimmed_audio_data_matrix = []
    for digit_audio_files in audio_data_matrix:
        trimmed_digit_audio_files = []
        for audio_data in digit_audio_files:
            # Find segments with amplitude above the threshold
            non_zero_indices = np.where(np.abs(audio_data) > threshold)[0]

            # Get the start and end indices of non-zero segments
            if len(non_zero_indices) > 0:
                start_idx = non_zero_indices[0]
                end_idx = non_zero_indices[-1]

                # Extract the non-zero segment from the signal
                trimmed_audio = audio_data[start_idx:end_idx + 1]
                trimmed_digit_audio_files.append(trimmed_audio)
        trimmed_audio_data_matrix.append(trimmed_digit_audio_files)
    return trimmed_audio_data_matrix

# Function to remove amplitudes from a single audio file
# that are not within a certain threshold

def remove_low_amplitude_segments_single(audio_data, threshold=0.02):
    # Find segments with amplitude above the threshold
    non_zero_indices = np.where(np.abs(audio_data) > threshold)[0]

    # Get the start and end indices of non-zero segments
    if len(non_zero_indices) > 0:
        start_idx = non_zero_indices[0]
        end_idx = non_zero_indices[-1]

        # Extract the non-zero segment from the signal
        trimmed_audio = audio_data[start_idx:end_idx + 1]
        return trimmed_audio
    else:
        # If no segment found above the threshold, return the original audio data
        return audio_data

# Function to 3D plot the 3 main principal components for all 10 classifications
def plot_all_classificationt_PCA(projected_data):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'gray']

    # Scatter plot for each spoken digit
    for digit, color in zip(range(10), colors):
        digit_indices = range(digit * 100, (digit + 1) * 100)
        ax.scatter(projected_data[digit_indices, 0], projected_data[digit_indices, 1], projected_data[digit_indices, 2],
                   c=color, label=str(digit))

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Projection of Spoken Digits onto Three Main Principal Components')
    ax.legend(title='Digit')
    plt.show()

# Function to 3D plot the 3 main principal components for only two selected digits


def plot_classification_PCA_two_digits(projected_data, digit1, digit2):
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for the two digits
    colors = ['r', 'b']

    # Scatter plot for each specified digit
    for digit, color in zip([digit1, digit2], colors):
        digit_indices = range(digit * 100, (digit + 1) * 100)
        ax.scatter(projected_data[digit_indices, 0], projected_data[digit_indices, 1], projected_data[digit_indices, 2],
                   c=color, label=str(digit))

    # Set labels and title
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Projection of Spoken Digits onto Three Main Principal Components')

    # Add legend
    ax.legend(title='Digit')

    # Show plot
    plt.show()

def plot_classification_PCA_two_digits_unknown(projected_data, digit1, digit2, unknown_projected_data):
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'b']
    unknown_color = 'g'

    for digit, color in zip([digit1, digit2], colors):
        digit_indices = range(digit * 100, (digit + 1) * 100)
        ax.scatter(projected_data[digit_indices, 0], projected_data[digit_indices, 1], projected_data[digit_indices, 2],
                   c=color, label=str(digit))


        ax.scatter(unknown_projected_data[0], unknown_projected_data[1], unknown_projected_data[2],
                   c=unknown_color, label='Unknown')

    # Set labels and title
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Projection of Spoken Digits onto Three Main Principal Components')

    # Add legend
    ax.legend()

    # Show plot
    plt.show()

def plot_two_selected_digits(digit1, digit2, projected_data):
    indices_digit1 = range(digit1 * 100, (digit1 + 1) * 100)
    indices_digit2 = range(digit2 * 100, (digit2 + 1) * 100)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(projected_data[indices_digit1, 0], projected_data[indices_digit1, 1], projected_data[indices_digit1, 2],
               c='r', label=str(digit1))
    ax.scatter(projected_data[indices_digit2, 0], projected_data[indices_digit2, 1], projected_data[indices_digit2, 2],
               c='b', label=str(digit2))

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Projection of Selected Spoken Digits onto Three Main Principal Components')
    ax.legend(title='Digit')
    plt.show()

# Function to plot an audio signal given a spoken digit and file number
def plot_audio_signal(audio_data_matrix, digit, file_number):
    if digit < 0 or digit > 9:
        print("Invalid digit. Digit should be between 0 and 9.")
        return
    if file_number < 0 or file_number >= len(audio_data_matrix[digit]):
        print("Invalid file number for the given digit.")
        return

    audio_signal = audio_data_matrix[digit][file_number]
    plt.figure(figsize=(10, 4))
    plt.plot(audio_signal)
    plt.title(f"Digit {digit}, File {file_number}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

# Function to preprocess the new audio file for part 10


# Function to 3D plot the 3 main principal components for only two selected digits plus the new audio file


'''
    Project is to classify spoken audio digits. In my directory, I have 10 folders
    that represent the audio digits 0-9. Each folder contains 100 .wav files for 
    each folder. 

'''

'''
    Step 1 & 2
'''
root_directory = 'C:\\1\\ECE 172\\Project\\Data\\Spoken Digits'
audio_data_matrix = load_audio_data(root_directory)

# audio_data_matrix is a 3D numpy array where the first dimension represents digits 0 through 9,
# the second dimension represents the individual audio files, and the third dimension contains the audio data itself.
# For example, to access the audio data for digit 0 and the 5th audio file, you can use:
# audio_data_digit0_file5 = audio_data_matrix[0][4]


'''
#Code to verify the size of the matrix
The matrix should be 

rows = len(audio_data_matrix)  # returns the amount of rows in our case (spoken digits) should be 10

cols = len(audio_data_matrix[0])  # return the amount of audio files of the digit in the brackets
#should be 100

length_audio_file_0_0 = len(audio_data_matrix[0][0])  # returns the length of the audio file at first [spoken digit] and
# [row number]

print("Amount of rows in the matrix (Spoken digits): {}" .format(rows))
print("Amount of audio files for digit 0: {}".format(cols))
print("The number of samples for the first audio file for digit 0: {}".format(length_audio_file_0_0))

'''

'''
    Step 3.a
    
    For since I am using audio files, I had to do additional steps.
    
    After completing all of part A, I noticed that all the digits in the
    PCA plot was not distinguishable. I believe its because the all of the zeros that I am appending 
    to the other audio files. 

'''
trimmed_audio_data = remove_low_amplitude_segments(audio_data_matrix)

'''
    Step 3.b
    All audio files need to have the same number of samples
    To do this, the maximum number of samples will be found in 
    the Audio data matrix, and that will minimum number of samples
    for all audio files. All other files will have 0 appended to them
'''

max_samples = find_max_samples(trimmed_audio_data)

max_digit, audio_num_max = find_max_samples_position(trimmed_audio_data)

audio_data_matrix = append_zeros_to_all_audio_files(trimmed_audio_data)

'''
    Step 4
    
    The next part turns a 3D matrix into a 2D
    Flatten each audio file in the audio_data_matrix and collect the flattened arrays into a list.
    Nested loop iterates over each digit and each audio file within that digit,
    flattens each audio file into a 1D array, and appends it to the flattened_audio_data list.
    Matrix B is created by column stacking the matrix
    
    The rows represent the samples of each audio file with the size of max samples,  
    col0 is the first sample of the audio digit
    
    The columns represent each audio file, first 100 columns is audio digit 0, total is 1000
    final matrix size is max samples x 1000
'''

flattened_audio_data = []
for digit in range(10):
    for audio_file in trimmed_audio_data[digit]:
        flattened_audio_data.append(audio_file.flatten())

# Stack Flattened Audio Files as Columns of Matrix B
matrix_B = np.column_stack(flattened_audio_data)

'''
    Step 5
    I interpreted this as I need to calculate the average audio file with respect to all 1000 audio files
    so I calculate the average amplitude of all files in sample 0 (also meaning row 0)
'''
average_audio = np.mean(matrix_B, axis=1)

'''
    Step 6
    Subtract the average audio from the original matrixB
'''
centered_matrix = matrix_B - average_audio[:, np.newaxis]

'''
    Step 7 
    SVD analysis on the centered audio dataset
    # Perform Singular Value Decomposition (SVD) on matrix B
'''
# Perform Singular Value Decomposition (SVD) on matrix B
U, S, Vt = np.linalg.svd(centered_matrix,full_matrices= False)

'''

    Step 8
    Compute SVD analysis on each category


'''
# Step 8: Compute SVD analysis on each category
SVD_results = []

for digit_audio_data in trimmed_audio_data:
    flattened_digit_audio_data = [audio_file.flatten() for audio_file in digit_audio_data]

    matrix_C = np.column_stack(flattened_digit_audio_data)

    centered_matrix_digit = matrix_C - np.mean(matrix_C, axis=1)[:, np.newaxis]

    U_digit, S_digit, Vt_digit = np.linalg.svd(centered_matrix_digit, full_matrices=False)

    SVD_results.append((U_digit, S_digit, Vt_digit))



'''
    Step 9
    
    Since it is difficult to see separation with all 10 classifications
    on the 3D plot, a function is added where only two 
    spoken digits are plotted

'''


plot_all_classificationt_PCA(U)
plot_classification_PCA_two_digits(U,0,7)


'''
    Step 10
    I am loading in an audio files has the spoken digit four

'''

unknown_audio_data, _ = librosa.load('0_nicolas_10.wav', sr=8000)

unknown_audio_data_trimmed = remove_low_amplitude_segments_single(unknown_audio_data,0.02)

if len(unknown_audio_data_trimmed) < max_samples:
    unknown_audio_data_trimmed = np.append(unknown_audio_data_trimmed, [0] * (max_samples - len(unknown_audio_data_trimmed)))

flattened_unknown_audio_data = unknown_audio_data_trimmed.flatten()

unknown_audio_data_col_stack = np.column_stack(flattened_unknown_audio_data)
centered_unknown_audio = unknown_audio_data_col_stack - average_audio[:, np.newaxis]

U_unknown, S_unknown, Vt_unknown = np.linalg.svd(centered_matrix,full_matrices= False)

plot_classification_PCA_two_digits_unknown(U,0,7,U_unknown)



'''
    Will be used to load into Part B, Parc C, and Part D
'''
projected_data = centered_matrix.T @ U[:, :3]
np.save('projected_data.npy', projected_data)




