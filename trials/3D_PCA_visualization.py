import numpy as np
from similarity.source import pca_tranform
from similarity.source import fingerprint
from similarity.source import similarity
from .perturbations import *
from .utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_window(title, data_pairs, figures, similarity_pca, angles):
    """
    Create a window with plots and data
    """
    window = tk.Tk()
    window.title(title)

    panedwindow = tk.PanedWindow(window, orient=tk.HORIZONTAL)
    panedwindow.pack(fill=tk.BOTH, expand=True)

     # Frame for plots
    plot_frame = tk.Frame(panedwindow)
    panedwindow.add(plot_frame)

    # Frame for data
    data_frame = tk.Frame(panedwindow)  
    panedwindow.add(data_frame)

    for fig in figures:
        plot_graph(plot_frame, fig)

    for i, (label, (data1, data2)) in enumerate(data_pairs.items()):
        # Create an intermediary frame for each pair of ScrolledText widgets
        pair_frame = tk.Frame(data_frame)
        pair_frame.grid(row=i, column=0)

        tk.Label(pair_frame, text=label, font=("Arial Bold", 12)).pack()

        width1 = min(max(len(line) for line in data1.split('\n')), 35)
        width2 = min(max(len(line) for line in data2.split('\n')), 35)

         # Calculate number of lines for each dataset and set the height parameter
        lines1 = min(data1.count('\n') + 1, 10)  # calculate lines in data1, limit to 10
        lines2 = min(data2.count('\n') + 1, 10)  # calculate lines in data2, limit to 10

        # Create a scrollbar in the window for the first dataset
        scr1 = scrolledtext.ScrolledText(pair_frame, wrap=tk.WORD, width=width1, height=lines1)
        scr1.pack(side=tk.LEFT)

        # Create a scrollbar in the window for the second dataset
        scr2 = scrolledtext.ScrolledText(pair_frame, wrap=tk.WORD, width=width2, height=lines2)
        scr2.pack(side=tk.LEFT)
            
        # insert the data to scrollbar text boxes
        scr1.insert(tk.INSERT, data1)
        scr2.insert(tk.INSERT, data2)

    # Create an additional frame inside data_frame
    info_frame = tk.Frame(data_frame)
    info_frame.grid(row=i+1, column=0)  # Make sure it is positioned after the data 

    # create labels for similarity_pca and angles
    similarity_label = tk.Label(info_frame, text=f"Similarity_pca: \n{similarity_pca}", font=("Arial Bold", 16))
    similarity_label.grid(row=0, column=0)

    angles_label = tk.Label(info_frame, text=f"Rotation angles: \nX: {angles[0]}, \nY: {angles[1]}, \nZ: {angles[2]}", font=("Arial Bold", 16))
    angles_label.grid(row=1, column=0)

    window.mainloop()

def plot_graph(master, fig):
    """
    Plot a graph in a frame
    """
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    toolbar_frame = tk.Frame(master)  # create a new frame to hold the toolbar
    toolbar_frame.grid(row=1, column=0, sticky='w')  # position it below the canvas


def set_axes_equal1(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])    

def visualize(points, center_of_mass, principal_axes, ax, labels):

    # Visualize points with size based on mass (assuming protons)
    for i, label in enumerate((labels)):
        ax.scatter(points[i, 0], points[i, 1], points[i, 2], c='b', alpha=0.6, label = label)
        ax.text(points[i, 0], points[i, 1], points[i, 2], labels[i], fontsize=12)

    colors = ['r', 'g', 'k']
    for axis, color in zip(principal_axes, colors):
        ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
                  axis[0]*5, axis[1]*5, axis[2]*5, # The scaling factor is used to make the eigenvector more visible
                  color=color, lw=2, arrow_length_ratio=0.1)
    
    padding = 1 # Increase or decrease this value to change the padding around the axes

    # Find the maximum arrow length
    max_arrow_length = np.max([np.linalg.norm(axis) for axis in principal_axes])

    # Adjust axes to fit all points and arrows
    min_values = np.min(points, axis=0) - max_arrow_length - padding
    max_values = np.max(points, axis=0) + max_arrow_length + padding

    ax.set_xlim(min_values[0], max_values[0])
    ax.set_ylim(min_values[1], max_values[1])
    ax.set_zlim(min_values[2], max_values[2])

    set_axes_equal1(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plt.show(block=False)
    # plt.pause(0.001)

#############################################################

# Generate random 3D points
data = np.random.uniform(low = -10 , high = 10 , size = (5, 3))

# Rotate the points
#tmp_data = perturb_coordinates(data, 7)
tmp_data = data
# generate random float number between 0 and 360
angle1 = np.random.uniform(0, 360)
angle2 = np.random.uniform(0, 360)
angle3 = np.random.uniform(0, 360)
data1 = rotate_points(tmp_data, angle1, angle2, angle3)

###### PCA #########
data, transformed_data_pca, principal_axis_pca, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(data)
standard_axis_pca = np.eye(data.shape[1])
reference_points = fingerprint.get_reference_points(data.shape[1])  
distances = fingerprint.compute_distances(transformed_data_pca)
fingerprint_pca = fingerprint.get_fingerprint(transformed_data_pca)

data1, transformed_data_pca1, principal_axis_pca1, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(data1)
standard_axis_pca1 = np.eye(data1.shape[1])
reference_points1 = fingerprint.get_reference_points(data1.shape[1])
distances1 = fingerprint.compute_distances(transformed_data_pca1)
fingerprint_pca1 = fingerprint.get_fingerprint(transformed_data_pca1)

similarity_pca = 1/(1 + similarity.calculate_partial_score(fingerprint_pca, fingerprint_pca1))
print('similarity_pca: ', similarity_pca)

print(f'Angles: X: {angle1}, Y: {angle2}, Z: {angle3}')


####### PRE_PROCESSING #########
labels = [str(i) for i in range(len(data))]
fingerprint_pca_str = '\n'.join(map(str, fingerprint_pca))
fingerprint_pca1_str = '\n'.join(map(str, fingerprint_pca1))

data_pairs = {
    "Data": (str(data), str(data1)),
    "Transformed Data": (str(transformed_data_pca), str(transformed_data_pca1)),
    "Principal Components": (str(principal_axis_pca), str(principal_axis_pca1)),
    "Standard Components": (str(standard_axis_pca), str(standard_axis_pca1)),
    "Reference Points": (str(reference_points), str(reference_points1)),
    "Distances": (str(distances), str(distances1)),
    "Fingerprint": (fingerprint_pca_str, fingerprint_pca1_str) 
}
###############################


##### 3D VISUALIZATION #####
figures = []
fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(221, projection='3d')
ax1.set_title('Original data')
visualize(data, np.mean(data, axis=0), principal_axis_pca, ax1, labels)
figures.append(fig)

ax2 = fig.add_subplot(222, projection='3d')
ax2.set_title('Rotated/Perturb data')
visualize(data1, np.mean(data1, axis=0), principal_axis_pca1, ax2, labels)
figures.append(fig)

ax3 = fig.add_subplot(223, projection='3d')
ax3.set_title('PCA Transformed data')
visualize(transformed_data_pca, np.mean(transformed_data_pca, axis=0), standard_axis_pca, ax3, labels)
figures.append(fig)

ax4 = fig.add_subplot(224, projection='3d')
ax4.set_title('PCA Transformed data')
visualize(transformed_data_pca1, np.mean(transformed_data_pca1, axis=0), standard_axis_pca1, ax4, labels)
figures.append(fig)

#### DISPLAY DATA
create_window("Data Window", data_pairs, figures, similarity_pca, (angle1, angle2, angle3))
