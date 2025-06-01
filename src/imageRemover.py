import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

# ==== Configuration ====
IMAGE_PATH = 'test5.jpg'
OUTPUT_PATH = 'modified_test1.png'
LOG_PATH = 'removal_log.txt'
DETECTION_FILE = 'detection_results.txt'

# ==== Load Image ====
orig_image = cv2.imread(IMAGE_PATH)
if orig_image is None:
    raise FileNotFoundError(f"Image '{IMAGE_PATH}' not found.")
working_image = orig_image.copy()
image_rgb = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)

# ==== Parse detection_results.txt ====
detections = []
if os.path.exists(DETECTION_FILE):
    with open(DETECTION_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 6:
                label, shape, x1, y1, x2, y2 = parts
                detections.append({
                    'label': label,
                    'shape': shape,
                    'coords': (int(x1), int(y1), int(x2), int(y2))
                })

startRemoval =True #if new image , we have to change this to true for reusability
# ==== Logging Utility ====
def log_removal(entry: str):
    global startRemoval
    if(startRemoval): 
        with open(LOG_PATH, 'w') as f:
            f.write(entry + '\n')
            startRemoval = False
    else:
        with open(LOG_PATH, 'a') as f:
            f.write(entry + '\n')


# ==== Tkinter Setup ====
root = tk.Tk()
root.title("Image Editor with Manual and Dropdown Removal")

# Handle window close event
def on_closing():
    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Create frames
control_frame = tk.Frame(root)
control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

figure_frame = tk.Frame(root)
figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# ==== Matplotlib Setup ====
fig, ax = plt.subplots(figsize=(8, 6))
img_disp = ax.imshow(image_rgb)
ax.set_title('Select a mode and draw to remove')
ax.axis('off')

canvas = FigureCanvasTkAgg(fig, master=figure_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ==== Dropdown Menu ====
dropdown_label = tk.Label(control_frame, text="Select Detected Item:")
dropdown_label.pack(pady=(0, 5))

dropdown_options = [f"{d['label']} ({i})" for i, d in enumerate(detections)]
dropdown_var = tk.StringVar()
dropdown = ttk.Combobox(control_frame, textvariable=dropdown_var, values=dropdown_options, state="readonly")
dropdown.pack()

highlight_patch = None

def on_select(event):
    global highlight_patch
    selection = dropdown_var.get()
    if not selection:
        return
    index = int(selection.split('(')[-1].strip(')'))
    detection = detections[index]
    x1, y1, x2, y2 = detection['coords']

    # Remove previous highlight if exists
    if highlight_patch:
        highlight_patch.remove()

    # Draw rectangle around the selected area
    width = x2 - x1
    height = y2 - y1
    highlight_patch = Rectangle((x1, y1), width, height,
                                edgecolor='yellow', facecolor='none', linewidth=2)
    ax.add_patch(highlight_patch)
    ax.set_title(f"Selected: {detection['label']} ({index})")
    canvas.draw()

dropdown.bind("<<ComboboxSelected>>", on_select)

# ==== Remove Button ====
def remove_selected():
    selection = dropdown_var.get()
    if not selection:
        return
    index = int(selection.split('(')[-1].strip(')'))
    detection = detections[index]
    x1, y1, x2, y2 = detection['coords']

    # Blackout the selected area
    working_image[y1:y2, x1:x2] = 0

    # Update the displayed image
    img_disp.set_data(cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB))
    canvas.draw()

    # Log the removal
    log_removal(f"{detection['shape']},{x1},{y1},{x2},{y2}")

    # Remove the highlight
    global highlight_patch
    if highlight_patch:
        highlight_patch.remove()
        highlight_patch = None
    ax.set_title('Select a mode and draw to remove')
    canvas.draw()

remove_button = tk.Button(control_frame, text="Remove Selected", command=remove_selected)
remove_button.pack(pady=10)

# ==== Save Button ====
def save_image():
    cv2.imwrite(OUTPUT_PATH, working_image)
    print(f"Image saved to {OUTPUT_PATH}")

save_button = tk.Button(control_frame, text="Save Image", command=save_image)
save_button.pack(pady=10)

# ==== Manual Removal Modes ====
mode_label = tk.Label(control_frame, text="Manual Removal Mode:")
mode_label.pack(pady=(20, 5))

mode_var = tk.StringVar(value='Rectangle')
modes = ['Rectangle', 'Circle', 'Freehand']
for mode in modes:
    rb = tk.Radiobutton(control_frame, text=mode, variable=mode_var, value=mode)
    rb.pack(anchor='w')

start_pt = None
rect_patch = None
circle_patch = None
lasso = None

def clear_drawings():
    global rect_patch, circle_patch, lasso
    if rect_patch:
        rect_patch.remove()
        rect_patch = None
    if circle_patch:
        circle_patch.remove()
        circle_patch = None
    if lasso:
        lasso.disconnect_events()
        lasso = None
    canvas.draw()

def on_mouse_press(event):
    global start_pt, rect_patch, circle_patch
    if event.inaxes != ax:
        return
    start_pt = (int(event.xdata), int(event.ydata))
    mode = mode_var.get()
    if mode == 'Rectangle':
        rect_patch = Rectangle(start_pt, 0, 0, edgecolor='green', facecolor='none', linewidth=2)
        ax.add_patch(rect_patch)
    elif mode == 'Circle':
        circle_patch = Circle(start_pt, 0, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(circle_patch)
    canvas.draw()

def on_mouse_move(event):
    global rect_patch, circle_patch
    if start_pt is None or event.inaxes != ax:
        return
    x0, y0 = start_pt
    x1, y1 = int(event.xdata), int(event.ydata)
    mode = mode_var.get()
    if mode == 'Rectangle' and rect_patch:
        rect_patch.set_width(x1 - x0)
        rect_patch.set_height(y1 - y0)
    elif mode == 'Circle' and circle_patch:
        radius = int(np.hypot(x1 - x0, y1 - y0))
        circle_patch.set_radius(radius)
    canvas.draw()

def on_mouse_release(event):
    global start_pt, rect_patch, circle_patch
    if start_pt is None or event.inaxes != ax:
        return
    x0, y0 = start_pt
    x1, y1 = int(event.xdata), int(event.ydata)
    mode = mode_var.get()
    if mode == 'Rectangle':
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        working_image[y_min:y_max, x_min:x_max] = 0
        log_removal(f"rectangle,{x_min},{y_min},{x_max},{y_max}")
        if rect_patch:
            rect_patch.remove()
            rect_patch = None
    elif mode == 'Circle':
        radius = int(np.hypot(x1 - x0, y1 - y0))
        mask = np.zeros(working_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x0, y0), radius, 255, -1)
        working_image[mask == 255] = 0
        log_removal(f"circle,{x0},{y0},{radius}")
        if circle_patch:
            circle_patch.remove()
            circle_patch = None
    img_disp.set_data(cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB))
    canvas.draw()
    start_pt = None

def on_freehand_select(verts):
    if not verts:
        return
    closed = verts + [verts[0]]
    poly_path = Path(closed)
    h, w = working_image.shape[:2]
    y, x = np.mgrid[:h, :w]
    coords = np.vstack((x.flatten(), y.flatten())).T
    mask = poly_path.contains_points(coords).reshape(h, w)
    working_image[mask] = [0, 0, 0]
    pts = ";".join(f"{int(x)},{int(y)}" for x, y in verts)
    log_removal(f"freehand,{pts}")
    img_disp.set_data(cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB))
    canvas.draw()

def on_mode_change(*args):
    clear_drawings()
    mode = mode_var.get()
    if mode == 'Freehand':
        global lasso
        lasso = LassoSelector(ax, on_select=on_freehand_select)

mode_var.trace_add('write', on_mode_change)

canvas.mpl_connect('button_press_event', on_mouse_press)
canvas.mpl_connect('motion_notify_event', on_mouse_move)
canvas.mpl_connect('button_release_event', on_mouse_release)

# ==== Run the Tkinter main loop ====
root.mainloop()
