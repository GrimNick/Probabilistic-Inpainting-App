#suru ma yolo wawla code has to run before this
#add dropdown and select options, add deepfill v2 or other probabilistic inpainting
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from matplotlib.patches import Circle, Rectangle
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import tkinter as tk
from tkinter import ttk

import os
# ==== Configuration ==== 
IMAGE_PATH = 'test1.jpg'
OUTPUT_PATH = 'modified_test1.png'
LOG_PATH = 'removal_log.txt'
DETECTION_FILE = 'detection_results.txt'
detections = []
#suru ma yolo wawla code has to run before this
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




# ==== Clear previous log ==== 
with open(LOG_PATH, 'w') as f:
    pass  # truncate log file at startup

# ==== Load Image ==== 
orig_image = cv2.imread(IMAGE_PATH)
if orig_image is None:
    raise FileNotFoundError(f"Image '{IMAGE_PATH}' not found.")
working_image = orig_image.copy()
image_rgb = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)

# ==== Matplotlib Setup ==== 
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1)
img_disp = ax.imshow(image_rgb)
ax.set_title('Select a mode and draw to remove')
ax.axis('off')

# Toolbar for mode selection
rax = plt.axes([0.78, 0.5, 0.18, 0.35], frameon=True)
radio = RadioButtons(rax, ('Rectangle', 'Circle', 'Freehand'), active=0)
# Save button
sax = plt.axes([0.78, 0.3, 0.18, 0.1])
save_button = Button(sax, 'Save')

# State variables
draw_mode = 'Rectangle'
start_pt = None
circle_patch = None
rect_patch = None
lasso_widget = None
cid_press = cid_motion = cid_release = None

# ==== Logging Utility ==== 
def log_removal(entry: str):
    with open(LOG_PATH, 'a') as f:
        f.write(entry + '\n')

# ==== Removal Callbacks ==== 
def blackout_rectangle(extents):
    x1, x2, y1, y2 = map(int, extents)
    working_image[y1:y2, x1:x2] = 0
    log_removal(f"rectangle,{x1},{y1},{x2},{y2}")

def blackout_circle(center, radius):
    mask = np.zeros(working_image.shape[:2], dtype=np.uint8)
    cx, cy = center
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    working_image[mask == 255] = 0
    log_removal(f"circle,{cx},{cy},{radius}")

def blackout_freehand(verts):
    verts = list(verts)
    closed = verts + [verts[0]]
    poly_path = Path(closed)
    h, w = working_image.shape[:2]
    y, x = np.mgrid[:h, :w]
    coords = np.vstack((x.flatten(), y.flatten())).T
    mask = poly_path.contains_points(coords).reshape(h, w)
    working_image[mask] = [0, 0, 0]
    # Log freehand points as comma-separated pairs
    pts = ";".join(f"{int(x)},{int(y)}" for x, y in verts)
    log_removal(f"freehand,{pts}")

# ==== Mode Cleanup ==== 
def clear_mode():
    global rect_patch, circle_patch, cid_press, cid_motion, cid_release, lasso_widget
    for cid in (cid_press, cid_motion, cid_release):
        if cid:
            fig.canvas.mpl_disconnect(cid)
    if rect_patch:
        rect_patch.remove()
        rect_patch = None
    if circle_patch:
        circle_patch.remove()
        circle_patch = None
    if lasso_widget:
        lasso_widget.disconnect_events()
        lasso_widget = None

# ==== Redraw ==== 
def finalize():
    img_disp.set_data(cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB))
    fig.canvas.draw()

# ==== Save Handler ==== 
def on_save(event):
    cv2.imwrite(OUTPUT_PATH, working_image)
    print(f"Image saved to {OUTPUT_PATH}")

# ==== Activation Functions ==== 
def activate_rectangle():
    global start_pt, rect_patch, cid_press, cid_motion, cid_release
    def on_press(event):
        global start_pt, rect_patch
        if event.inaxes != ax:
            return
        start_pt = (int(event.xdata), int(event.ydata))
        if rect_patch:
            rect_patch.remove()
        rect_patch = Rectangle(start_pt, 0, 0, edgecolor='green', facecolor='none', linewidth=2)
        ax.add_patch(rect_patch)
        fig.canvas.draw()
    def on_motion(event):
        if start_pt is None or event.inaxes != ax:
            return
        x0, y0 = start_pt
        x1, y1 = int(event.xdata), int(event.ydata)
        rect_patch.set_width(x1 - x0)
        rect_patch.set_height(y1 - y0)
        fig.canvas.draw()
    def on_release(event):
        global start_pt
        if start_pt is None or event.inaxes != ax:
            return
        x0, y0 = start_pt
        x1, y1 = int(event.xdata), int(event.ydata)
        extents = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
        blackout_rectangle(extents)
        start_pt = None
        finalize()
    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

def activate_circle():
    global start_pt, circle_patch, cid_press, cid_motion, cid_release
    def on_press(event):
        global start_pt, circle_patch
        if event.inaxes != ax:
            return
        start_pt = (int(event.xdata), int(event.ydata))
        if circle_patch:
            circle_patch.remove()
        circle_patch = Circle(start_pt, 0, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(circle_patch)
        fig.canvas.draw()
    def on_motion(event):
        if start_pt is None or event.inaxes != ax:
            return
        x0, y0 = start_pt
        x1, y1 = int(event.xdata), int(event.ydata)
        circle_patch.set_radius(int(np.hypot(x1 - x0, y1 - y0)))
        fig.canvas.draw()
    def on_release(event):
        global start_pt
        if start_pt is None or event.inaxes != ax:
            return
        x0, y0 = start_pt
        x1, y1 = int(event.xdata), int(event.ydata)
        blackout_circle((x0, y0), int(np.hypot(x1 - x0, y1 - y0)))
        start_pt = None
        finalize()
    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

def activate_freehand():
    global lasso_widget
    def on_select(verts):
        if not verts:
            return
        blackout_freehand(verts)
        finalize()
    lasso_widget = LassoSelector(ax, on_select=on_select)

# ==== Mode Change Handler ==== 
def on_mode_change(label):
    global draw_mode
    draw_mode = label
    clear_mode()
    if draw_mode == 'Rectangle':
        activate_rectangle()
    elif draw_mode == 'Circle':
        activate_circle()
    else:
        activate_freehand()

def on_closing():
    # Perform any necessary cleanup here
    root.destroy()  # This will close the window and exit the mainloop


# ==== Connect Events ==== 
save_button.on_clicked(on_save)
on_mode_change('Rectangle')
radio.on_clicked(on_mode_change)
plt.show()
