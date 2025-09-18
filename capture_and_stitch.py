import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from pynput import mouse
from pynput.keyboard import Key, KeyCode, Listener as KeyboardListener
import mss
import sys
from typing import Union

# --- Part 1: The "Frozen Screen" Visual Selection ---

with mss.mss() as sct:
    monitor = sct.monitors[0]
    screen_shot = sct.grab(monitor)
    pil_img = Image.frombytes("RGB", screen_shot.size, screen_shot.bgra, "raw", "BGRX")

root = tk.Tk()
root.attributes('-fullscreen', True)
root.attributes('-topmost', True)

tk_img = ImageTk.PhotoImage(pil_img)
canvas = tk.Canvas(root, width=tk_img.width(), height=tk_img.height(), highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=tk.TRUE)
canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)

selection_rectangle = canvas.create_rectangle(0, 0, 0, 0, outline="red", width=2)

# --- START OF FIX ---
# We need variables to store the final position
start_x, start_y, end_x, end_y = 0, 0, 0, 0
is_drawing = False

def on_click(x: int, y: int, button, pressed: bool):
    global start_x, start_y, end_x, end_y, is_drawing
    if pressed:
        is_drawing = True
        start_x, start_y = x, y
        canvas.coords(selection_rectangle, start_x, start_y, start_x, start_y)
    else: # Mouse button is released
        is_drawing = False
        # **THE CRITICAL FIX**: Store the final coordinates *before* destroying.
        end_x, end_y = x, y
        root.destroy()
        return False
# --- END OF FIX ---

def on_move(x: int, y: int):
    if is_drawing:
        canvas.coords(selection_rectangle, start_x, start_y, x, y)

print("--- Selection Instructions ---")
print("Your screen is now frozen. Click and drag to select an area.")
print("Release the mouse button to confirm.")
print("----------------------------")

mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
mouse_listener.start()
root.mainloop()
mouse_listener.stop()

# --- Process the coordinates ---
# **THE CRUCIAL CHANGE**: Use our saved global variables instead of asking the dead canvas.
left = min(start_x, end_x)
top = min(start_y, end_y)
right = max(start_x, end_x)
bottom = max(end_y, start_y)

if right - left < 10 or bottom - top < 10:
    print("\nError: Selected area is too small. Exiting.")
    sys.exit(1)

monitor_area = {"left": left, "top": top, "width": right - left, "height": bottom - top}
print(f"\nArea selected! Dimensions: {monitor_area['width']}x{monitor_area['height']}")


# --- The rest of the script is unchanged ---

# --- Part 2: The Manual Capture Loop ---
image_fragments = []

def on_press(key: Union[Key, KeyCode, None]):
    if isinstance(key, KeyCode):
        if key.char == 'c':
            with mss.mss() as sct:
                sct_img = sct.grab(monitor_area)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                image_fragments.append(img)
                print(f"  > Captured fragment #{len(image_fragments)}")
        elif key.char == 'q':
            print("Quitting capture mode...")
            return False

print("\n--- Capture Instructions ---")
print("1. Scroll your content down.")
print("2. Press the 'c' key to capture the next fragment.")
print("3. Repeat until you are done.")
print("4. Press the 'q' key to quit and stitch the images.")
print("--------------------------")

with KeyboardListener(on_press=on_press) as listener:
    listener.join()

# --- Part 3 & 4: Stitching and Saving ---
if not image_fragments:
    print("\nNo images were captured. Exiting.")
    sys.exit()

if len(image_fragments) == 1:
    print("\nOnly one image was captured. Saving it directly.")
    final_image = image_fragments[0]
else:
    print(f"\nStitching {len(image_fragments)} fragments...")
    img1 = image_fragments[0]
    stitched_image = img1.copy()
    for i in range(1, len(image_fragments)):
        img2 = image_fragments[i]
        template_height = min(30, img1.shape[0])
        template = img1[-template_height:, :]
        res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        top_left_match = max_loc
        crop_y_start = top_left_match[1] + template_height
        new_part = img2[crop_y_start:, :]
        if new_part.shape[0] > 0:
            stitched_image = np.vstack((stitched_image, new_part))
        img1 = img2
    final_image = stitched_image

cv2.imwrite("scroll_capture.png", final_image)
print("\nSuccess! Result written to `scroll_capture.png`.")
