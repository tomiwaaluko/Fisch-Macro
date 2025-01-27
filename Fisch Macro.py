import tkinter as tk
import pyautogui
import threading
import time
from pynput import mouse, keyboard as kb
import keyboard  # For global hotkeys

# Global variables for keyboard/mouse macros
recorded_keyboard_actions = []
recorded_mouse_actions = []
recording_keyboard = False
recording_mouse = False
playing = False

# Global variables for SHAKE detection
detecting_shake = False
reference_image_path = r"C:\Users\gokug\Downloads\ShakeButton.png"  # Replace with your reference image path


### Keyboard and Mouse Macros ###
def record_keyboard_macro():
    global recording_keyboard, recorded_keyboard_actions
    recording_keyboard = True
    recorded_keyboard_actions = []

    def on_key_press(key):
        if recording_keyboard:
            recorded_keyboard_actions.append(("key_press", key, time.time()))

    def on_key_release(key):
        if recording_keyboard:
            recorded_keyboard_actions.append(("key_release", key, time.time()))

    keyboard_listener = kb.Listener(on_press=on_key_press, on_release=on_key_release)
    keyboard_listener.start()

    while recording_keyboard:
        time.sleep(0.1)

    keyboard_listener.stop()


def play_keyboard_macro():
    global playing
    playing = True
    kb_controller = kb.Controller()

    if not recorded_keyboard_actions:
        print("No keyboard actions recorded.")
        return

    for i, action in enumerate(recorded_keyboard_actions):
        if not playing:
            break
        action_type, key, timestamp = action
        if action_type == "key_press":
            kb_controller.press(key)
        elif action_type == "key_release":
            kb_controller.release(key)
        if i < len(recorded_keyboard_actions) - 1:
            time.sleep(0.1)


def record_mouse_macro():
    global recording_mouse, recorded_mouse_actions
    recording_mouse = True
    recorded_mouse_actions = []

    def on_mouse_click(x, y, button, pressed):
        if recording_mouse:
            recorded_mouse_actions.append(("mouse_click", x, y, button, pressed, time.time()))

    mouse_listener = mouse.Listener(on_click=on_mouse_click)
    mouse_listener.start()

    while recording_mouse:
        time.sleep(0.1)

    mouse_listener.stop()


def play_mouse_macro():
    global playing
    playing = True
    mouse_controller = mouse.Controller()

    if not recorded_mouse_actions:
        print("No mouse actions recorded.")
        return

    for action in recorded_mouse_actions:
        if not playing:
            break
        action_type, x, y, button, pressed, timestamp = action
        if action_type == "mouse_click":
            mouse_controller.position = (x, y)
            if pressed:
                mouse_controller.press(button)
            else:
                mouse_controller.release(button)


def stop_recording():
    global recording_keyboard, recording_mouse
    recording_keyboard = False
    recording_mouse = False


def pause_macro():
    global playing
    playing = False


### SHAKE Detection ###
def detect_and_click_image():
    """Detect the reference image on the screen and click on it."""
    global detecting_shake

    while detecting_shake:
        # Locate the reference image on the screen
        location = pyautogui.locateOnScreen(reference_image_path, confidence=0.8)
        if location:
            # Get the center of the detected location
            center_x, center_y = pyautogui.center(location)
            # Move the mouse to the location and click
            pyautogui.moveTo(center_x, center_y, duration=0.1)
            pyautogui.click()
            print("Clicked on the detected object!")
        else:
            print("Object not found on the screen.")

        # Delay between checks
        time.sleep(0.5)


def start_image_detection():
    """Start detecting the reference image."""
    global detecting_shake
    detecting_shake = True
    threading.Thread(target=detect_and_click_image, daemon=True).start()


def stop_image_detection():
    """Stop detecting the reference image."""
    global detecting_shake
    detecting_shake = False


### UI ###
def create_ui():
    root = tk.Tk()
    root.title("Macro Recorder and SHAKE Detection")

    # Keyboard Section
    keyboard_frame = tk.Frame(root, padx=10, pady=10)
    keyboard_frame.grid(row=0, column=0, sticky="n")

    keyboard_label = tk.Label(keyboard_frame, text="Keyboard Macros", font=("Arial", 12, "bold"))
    keyboard_label.pack(pady=5)

    tk.Button(keyboard_frame, text="Start Recording (Keyboard)", command=lambda: threading.Thread(target=record_keyboard_macro).start()).pack(pady=5)
    tk.Button(keyboard_frame, text="Play Keyboard Macro", command=lambda: threading.Thread(target=play_keyboard_macro).start()).pack(pady=5)

    # Mouse Section
    mouse_frame = tk.Frame(root, padx=10, pady=10)
    mouse_frame.grid(row=0, column=1, sticky="n")

    mouse_label = tk.Label(mouse_frame, text="Mouse Macros", font=("Arial", 12, "bold"))
    mouse_label.pack(pady=5)

    tk.Button(mouse_frame, text="Start Recording (Mouse)", command=lambda: threading.Thread(target=record_mouse_macro).start()).pack(pady=5)
    tk.Button(mouse_frame, text="Play Mouse Macro", command=lambda: threading.Thread(target=play_mouse_macro).start()).pack(pady=5)

    # SHAKE Detection Section
    shake_frame = tk.Frame(root, padx=10, pady=10)
    shake_frame.grid(row=1, column=0, columnspan=2, sticky="n")

    shake_label = tk.Label(shake_frame, text="SHAKE Image Detection", font=("Arial", 12, "bold"))
    shake_label.pack(pady=5)

    tk.Button(shake_frame, text="Start Detecting SHAKE", command=start_image_detection).pack(pady=5)
    tk.Button(shake_frame, text="Stop Detecting SHAKE", command=stop_image_detection).pack(pady=5)

    # Legends Section
    legend_frame = tk.Frame(root, padx=10, pady=10)
    legend_frame.grid(row=2, column=0, columnspan=2, sticky="n")

    legend_label = tk.Label(
        legend_frame,
        text=(
            "Keyboard Shortcuts:\n"
            "F5: Start Keyboard Recording\n"
            "F6: Start Mouse Recording\n"
            "F7: Play Macros\n"
            "F8: Pause Macros\n"
            "F9: Start SHAKE Detection\n"
            "F10: Stop SHAKE Detection"
        ),
        font=("Arial", 10),
        justify="left",
    )
    legend_label.pack(pady=5)

    root.mainloop()


### Hotkeys ###
def setup_hotkeys():
    keyboard.add_hotkey("F5", lambda: threading.Thread(target=record_keyboard_macro).start())
    keyboard.add_hotkey("F6", lambda: threading.Thread(target=record_mouse_macro).start())
    keyboard.add_hotkey("F7", lambda: threading.Thread(target=play_keyboard_macro).start())
    keyboard.add_hotkey("F8", pause_macro)
    keyboard.add_hotkey("F9", start_image_detection)
    keyboard.add_hotkey("F10", stop_image_detection)


if __name__ == "__main__":
    threading.Thread(target=setup_hotkeys, daemon=True).start()
    create_ui()
