import cv2
import pandas as pd
from tkinter import Tk, filedialog, Button, Label, Frame
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load multiple datasets
dataset_paths = ['colors.csv', 'wikipedia_color_names.csv', 'wikipedia_x11_colors.csv', 'colorhexa_com.csv']
# Read datasets and concatenate into a single dataframe
dfs = []
for path in dataset_paths:
    index = ["color", "color_name", "hex", "R", "G", "B"]
    df = pd.read_csv(path, names=index, header=None, skiprows=1)  # Skip the header row
    df[['R', 'G', 'B']] = df[['R', 'G', 'B']].apply(pd.to_numeric, errors='coerce')
    dfs.append(df)

csv = pd.concat(dfs, ignore_index=True)

# Explicitly convert 'color_name' column to string
csv['color_name'] = csv['color_name'].astype(str)

# Prepare data for KNN
X = csv[['R', 'G', 'B']].values
y = csv['color_name'].values
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X, y)

# Initialize global variables
clicked = False
r = g = b = x_pos = y_pos = 0
img = None

# Function to get x, y coordinates of mouse double click
def draw_function(event, x, y, flags, param):
    global b, g, r, x_pos, y_pos, clicked, img
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked = True
        x_pos = x
        y_pos = y
        b, g, r = map(int, img[y, x])

# Function to handle file selection
def open_file():
    global img
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = cv2.imread(file_path)
        detect_color(img)

# Function to detect color using KNN
def detect_color(img):
    global clicked, r, g, b, x_pos, y_pos
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_function)

    # Create a frame for labels
    label_frame = Frame(root)
    label_frame.pack(pady=10)

    # Label to display color information
    color_label = Label(label_frame, text="Color Information:")
    color_label.grid(row=0, column=0, padx=10)

    while True:
        cv2.imshow("image", img)
        if clicked:
            r_norm = r / 255.0
            g_norm = g / 255.0
            b_norm = b / 255.0
            cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

            # Use KNN to predict the color name
            color_array = np.array([[r, g, b]])
            predicted_color = knn_classifier.predict(color_array)
            color_name = predicted_color[0]

            color_info = f"{color_name} R={int(r_norm * 255)} G={int(g_norm * 255)} B={int(b_norm * 255)}"
            cv2.putText(img, color_info, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if r + g + b >= 600:
                cv2.putText(img, color_info, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            # Update label with color information
            color_label.config(text="Color Information: " + color_info)
            clicked = False

        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

# Create a Tkinter window
root = Tk()
root.title("Color Detector")

# Label for instructions
instruction_label = Label(root, text="Add your image and then double-click on a particular area to detect color")
instruction_label.pack(pady=10)

# Button to open file dialog
btn_open = Button(root, text="Open Image", command=open_file)
btn_open.pack(pady=20)

root.mainloop()
