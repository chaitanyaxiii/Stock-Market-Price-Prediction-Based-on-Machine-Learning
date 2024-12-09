import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'D:\\Excels\\Stock Market Dataset.csv'
df = pd.read_csv(file_path)

# Drop any rows with missing values
df.dropna(inplace=True)

# Define the features (independent variables) and target (dependent variable)
X = df[['Open', 'High', 'Last']]
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Check the model's performance
score = model.score(X_test, y_test)
print(f'Model R^2 score: {score}')

# Function to predict the stock price
def predict_stock_price(open_price, high_price, last_price):
    input_data = pd.DataFrame([[open_price, high_price, last_price]], columns=['Open', 'High', 'Last'])
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Function to show the prediction result
def show_result():
    try:
        open_price = float(entry_open.get())
        high_price = float(entry_high.get())
        last_price = float(entry_last.get())
        predicted_price = predict_stock_price(open_price, high_price, last_price)
        result_label.config(text=f'Predicted price: {predicted_price:.2f}')
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for prices.")

# Function to show the first 20 rows of the dataset
def show_data():
    data_window = tk.Toplevel(root)
    data_window.title("First 20 Rows of Data")
    text = tk.Text(data_window, wrap='none', font=('Arial', 10))
    text.pack(expand=True, fill='both')
    text.insert(tk.END, df.head(20).to_string())

# Function to show the model's accuracy
def show_accuracy():
    accuracy_label.config(text=f'Model R^2 score: {score:.2f}')

# Function to show the graph
def show_graph():
    plt.scatter(y_test, model.predict(X_test))
    plt.xlabel("True Prices")
    plt.ylabel("Predicted Prices")
    plt.title("True Prices vs Predicted Prices")
    plt.show()

# Create the main window
root = tk.Tk()
root.title("STOCK MARKET PRICE ANALYSIS AND PREDICTION")
root.geometry("800x600")

# Set background image
image_path = "D:\\Excels\\Stock.jpg"
image = Image.open(image_path)
bg_image = ImageTk.PhotoImage(image)
bg_label = tk.Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

# Set text and button colors
text_color = "white"
button_color = "#007acc"

# Add labels and text inputs
tk.Label(root, text="Stock Name", font=('Arial', 12), fg=text_color, bg=button_color).place(x=50, y=50)
entry_name = tk.Entry(root, font=('Arial', 12))
entry_name.place(x=200, y=50)

tk.Label(root, text="Date (YYYY-MM-DD)", font=('Arial', 12), fg=text_color, bg=button_color).place(x=50, y=100)
entry_date = tk.Entry(root, font=('Arial', 12))
entry_date.place(x=200, y=100)

tk.Label(root, text="Open Price", font=('Arial', 12), fg=text_color, bg=button_color).place(x=50, y=150)
entry_open = tk.Entry(root, font=('Arial', 12))
entry_open.place(x=200, y=150)

tk.Label(root, text="High Price", font=('Arial', 12), fg=text_color, bg=button_color).place(x=50, y=200)
entry_high = tk.Entry(root, font=('Arial', 12))
entry_high.place(x=200, y=200)

tk.Label(root, text="Last Price", font=('Arial', 12), fg=text_color, bg=button_color).place(x=50, y=250)
entry_last = tk.Entry(root, font=('Arial', 12))
entry_last.place(x=200, y=250)

# Add result label
result_label = tk.Label(root, text="", font=('Arial', 12), fg=text_color, bg=button_color)
result_label.place(x=50, y=300)

# Add buttons
tk.Button(root, text="Show Result", font=('Arial', 12), command=show_result, bg=button_color, fg=text_color).place(x=50, y=350)
tk.Button(root, text="Show Data", font=('Arial', 12), command=show_data, bg=button_color, fg=text_color).place(x=150, y=350)
tk.Button(root, text="Show Accuracy", font=('Arial', 12), command=show_accuracy, bg=button_color, fg=text_color).place(x=250, y=350)
tk.Button(root, text="Show Graph", font=('Arial', 12), command=show_graph, bg=button_color, fg=text_color).place(x=370, y=350)

# Add accuracy label
accuracy_label = tk.Label(root, text="", font=('Arial', 12), fg=text_color, bg=button_color)
accuracy_label.place(x=50, y=400)

# Run the Tkinter main loop
root.mainloop()
