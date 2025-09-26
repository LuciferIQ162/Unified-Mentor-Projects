
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Function to get user input and predict
def predict_price_range():
    # Get user input for all features
    print("Please enter the mobile phone specifications:")
    battery_power = int(input("Battery Power (mAh): "))
    blue = int(input("Bluetooth (1 for Yes, 0 for No): "))
    clock_speed = float(input("Clock Speed (GHz): "))
    dual_sim = int(input("Dual SIM (1 for Yes, 0 for No): "))
    fc = int(input("Front Camera (MP): "))
    four_g = int(input("4G (1 for Yes, 0 for No): "))
    int_memory = int(input("Internal Memory (GB): "))
    m_dep = float(input("Mobile Depth (cm): "))
    mobile_wt = int(input("Mobile Weight (g): "))
    n_cores = int(input("Number of Cores: "))
    pc = int(input("Primary Camera (MP): "))
    px_height = int(input("Pixel Resolution Height: "))
    px_width = int(input("Pixel Resolution Width: "))
    ram = int(input("RAM (MB): "))
    sc_h = int(input("Screen Height (cm): "))
    sc_w = int(input("Screen Width (cm): "))
    talk_time = int(input("Talk Time (hours): "))
    three_g = int(input("3G (1 for Yes, 0 for No): "))
    touch_screen = int(input("Touch Screen (1 for Yes, 0 for No): "))
    wifi = int(input("WiFi (1 for Yes, 0 for No): "))

    # Create a DataFrame from the user input
    user_data = pd.DataFrame({
        'battery_power': [battery_power],
        'blue': [blue],
        'clock_speed': [clock_speed],
        'dual_sim': [dual_sim],
        'fc': [fc],
        'four_g': [four_g],
        'int_memory': [int_memory],
        'm_dep': [m_dep],
        'mobile_wt': [mobile_wt],
        'n_cores': [n_cores],
        'pc': [pc],
        'px_height': [px_height],
        'px_width': [px_width],
        'ram': [ram],
        'sc_h': [sc_h],
        'sc_w': [sc_w],
        'talk_time': [talk_time],
        'three_g': [three_g],
        'touch_screen': [touch_screen],
        'wifi': [wifi]
    })

    # Make a prediction
    prediction = model.predict(user_data)

    # Interpret the prediction
    price_ranges = {
        0: 'Budget-Friendly',
        1: 'Mid-Range',
        2: 'High-End',
        3: 'Premium'
    }

    recommendation = price_ranges.get(prediction[0], "Unknown")

    print(f"\nPredicted Price Range: {prediction[0]} ({recommendation})")
    if recommendation == 'Budget-Friendly':
        print("This is a good choice for a budget-friendly phone.")
    elif recommendation == 'Mid-Range':
        print("This phone is in the mid-range price category.")
    elif recommendation == 'High-End':
        print("This is a high-end phone, a good buying decision if you are looking for premium features.")
    elif recommendation == 'Premium':
        print("This is a premium phone, a good buying decision if you are looking for top-of-the-line features.")

if __name__ == '__main__':
    predict_price_range()
