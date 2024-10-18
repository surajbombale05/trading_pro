import requests
import csv
from datetime import datetime

def fetch_live_data_direct(currency_pair='EUR/USD'):
    api_key = "ef575764518782bed2808081"  # Use your actual API key here
    base_currency = currency_pair.split('/')[0]  # Base currency, e.g., 'USD'
    target_currency = currency_pair.split('/')[1]  # Target currency, e.g., 'EUR'

    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"
    
    try:
        response = requests.get(url)
        print(f"Raw response: {response.text}")  # Print raw response for debugging
            
        if response.status_code == 200:
            data = response.json()
            if data['result'] == 'success':  # Check if the API call was successful
                rate = data['conversion_rates'][target_currency]
                print(f"Exchange rate {base_currency} to {target_currency}: {rate}")
                return rate
            else:
                print(f"API Error: {data['error-type']}")
        else:
            print(f"Error: Received status code {response.status_code}")
    except Exception as e:
        print(f"Error fetching live Forex data: {e}")
    
    return None

# Save data to CSV function
def save_to_csv(data, csv_file='live_forex_data.csv'):
    file_exists = False
    try:
        with open(csv_file, 'r', newline='') as file:
            file_exists = True  # CSV exists
    except FileNotFoundError:
        file_exists = False  # CSV doesn't exist, need to create it

    # Define fieldnames for CSV file
    fieldnames = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Simulate a random volume
    volume = 0.001 + (0.002 - 0.001) * 1.0  # Simulating random volume (you can adjust this as needed)

    # Add live data with same Open, High, Low, Close for this example
    live_data = {
        'Local time': datetime.now().strftime('%d.%m.%Y %H:%M:%S.%f')[:-3] + ' GMT+0530',
        'Open': data,
        'High': data,   # Assuming real-time fetching of live data, Open = High = Low = Close
        'Low': data,
        'Close': data,
        'Volume': volume
    }
    
    # Append the data to the CSV
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file does not exist
        if not file_exists:
            writer.writeheader()
        
        # Write the live data row
        writer.writerow(live_data)
        print(f"Live data appended to {csv_file}.")

# Function to fetch and save live data
def fetch_and_save():
    currency_pair = 'EUR/USD'
    rate = fetch_live_data_direct(currency_pair)
    if rate is not None:
        save_to_csv(rate)
    else:
        print("Failed to fetch live data.")

# Call the function to fetch and save data
fetch_and_save()
