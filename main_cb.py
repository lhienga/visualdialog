import argparse
import pandas as pd
from chatbot_website import create_app
import csv
import sqlite3
print("MAIN CBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")

parser = argparse.ArgumentParser()

parser.add_argument('--len_history', type=int, default=3, help="combined prompt upper limit")

args = parser.parse_args()

app = create_app(args)

if __name__ == '__main__':
    # data = pd.DataFrame(columns = ["id", "image_url", "user_message", "bot_message", "feedback"])
    app.run(debug=False)
    # data.to_csv("response.csv")
