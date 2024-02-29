import argparse
import pandas as pd
from chatbot_website import create_app
import csv
import sqlite3
print("MAIN CBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")

parser = argparse.ArgumentParser()

parser.add_argument('--len_history', type=int, default=3, help="combined prompt upper limit")
parser.add_argument('--sum', type=int, default=1, help="summarise answer for prompt or not")

args = parser.parse_args()

app = create_app(args)

if __name__ == '__main__':
    # data = pd.DataFrame(columns = ["id", "image_url", "user_message", "bot_message", "feedback"])
    app.run(debug=False)
    # data.to_csv("response.csv")
