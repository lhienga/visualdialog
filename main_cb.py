import pandas as pd
from chatbot_website import create_app
print("MAIN CBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
app = create_app()

if __name__ == '__main__':
    # data = pd.DataFrame(columns = ["id", "image_url", "user_message", "bot_message", "feedback"])
    app.run(debug=False)
    
    # data.to_csv("response.csv")
