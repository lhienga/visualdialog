import pandas as pd
from chatbot_website import create_app

app = create_app()

if __name__ == '__main__':
    # data = pd.DataFrame(columns = ["id", "image_url", "user_message", "bot_message", "feedback"])
    app.run(debug=True)
    
    # data.to_csv("response.csv")
