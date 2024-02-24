from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager, current_user
from PIL import Image
import requests
from sqlalchemy import update
from sqlalchemy import schema, create_engine
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BartForConditionalGeneration, BartTokenizer
import torch 
import time

db = SQLAlchemy()
DB_NAME = "database_cb.db"

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sum_model_name = "facebook/bart-large-cnn"
sum_model = BartForConditionalGeneration.from_pretrained(sum_model_name)
sum_tokenizer = BartTokenizer.from_pretrained(sum_model_name)

def create_app():
    
    app = Flask(__name__)
    # app.config['UPLOAD_FOLDER'] = "static/uploads"
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    from .views import views
    from .auth import auth
    
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Chat
    
    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    @app.route("/get", methods=["GET", "POST"])
    def getResponse():
        # global data
        id = request.form.get('id')
        url = request.form.get('url')
        user_msg = request.form.get('msg')

        # if 'image' in request.files:
        #     img = request.files['image']
        #     img.save("uploaded_image.jpg")
        #     print("saved image")
        #img = request.files["image"]
        #image_input = img
        #img = Image.open("test/20221218_205725.jpg").convert('RGB')
        print("opened img")
        #img = Image.fromarray(img).convert('RGB')
        print("url: ", url)
        with app.app_context():
            history = Chat.query.filter_by(img_url = url, user_id = current_user.id).all()

        
        print(user_msg)



        if len(history)<3:
            input = " ".join([h.user_msg + " " + h.bot_msg for h in history])
        else:
            input = " ".join([h.user_msg + " " + h.bot_msg for h in history[-3:]])
        
        input = input + " " + user_msg

        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        print("opened img")
        #img = Image.fromarray(img).convert('RGB')
        start = time.time()
        bot_msg = get_Chat_response(input, img)
        end = time.time()
        runtime = start - end
        print("run time", end - start)
        #feedback = request.form.get('feedback')

        new_chat = Chat(msg_timestamp = str(id),
                        img_url=url,
                        user_msg=user_msg,
                        bot_msg=bot_msg,
                        runtime = runtime,
                        #feedback=feedback,
                        user_id=current_user.id)
        db.session.add(new_chat) #adding the chat to the database 
        db.session.commit()
        
        new = {
            "id": id,
            "image_url": url,
            "user_message": user_msg,
            "bot_message": bot_msg,
            "feedback": None
        }
        #data = data.append(new, ignore_index = True)
        # data = pd.concat([data, pd.DataFrame([new])], ignore_index=True)

        return bot_msg
    
    def summarize_text(model, tokenizer, input_text, max_length=100):
        # Tokenize and generate summary
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(input_ids, max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def get_Chat_response(text, img):

        # Let's chat for 5 lines
        #for step in range(1000):
            #time.sleep(5)
            global history
            inputs = processor(images=img, text=text, return_tensors="pt").to(model.device)
            #print("inputs:", inputs)
            print("generating output.........")
        
            outputs = model.generate(
                                        **inputs,
                                        do_sample=False,
                                        num_beams=2,
                                        max_length= 256,
                                        min_length=1,
                                        #top_p=0.9,
                                        repetition_penalty=1.5,
                                        length_penalty=2.0,
                                        temperature=1,
                                    )
            
        
            print("generated!!!!!!!!!!")
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            if len(generated_text)>100:
                sum_text = summarize_text(sum_model, sum_tokenizer, generated_text)
                print("summairised ans", sum_text)
            else: 
                sum_text = generated_text
            history[-1].append(sum_text)
            torch.cuda.empty_cache()
            #response = model.generate({"image": image, "prompt": prompt})[0]
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            #print(generated_text)
            
            
            # pretty print last ouput tokens from bot
            return generated_text
            #generated_text

    @app.route("/sendfeedback", methods=["GET", "POST"])
    def saveFeedback():
        # global data
        feedback = request.form["feedback"]
        id = request.form["id"]
        with app.app_context():
            update = Chat.query.filter_by(msg_timestamp = str(id), user_id = current_user.id).first()
           
            if update:
                new_chat = Chat(id = update.id,
                                msg_timestamp = str(id),
                                img_url=update.img_url,
                                user_msg=update.user_msg,
                                bot_msg=update.bot_msg,
                                feedback=feedback,
                                date = update.date,
                                user_id=current_user.id)
                db.session.delete(update)
                db.session.commit()
                db.session.add(new_chat) #adding the chat to the database 
                db.session.commit()
        # data.loc[int(id) - 1, 'feedback'] = feedback
        print("hiasdsaf feedback", feedback)
        return "feedback saved!"
    
    return app

def create_database(app):
    if not path.exists('chatbot_website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')
