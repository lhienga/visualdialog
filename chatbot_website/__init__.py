from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager, current_user
from PIL import Image
import requests
from sqlalchemy import update
from sqlalchemy import schema, create_engine
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BartForConditionalGeneration, BartTokenizer, InstructBlipConfig, AutoModelForVision2Seq
import torch
import torch 
import time
from accelerate import infer_auto_device_map, init_empty_weights

db = SQLAlchemy()
DB_NAME = "database_cb.db"

def create_app(args):
    print("CREATING APPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
    app = Flask(__name__)
    # app.config['UPLOAD_FOLDER'] = "static/uploads"
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    # Load the model configuration.
    config = InstructBlipConfig.from_pretrained("Salesforce/instructblip-vicuna-13b")

    # Initialize the model with the given configuration.
    with init_empty_weights():
        model = AutoModelForVision2Seq.from_config(config)
        model.tie_weights()

    # Infer device map based on the available resources.
    device_map = infer_auto_device_map(model, max_memory={0: "20GiB", 1: "20GiB", 2:"20GiB"},
                                    no_split_module_classes=['InstructBlipEncoderLayer', 'InstructBlipQFormerLayer',
                                                                'LlamaDecoderLayer'])
    device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model'
                                                                                        '.embed_tokens')]

    offload = "offload"
    # Load the processor and model for image processing.
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b", device_map="auto")
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b",
                                                                device_map=device_map,
                                                                offload_folder=offload, offload_state_dict=True)

    # Summarisation model
    sum_model_name = "facebook/bart-large-cnn"
    sum_model = BartForConditionalGeneration.from_pretrained(sum_model_name)
    sum_tokenizer = BartTokenizer.from_pretrained(sum_model_name)


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
            history = Chat.query.filter_by(img_url = url, user_id = current_user.id, prompt_num = args.len_history, summarise = args.sum).all()

        print(user_msg)


        if args.len_history == 0:
            input = user_msg
        else:
            if len(history) < args.len_history:
                input = " ".join([h.sum_msg for h in history])
                #input = " ".join([h.user_msg + " " + h.sum_msg for h in history])
            else:
                input = " ".join([h.sum_msg for h in history[-args.len_history:]])
                #input = " ".join([h.user_msg + " " + h.sum_msg for h in history[-args.len_history:]])
            
            input = input + " " + user_msg
        print("prompt: ", input)

        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        print("opened img")
        #img = Image.fromarray(img).convert('RGB')
        start = time.time()
        bot_msg , sum_text = get_Chat_response(input, img)
        end = time.time()
        runtime = end-start
        print("run time", end - start)
        #feedback = request.form.get('feedback')

        new_chat = Chat(msg_timestamp = str(id),
                        img_url=url,
                        user_msg=user_msg,
                        bot_msg=bot_msg,
                        prompt = input, 
                        prompt_num =  args.len_history,
                        summarise = args.sum, 
                        sum_msg = sum_text,
                        runtime = runtime,
                        #feedback=feedback,
                        user_id=current_user.id)
        db.session.add(new_chat) #adding the chat to the database 
        db.session.commit()
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

            inputs = processor(images=img, text=text, return_tensors="pt").to(model.device)
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

            if args.sum == 1: 
                if len(generated_text)>100:
                    sum_text = summarize_text(sum_model, sum_tokenizer, generated_text)
                    print("summarised ans", sum_text)
                else: 
                    sum_text = generated_text
            else:
                sum_text = generated_text
            #history[-1].append(sum_text)
            torch.cuda.empty_cache()
            
            # pretty print last ouput tokens from bot
            return generated_text, sum_text
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
                                sum_msg = update.sum_msg,
                                prompt = update.prompt, 
                                prompt_num = update.prompt_num, 
                                summarise = update.summarise, 
                                feedback=feedback,
                                runtime = update.runtime, 
                                date = update.date,
                                user_id=current_user.id)
                db.session.delete(update)
                db.session.commit()
                db.session.add(new_chat) #adding the chat to the database 
                db.session.commit()
        # data.loc[int(id) - 1, 'feedback'] = feedback
        print("feedback", feedback)
        return "feedback saved!"
    
    return app

def create_database(app):
    if not path.exists('chatbot_website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')
