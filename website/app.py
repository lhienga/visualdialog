from flask import Flask, render_template, request, jsonify
from flask import Flask, flash, request, redirect, url_for, render_template
import os
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig, AutoModelForVision2Seq
import torch
from PIL import Image
from werkzeug.utils import secure_filename
import requests
import pandas as pd
import time

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Path to save the CSV file
CSV_FILE_PATH = "response.csv"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/getimg', methods=['POST'])
def upload_image():
	if 'image' not in request.files:
		print('No file part')
		return redirect(request.url)
	file = request.files['image']
	if file.filename == '':
		print('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		print('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		print('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)
      
@app.route("/sendfeedback", methods=["GET", "POST"])
def saveFeedback():
    global data
    global start
    fb = request.form["feedback"]
    id = int(request.form["id"])
    data.loc[start+id, 'feedback'] = fb
    data.to_csv(CSV_FILE_PATH, index=False)  # Save to CSV
    print("hiasdsaf feedback", fb)
    return "feedback saved!"

@app.route("/get", methods=["GET", "POST"])
def chat():
    global data
    global history
    global last_url
    global start
    id = request.form["id"]
    msg = request.form["msg"]
    url = request.form["url"]
    if url!= last_url:
      history = []
      start = data.shape[0] -1
    last_url = url
    
    if 'image' in request.files:
        img = request.files['image']
        img.save("uploaded_image.jpg")
        print("saved image")
    print("url: ", url)
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    print(msg)
    if len(history)<3:
      input = " ".join([" ".join(x) for x in history])
    else:
      input = " ".join([" ".join(x) for x in history[-3:]])
    input = input + " " + msg
    history.append([msg])
    #img = request.files["image"]
    #image_input = img
    #img = Image.open("test/20221218_205725.jpg").convert('RGB')
    print("opened img")
    #img = Image.fromarray(img).convert('RGB')
    start = time.time()
    ans = get_Chat_response(input, img)
    end = time.time()
    print("run time", end - start)
    
    new = {"id": id, "image_url": url, "user_message": msg, "bot_message": ans, "feedback": None, "runtime":end-start}
    #data = data.append(new, ignore_index = True)
    data = pd.concat([data, pd.DataFrame([new])], ignore_index=True)
    data.to_csv(CSV_FILE_PATH, index=False)  # Save to CSV
    return ans


def get_Chat_response(text, img):

    # Let's chat for 5 lines
    #for step in range(1000):
        #time.sleep(5)
        global history
        inputs = processor(images=img, text=text, return_tensors="pt").to(model.device)
        print("inputs:", inputs)
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
        history[-1].append(generated_text)
        torch.cuda.empty_cache()
        #response = model.generate({"image": image, "prompt": prompt})[0]
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        #print(generated_text)
        
        
        # pretty print last ouput tokens from bot
        return generated_text
    #generated_text


if __name__ == '__main__':
    global data
    global start
    global history
    global last_url
    history = []
    last_url = ""
    data = pd.DataFrame(columns=["id", "image_url", "user_message", "bot_message", "feedback", "runtime"])

    # Load data from existing CSV file if available
    if os.path.exists(CSV_FILE_PATH):
        data = pd.read_csv(CSV_FILE_PATH)
    start = data.shape[0] -1
    app.run()

    # Save the final data to CSV before the program ends
    data.to_csv(CSV_FILE_PATH, index=False)
