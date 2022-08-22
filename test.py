# from markupsafe import escape
# from flask import url_for
import nemo
# Import Speech Recognition collection
import nemo.collections.asr as nemo_asr
# Import Natural Language Processing colleciton
import nemo.collections.nlp as nemo_nlp

from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def translate():
    audiofile = request.files['audiofile']
    audio_path = "./audios/" + audiofile.filename
    audiofile.save(audio_path)
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_large")
    nmt_model = nemo_nlp.models.MTEncDecModel.from_pretrained(model_name="nmt_en_hi_transformer12x2")
    english_text = asr_model.transcribe([audiofile])
    
    hindi_text = nmt_model.translate(english_text)
    

    return render_template('index.html', translate=hindi_text)


# @app.route('/hello')
# def hello():
#     return "Hello, World!"

if __name__ == '__main__':
    app.run(port=5000, debug=True)


# URL Building:

# To build a URL to a specific function, use the url_for() function. 
# It accepts the name of the function as its first argument and any number of keyword arguments, 
# each corresponding to a variable part of the URL rule. Unknown variable parts are appended to the URL as 
# query parameters.

# @app.route('/')
# def index():
#     return 'index'

# @app.route('/login')
# def login():
#     return 'login'

# @app.route('/user/<username>')
# def profile(username):
#     return f'{username}\'s profile'

# with app.test_request_context():
#     print(url_for('index'))
#     print(url_for('login'))
#     print(url_for('login', next='/'))
#     print(url_for('profile', username='John Doe'))

