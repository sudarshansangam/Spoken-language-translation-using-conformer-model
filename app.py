from fnmatch import translate
from flask import Flask, render_template, request, redirect
import nemo
# # Import Speech Recognition collection
import nemo.collections.asr as nemo_asr
# # Import Natural Language Processing colleciton
import nemo.collections.nlp as nemo_nlp
#import speech_recognition as sr

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    translate = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
            
        if file:
            asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_large")
            path = "/home/psg/Desktop/NMT_new/GUI_Model/sample_audios/"
            file.filename = path + file.filename
            transcript = asr_model.transcribe([file.filename])
            transcript = list(transcript[0])
            
            nmt_model = nemo_nlp.models.MTEncDecModel.from_pretrained(model_name="nmt_en_hi_transformer12x2")
            translate = nmt_model.translate(transcript)
            #print(str(translate[0]))
            
            '''Google Speech recognition model
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)'''


    return render_template('index.html', transcript="".join(transcript), translate="".join(translate))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
