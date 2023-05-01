import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import load_model
import pandas as pd

#LOADING SONGS
songs = pd.read_csv("data_moods.csv")
rec_songs = songs[["name", "artist", "mood", "popularity"]]

# LOADING THE MODEL

model = load_model("detection_model.h5")


from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)



def face_extraction(frame):

    ''' Detect faces in a frame and extract them '''

    faces = cascade_model.detectMultiScale(frame, 1.1, 5)

    for x, y, w, h in faces:
        frame = frame[y:y+h, x:x+w]

    return frame





def image_processing(frame):

    ''' Preprocessing of the image for predictions '''
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (48, 48))
    frame = image.img_to_array(frame)
    frame = frame/255
    frame = np.expand_dims(frame, axis=0)

    return frame




def detect_expressions(frame, detection_model):

    ''' Detect final expressions and return the predictions
        done by the detection_model '''

    #cropped_frame = face_extraction(frame)

    #test_frame = image_processing(cropped_frame)
    test_frame = image_processing(frame)

    prediction = model.predict(test_frame)

    return prediction


def recommend_songs(songs_data, emotion):
    final_df = pd.DataFrame()
    
    if emotion == [0]:
        filter_0 = songs_data["mood"] == 'Calm'
        f0 = filter_0
        df0 = songs_data.where(f0).dropna()
        df0.sort_values(by="popularity", ascending=False, inplace=True)
        final_df = pd.concat([final_df, df0.head()])
    
    elif emotion == [1]:
        filter_1 = songs_data["mood"] == 'Happy'
        f1 = filter_1
        df1 = songs_data.where(f1).dropna()
        df1.sort_values(by="popularity", ascending=False, inplace=True)
        final_df = pd.concat([final_df, df1.head()])
        
    elif emotion == [2]:
        filter_2 = songs_data["mood"] == 'Sad'
        f2 = filter_2
        df2 = songs_data.where(f2).dropna()
        df2.sort_values(by="popularity", ascending=False, inplace=True)
        final_df = pd.concat([final_df, df2.head()])
        
    else:
        filter_3 = songs_data["mood"] == 'Energetic'
        f3 = filter_3
        df3 = songs_data.where(f3).dropna()
        df3.sort_values(by="popularity", ascending=False, inplace=True)
        final_df = pd.concat([final_df, df3.head()])
    
    
    return final_df

def store_recommendations(recommendations):
    final_recommendations = pd.DataFrame()
    final_recommendations = pd.concat([final_recommendations, recommendations])
    return final_recommendations.to_csv("recommendations.csv", mode='a')


# LOADING HAAR CASCADE CLASSIFIER

cascade_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")




def generate_frames():
    while True:
            
        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            faces = cascade_model.detectMultiScale(frame, 1.1, 5)
        
            for x, y, w, h in faces:

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)    
                
                prediction = detect_expressions(frame, model)
        
                out = np.argmax(prediction) 
                
                recommendations = recommend_songs(rec_songs, out)
                #recommendations.to_csv("recommendations.csv", mode='a')
                store_recommendations(recommendations)

                font = cv2.FONT_ITALIC
                
                if out == [0]:
                    cv2.putText(frame, "Angry: " + str(np.ceil(prediction[0, out] * 100)) + "%" , (x, y), font, 0.5, (0, 0, 255), 2)

                elif out == [1]:
                    cv2.putText(frame, "Happy: " + str(np.ceil(prediction[0, out] * 100)) + "%", (x, y), font, 0.5, (0, 0, 255), 2)

                elif out == [2]:
                    cv2.putText(frame, "Sad: " + str(np.ceil(prediction[0, out] * 100)) + "%", (x, y), font, 0.5, (0, 0, 255), 2)

                else:
                    cv2.putText(frame, "Surprised: "+ str(np.ceil(prediction[0, out] * 100)) + "%", (x, y), font, 0.5, (0, 0, 255), 2)

                    
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/table')
def table():
    # converting csv to html
    rec_df = pd.read_csv("recommendations.csv")
    rec_df.drop_duplicates(inplace=True)
    return render_template('table.html', tables=[rec_df.to_html()], titles=[''])


    
if __name__=="__main__":
    app.run(debug=False)