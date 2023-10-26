import cv2
from tkinter import *
from deepface import DeepFace
from PIL import Image, ImageTk

# Loading Cascade file from the directory
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

url = "http://192.168.29.53:8080/video"
def openCam():

    cap = cv2.VideoCapture(url)

    while True:
        rat, frame = cap.read()

        # Reading Frame
        rat, frame = cap.read()
        
        # Flip the image horizontically 
        
        # Converting RGB image to Gray Scale image
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        
        # Detecting faces in image
        faces = face_cascade.detectMultiScale(gray,1.1,4)
    
        # Draw Ractangle Box around faces
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
            # Image of each face (Region of Interest)
            ROI = frame[y:y+h,x:x+w]
            
            # Prediction of Emotion using DeepFace
            prediction = DeepFace.analyze(ROI,actions=['emotion'],enforce_detection=False)
        
            # Put Text in the image
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame,str(prediction[0]['dominant_emotion']),(x,y),font,2,(0,0,255),2,cv2.LINE_4)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        video_frame['image'] = photo

        root.update()


root = Tk()
root.title("Emotion Detection App")
root.geometry('1200x700')


# Adding Video Frame
video_frame = Label(text='Image',bg="lightyellow")
video_frame.pack()

# Adding Button
button = Button(text="Open Camera",font=('','14'),bg='white',fg='black',borderwidth=2,padx=5,pady=5,command=openCam)
button.pack(pady=20)

root.mainloop()