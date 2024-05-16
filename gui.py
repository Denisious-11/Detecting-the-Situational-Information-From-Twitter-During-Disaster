#importing necessary libraries
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import numpy as np
import pickle
from tensorflow.keras.models import load_model #load  model
import re
import nltk
from nltk.corpus import stopwords   #to get collection of stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.wordnet import WordNetLemmatizer
import string
from transformers import TFRobertaModel
from tensorflow.keras.models import model_from_json
from textblob import TextBlob


english_stop_words=set(stopwords.words('english')) 
lemma = WordNetLemmatizer() 

# #load trained model
loaded_model=load_model("Models/model.h5",compile=False,custom_objects={"TFRobertaModel":TFRobertaModel}) #unknown layer error#custom_objects={"TFRobertaModel":TFRobertaModel},



#load tokenizer
with open("Models/tokenizer.pickle",'rb') as handle:
    tokenizer=pickle.load(handle)


with open('Models/max_length.txt','r') as fo:
    max_len=int(fo.read())

#creating window
a = Tk()
a.title("Situational Tweets Identifier")
a.geometry("1000x650")
a.maxsize(1000, 650)
a.minsize(1000, 650)


def tag_remove(text):
    clean=re.compile('<.*?>')
    cleantext=re.sub(clean,'',text)
    return cleantext

def do_pre1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text

def do_pre2(text):
    text=re.sub('[''"",,,]','',text)
    text=re.sub('\n','',text)
    return text

def tokenize_roberta(data,max_len=max_len):
    encoded = tokenizer.batch_encode_plus([data],
                                            is_split_into_words=True,
                                            truncation=True,
                                            add_special_tokens=True,
                                            max_length=max_len,
                                            padding='max_length',
                                            return_attention_mask=True)

    return np.array(encoded['input_ids']),np.array(encoded['attention_mask'])

# def get_sentiment(sentence):
#     # Create a TextBlob object
#     blob = TextBlob(sentence)
    
#     # Get the sentiment polarity
#     sentiment_polarity = blob.sentiment.polarity
    
#     # Classify sentiment
#     if sentiment_polarity > 0:
#         sentiment = "Positive"
#     elif sentiment_polarity < 0:
#         sentiment = "Negative"
#     else:
#         sentiment = "Neutral"
    
#     return sentiment



def prediction():
    
    ini_text=text1.get("1.0",'end')
    if ini_text=='' or ini_text=='\n':
        message.set("Fill the empty field!!!")
    else:
        get_my_l_box.insert(1, "")
        get_my_l_box.insert(2, "Input Tweet :")
        get_my_l_box.insert(3, ini_text)

        message.set("")


        text=ini_text

        #perform preprocessing
        tag_removed_txt=tag_remove(text)
        preprocessed1_txt=do_pre1(tag_removed_txt)
        preprocessed2_txt=do_pre2(preprocessed1_txt)
        print(preprocessed2_txt)
        print("*********************")

        remove_stopwords=[w for w in preprocessed2_txt.split() if w not in english_stop_words]
        apply_lemmatization=[lemma.lemmatize(word) for word in remove_stopwords]

        get_my_l_box.insert(4, "\nPreprocessed Text : ")
        get_my_l_box.insert(5, apply_lemmatization)
        print(apply_lemmatization)

        #perform word embedding
        my_input_id, my_attention_mask = tokenize_roberta(apply_lemmatization, max_len)

        get_my_l_box.insert(6, "\nFeature Extraction : ")
        get_my_l_box.insert(7, my_input_id)
        get_my_l_box.insert(8, my_attention_mask)
        get_my_l_box.insert(9, "\nLoading Trained Model...")
        get_my_l_box.insert(10, "")
        get_my_l_box.insert(11, "\nPrediction")
        prediction = loaded_model.predict([my_input_id,my_attention_mask])
        get_my_l_box.insert(12, "")
        get_my_l_box.insert(13, prediction)
        prediction=np.argmax(prediction)
        get_my_l_box.insert(14, "")
        get_my_l_box.insert(15, prediction)
        print(prediction)
        # sentiment_result = get_sentiment(text)
        # print("\nSentiment : ",sentiment_result)


        if prediction==0:
            print("Non-Situational")
            output="Non-Situational"
        elif prediction==1:
            print("Situational")
            output="Situational"

        get_my_l_box.insert(16, "\nResult")
        get_my_l_box.insert(17, output)
        r_label.config(text="Output : "+output)
        # messagebox.showinfo("Predicted Sentiment", sentiment_result)



def Check():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="#EDCBD2")
    f1.place(x=0, y=0, width=760, height=390)
    f1.config()

    input_label = Label(f1, text="INPUT", font="arial 16", bg="#EDCBD2")
    input_label.pack(padx=0, pady=10)

    
    global message
    message = StringVar()

    global text1
    text1=Text(f1,height=12,width=70)
    text1.pack()


    msg_label = Label(f1, text=
        "", textvariable=message,
                      bg='#EDCBD2').place(x=330, y=255)##EDCBD2

    predict_button = Button(
        f1, text="Predict", command=prediction, bg="light blue")
    predict_button.pack(side="bottom", pady=30)
    global f2
    f2 = Frame(f, bg="#80C4B7")
    f2.place(x=0, y=390, width=760, height=500)
    f2.config(pady=20)

    result_label = Label(f2, text="RESULT", font="arial 16", bg="#80C4B7")
    result_label.pack(padx=0, pady=0)

    global r_label
    r_label = Label(f2, text="", bg="#80C4B7", font="arial 18 bold")
    r_label.pack(pady=70)

    f3 = Frame(f, bg="#E3856B")
    f3.place(x=760, y=0, width=240, height=690)
    f3.config()

    name_label = Label(f3, text="Process", font="arial 14", bg="#E3856B")
    name_label.pack(pady=20)

    global get_my_l_box
    get_my_l_box = Listbox(f3, height=18, width=31)
    get_my_l_box.pack()



def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="light goldenrod")
    f.pack(side="top", fill="both", expand=True)

    front_image1 = Image.open("Project_Extra/home.png")
    front_photo1 = ImageTk.PhotoImage(front_image1.resize((1000, 650), Image.ANTIALIAS))
    front_label1 = Label(f, image=front_photo1)
    front_label1.image = front_photo1
    front_label1.pack()

    home_label = Label(f, text="Situational Tweets Identifier",
                       font="arial 35", bg="white")
    home_label.place(x=220, y=280)


f = Frame(a, bg="light goldenrod")
f.pack(side="top", fill="both", expand=True)

front_image1 = Image.open("Project_Extra/home.png")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((1000, 650), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

home_label = Label(f, text="Situational Tweets Identifier",
                   font="arial 35", bg="white")
home_label.place(x=220, y=280)

m = Menu(a)
m.add_command(label="Home", command=Home)
checkmenu = Menu(m)
m.add_command(label="Check", command=Check)
plotmenu=Menu(m)
a.config(menu=m)


a.mainloop()
