import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.datasets import imdb

# model parametreleri
max_features=20000  # en cok kullanilan 20 bin kelime
maxlen=500 # rnn modelinin beklediği sabit giriş uzunluğu : input_length

# stopword kurtulma ve sozlukleri hazirlama
nltk.download("stopwords")
stop_words=set(stopwords.words("english"))

# imdb veri setinden kelime -> index sozlugunu alalim
word_index=imdb.get_word_index()

# sayi -> kelime ve kelime -> sayi sozluklerini hazirlama

index_to_word={index+3 : word for word, index in word_index.items()}
index_to_word[0]='<PAD>'
index_to_word[1]='<START>'
index_to_word[2]='<UNK>'

word_to_index={word: index for index, word in index_to_word.items()}

# egitim modelini yukleme
model=load_model("rnn_imdb_model.h5")
print("model yuklendi.")

# tahmin yapma fonksiyonu 

def predict_review(text):

    """
         kullanican gelen metni temizle, modele uygun hale getir ve tahmin sonucunu yazdir.
    """

    # yorumun kucuk harfli kelime listesine cevir.
    words=text_to_word_sequence(text) 

    # stopword olmayan kelimeleri al

    cleaned=(

        word for word in words if word.isalpha() and word.lower() not in stop_words
    )

    # her kelimeyi index'e cevir
    encoded=[word_to_index.get(word,2) for word in cleaned] # bilinmeyen kelimeler 2 (<UNK>) ile kodlanir

    # modelin bekledigi sabit uzunlukta yap
    padded=pad_sequences([encoded], maxlen=maxlen)

    # tahmin yapalim -> prediction 0-1 arasi deger doner
    prediction=model.predict(padded)[0][0]

    # sonuclari yazdir
    print(f"Pozitif tahmin olasiligi: {prediction:.4f}%")
    if prediction >=0.5:
        print("Pozitif.")
    else:
        print("Negatif.")


# kullanıcıdan yorum al ve tahmin yap
user_review=input("Lutfen bir film yorumu giriniz:\n")
predict_review(user_review)