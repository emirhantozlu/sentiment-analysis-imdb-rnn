# import libraries
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential # base model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# stopword listeni belirle
nltk.download('stopwords') # nltk içinden  stopword veritabanini indir
stop_words = set(stopwords.words('english'))
 
# model parametreleri
max_features = 20000  # en cok kullanilan 20 bin kelime
maxlen = 500  # her yorum icin max kelime sayisi

# load dataset
(X_train, y_train), (X_test, y_test)=imdb.load_data(num_words=max_features) # train ve test ayrilmis sekilde yuklenir.

# ornek veri inceleme
original_word_index=imdb.get_word_index()

# sayi kelime donusumu icin sozlugu hazirlama
inv_word_index={index + 3 :word for word,index in original_word_index.items()}
inv_word_index[0]='<PAD>' # 0: bosluk padding
inv_word_index[1]='<START>' # 1: cümle baslangici
inv_word_index[2]='<UNK>' # 2: bilinmeyen kelime


# sayi dizisini kelimelere ceviren fonksiyon
def decode_review(encoded_review):
    return ' '.join([inv_word_index.get(i, '?') for i in encoded_review])

# ilk egitim verisini yazdiralim
movie_index=0
print("ilk yorum: (sayi dizisi olarak)")
print(X_train[movie_index])

print("\nilk yorum: (kelime olarak)")
print(decode_review(X_train[movie_index]))

print(f"label: {'Pozitif' if y_train[movie_index] == 1 else 'Negatif'}")

# gerekli sozluklerin olusturlmasi : word to index ve index to word
word_index=imdb.get_word_index()
index_to_word={index+3 :word for word, index in word_index.items()}
index_to_word[0]='<PAD>'
index_to_word[1]='<START>'
index_to_word[2]='<UNK>'
word_to_index={word:index for index, word in index_to_word.items()}

# data preprocessing 

def preprocess_data(encoded_reviews):
    #sayilari kelimelere cevir.
    words=[index_to_word.get(i,"?") for i in encoded_reviews if i >=3]

    # sadece harflerden olusan ve stopword olmayan kelimeleri al
    cleaned=[
        word.lower() for word in words if word.isalpha and word.lower not in stop_words
    ]

    # kelimeleri tekrar sayilara cevir
    return [word_to_index.get(word,2) for word in cleaned]

# veriyi temizle ve sabit uzunlugu pad et

X_train=[preprocess_data(review) for review in X_train]
X_test=[preprocess_data(review) for review in X_test]

# pad sequence: Rnn modelleri sabit uzunlukta girdi ister. bu nedenle yorumlari maxlen uzunlugunda pad ediyoruz.
"""
merhaba -> [merhaba, <PAD>, <PAD>, <PAD>, <PAD>]  (maxlen =5)
merhaba nasilsin -> [merhaba, nasilsin, <PAD>, <PAD>, <PAD>]  (maxlen =5)
"""
X_train=pad_sequences(X_train, maxlen=maxlen)
X_test=pad_sequences(X_test, maxlen=maxlen)

# rnn modeli olusturma
model=Sequential() # base model: katmanlari sirasiyla eklememizi saglar

# embedding katmani: kelime indexlerini 32 boyutlu vektorlere cevirir
model.add(Embedding(input_dim=max_features, output_dim=32, input_length=maxlen))

# output katmani: binary classification : sigmoid -> 0-1 arasi deger verir
model.add(SimpleRNN(32)) # 32 birimli rnn katmani
model.add(Dense(1,activation="sigmoid"))

# model compile

model.compile(
    optimizer="adam", # agirliklari guncelleme icin kullanilan algoritma
    loss="binary_crossentropy", # binary classification icin uygun kayip fonksiyonu
    metrics=["accuracy"] # egitim ve degerlendirme sirasinda izlenecek metrik
)

print(model.summary())

# training the model
history=model.fit(
    X_train,y_train,
    epochs=3,
    batch_size=64, # torba boyutu : her iterasyonda 64 ornek kullanilir
    validation_split=0.2
)

# model evaluation
def plot_history(hist):
    plt.figure(figsize=(12,4))

    # accuracy plot
    plt.subplot(1,2,1) # 1 satir 2 sutunluk plota 1. subplot
    plt.plot(hist.history["accuracy"], label="training Accuracy")
    plt.plot(hist.history["val_accuracy"], label="validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()


    # loss plot
    plt.subplot(1,2,2)
    plt.plot(hist.history["loss"], label="training loss")
    plt.plot(hist.history["val_loss"], label="validation loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout() # subplotlar arasindaki bosluklari ayarlar
    plt.show()


plot_history(history)

# test verisiyle modeli degerlendirme
test_loss,test_accuracy=model.evaluate(X_test,y_test)
print(f"Test accuracy: {test_accuracy:.2f}")


# modeli kaydetme
model.save("rnn_imdb_model.h5")
print("Model kaydedildi: rnn_imdb_model.h5")