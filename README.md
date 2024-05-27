# aai
## Redirect 
|Exp|Link|  
|---|------| 
|1|[link]()|
|2|[link]()|
|3|[link]()|
|4|[link](https://github.com/bonobonoyaa/aai/tree/main?tab=readme-ov-file#ex-4---implementation-of-hidden-markov-model-)|
|5|[link](https://github.com/bonobonoyaa/aai/tree/main?tab=readme-ov-file#ex-5---implementation-of-kalman-filter)|
|6|[link](https://github.com/bonobonoyaa/aai/tree/main?tab=readme-ov-file#ex-6---implementation-of-semantic-analysis)|
|7|[link](https://github.com/bonobonoyaa/aai?tab=readme-ov-file#exp-7---implementation-of-text-summarization)|
|8|[link](https://github.com/bonobonoyaa/aai?tab=readme-ov-file#exp-8---implementation-of-speech-recognition)|

# [Ex 1 - Implementation of Bayesian Networks ]()
```py
def probs(data, child, parent1=None, parent2=None):
    if parent1==None:
        # Calculate probabilities
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent1!=None:
            # Check if child node has 1 parent or 2 parents
            if parent2==None:
                # Caclucate probabilities
                prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:
                # Caclucate probabilities
                prob=pd.crosstab([data[parent1],data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    return prob

bbn = Bbn() 
    .add_node(H9am) 
    .add_node(H3pm) 
    .add_node(W) 
    .add_node(RT) 
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) 
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) 
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))
```
# [Ex 2 - Implementation of Exact Inference Method of Bayesian Network ]()
```py
cpd_burglary = TabularCPD(variable='Burglary',variable_card=2,values=[[0.999],[0.001]])
cpd_earthquake = TabularCPD(variable='Earthquake',variable_card=2,values=[[0.998],[0.002]])
cpd_alarm = TabularCPD(variable ='Alarm',variable_card=2, values=[[0.999, 0.71, 0.06, 0.05],[0.001, 0.29, 0.94, 0.95]],evidence=['Burglary','Earthquake'],evidence_card=[2,2])
cpd_john_calls = TabularCPD(variable='JohnCalls',variable_card=2,values=[[0.95,0.1],[0.05,0.9]],evidence=['Alarm'],evidence_card=[2])
cpd_mary_calls = TabularCPD(variable='MaryCalls',variable_card=2,values=[[0.99,0.3],[0.01,0.7]],evidence=['Alarm'],evidence_card=[2])

network.add_cpds(cpd_burglary,cpd_earthquake,cpd_alarm,cpd_john_calls,cpd_mary_calls)

inference = VariableElimination(network)

evidence ={'JohnCalls':1,'MaryCalls':0}
query_variable ='Burglary'
result = inference.query(variables=[query_variable],evidence=evidence)
print(result)

evidence1 ={'JohnCalls':1,'MaryCalls':1}
query_variable ='Burglary'
result2 = inference.query(variables=[query_variable],evidence=evidence)
print(result2)
```
# [Ex 3 - Implementation of Approximate Inference in Bayesian Networks ]()
```py
cpd_burglary = TabularCPD(variable='Burglary',variable_card=2,values=[[0.999],[0.001]])
cpd_earthquake = TabularCPD(variable='Earthquake',variable_card=2,values=[[0.998],[0.002]])
cpd_alarm = TabularCPD(variable ='Alarm',variable_card=2, values=[[0.999, 0.71, 0.06, 0.05],[0.001, 0.29, 0.94, 0.95]],evidence=['Burglary','Earthquake'],evidence_card=[2,2])
cpd_john_calls = TabularCPD(variable='JohnCalls',variable_card=2,values=[[0.95,0.1],[0.05,0.9]],evidence=['Alarm'],evidence_card=[2])
cpd_mary_calls = TabularCPD(variable='MaryCalls',variable_card=2,values=[[0.99,0.3],[0.01,0.7]],evidence=['Alarm'],evidence_card=[2])


network.add_cpds(cpd_burglary,cpd_earthquake,cpd_alarm,cpd_john_calls,cpd_mary_calls)


gibbs_sampler =GibbsSampling(network)
num_samples=10000
samples=gibbs_sampler.sample(size=num_samples)
query_variable='Burglary'
query_result= samples[query_variable].value_counts(normalize=True)
print("\n Approximate Probabilities of {}".format(query_variable))
print(query_result)

```
# [Ex 4 - Implementation of Hidden Markov Model ](https://github.com/Rajeshkannan-Muthukumar/Ex-4--AAI/blob/main/ai_exp4.ipynb)
```py
for t in range (1,len(observed_sequence)):
  for j in range (len(initial_probabilities)):
    alpha[t,j]=emission_matrix[j,observed_sequence[t]]*np.sum(alpha[t-1,:]*transition_matrix[:,j])

probability = np.sum(alpha[-1,:])

print("The probability of the observed sequence is:",probability)

most_likely_sequence=[]
for t in range(len(observed_sequence)):
  if alpha[t,0] > alpha[t,1]:
    most_likely_sequence.append("sunny")
  else:
    most_likely_sequence.append("rainy")

print("The most likely sequence of weather states is:",most_likely_sequence)
```
# [Ex 5 - Implementation of Kalman Filter](https://github.com/Kaushika-Anandh/Ex-5--AAI/blob/main/exp5_AAI.ipynb)
```py
def predict(self):
    #predict the next state
    self.x = np.dot(self.F, self.x)
    self.P = np.dot(np.dot(self.F, self.P),self.F.T) + self.Q

  def update(self, z):
    #update the state estimate based on the measurement z
    y = z - np.dot(self.H, self.x)
    S = np.dot(np.dot(self.H, self.P),self.H.T) + self.R
    K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
    self.x = self.x + np.dot(K, y)


kf = KalmanFilter(F,H,Q,R,x0,P0)

true_states=[]
measurements=[]
for i in range(100):
  true_states.append([i*dt, 1]) #assume constant velocity of 1m/s
  measurements.append(i*dt + np.random.normal(scale=1)) # add measurement noise


# run the Kalman filter on the simulated measurements
est_states = []
for z in measurements:
    kf.predict()
    kf.update(np.array([z]))
    est_states.append(kf.x)
```
# Ex 6 - Implementation of Semantic Analysis
```py
!pip install nltk

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

f = open("samplefile.txt", "r")

sentences = f.readlines()
sentences
f.close()

verbs = [[] for _ in sentences]
i=0
for sentence in sentences:
  print("Sentence",i+1,":", sentence)

  # Tokenize the sentence into words
  words = word_tokenize(sentence)

  # Identify the parts of speech for each word
  pos_tags = nltk.pos_tag(words)

  # Print the parts of speech
  for word,tag in pos_tags:
    print(word,"->",tag)

    # Save verbs
    if tag.startswith('VB'):
      verbs[i].append(word)
  i+=1
  print("\n\n")

# Identify synonyms and antonyms for each word
print("Synonyms and Antonymns for verbs in each sentence:\n")
i=0
for sentence in sentences:
  print("Sentence",i+1,":", sentence)
  pos_tags = nltk.pos_tag(verbs[i])
  for word,tag in pos_tags:
    print(word,"->",tag)
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
      for lemma in syn.lemmas():
        synonyms.append(lemma.name())
        if lemma.antonyms():
          for antonym in lemma.antonyms():
            antonyms.append(antonym.name())

    # Print the synonyms and antonyms
    print("Synonyms:",set(synonyms))
    print("Antonyms:", set(antonyms) if antonyms else "None")
    print()
  print("\n\n")
  i+=1
```
# Exp 7 - Implementation of Text Summarization
```py
! pip install nltk


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
  # Tokenize the text into words
  words = word_tokenize (text)
  # Remove stopwords and punctuation
  stop_words = set(stopwords.words('english'))
  filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
  # Stemming
  stemmer = PorterStemmer()
  stemmed_words = [stemmer.stem (word) for word in filtered_words]
  return stemmed_words

def generate_summary (text, num_sentences=3):
  sentences = sent_tokenize(text)
  preprocessed_text = preprocess_text(text)
  # Calculate the frequency of each word
  word_frequencies = nltk. FreqDist(preprocessed_text)
  # Calculate the score for each sentence based on word frequency
  sentence_scores = {}
  for sentence in sentences:
    for word, freq in word_frequencies.items():
      if word in sentence.lower():
        if sentence not in sentence_scores:
          sentence_scores [sentence] = freq
        else:
          sentence_scores [sentence] += freq
  # Select top N sentences with highest scores
  summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True) [:num_sentences]
  return''.join(summary_sentences)

if __name__ == "__main__":
    input_text = """(Any text you'd like ig)"""
    summary = generate_summary(input_text)
    print("Original Text:")
    print(input_text)
    print("\nSummary:")
    print(summary)
```
# Exp 8 - Implementation of Speech Recognition
```py
! pip install SpeechRecognition
!pip install pyaudio

import pyaudio
import speech_recognition as sr

# initialize the Recognizer
r = sr.Recognizer()

#Set duration for audio capture
duration = 10

#Record audio
print("Say Something")

# USe the default microphone as the audio source
with sr.Microphone() as source:
    audio_data = r.listen(source, timeout=duration)

try:
    text = r.recognize_google(audio_data)
    print("you said:",text)
except sr.UnknownValueError:
    print("Sorry, could not understand audio")
except sr.RequestError as e:
    print(f'Error with the request to Google Speech Recognition Service: {e}')
except Exception as e:
    print(f'Error: {e}')
```
