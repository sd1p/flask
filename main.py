from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
#from os import path
import json
from flask_cors import CORS
from urllib.parse import urlparse
from urllib.parse import parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
import json
import re
import pandas as pd


import spacy
import nltk
nltk.download('stopwords')
nltk.download('popular')
from sorcery import dict_of
from summarizer import Summarizer

import pprint #pretty print
import itertools
import re
import pke
import string
from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize
import flashtext
from flashtext import KeywordProcessor #FlashText is a Python library created specifically for the purpose of searching and replacing words in a document.

import requests
import json
import re
import random
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import cosine_lesk
from nltk.corpus import wordnet as wn

from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd 

def mainModel(context):
  
  full_text = context

  model = Summarizer()
  result = model(full_text, min_length=60, max_length=500, ratio=0.7)

  summ_text = ''.join(result)
  print(summ_text)


  def get_nouns(text):
    P_nouns=[]
    extractor = pke.unsupervised.MultipartiteRank()
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.load_document(input=text, stoplist=stoplist)
    pos = {'NOUN'} ##Here we using parts of speech as proper noun
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
    keyphrases = extractor.get_n_best(n=20)
    for key in keyphrases:
      P_nouns.append(key[0])

    return P_nouns


  keywords = get_nouns(full_text)
  print(keywords)

  summ_keys=[]
  for keyword in keywords:
    if keyword.lower() in summ_text.lower():
      summ_keys.append(keyword)
  
  print(summ_keys)


  def tokenize_sent(text):
    sentences = [sent_tokenize(text)] ## sent_tokenize already ek list deta hai to yaha par 2d list ban ja rhi hai
    sentences = [y for x in sentences for y in x] 
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 12]  ##remove sentences with length less than 20
    return sentences

  def get_keyword_sentences(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
      keyword_sentences[word] = []
      keyword_processor.add_keyword(word)
    for sentence in sentences:
      keyword_found = keyword_processor.extract_keywords(sentence)
      for key in keyword_found:
        keyword_sentences[key].append(sentence)
    for key in keyword_sentences.keys():
      values = keyword_sentences[key]
      values = sorted(values,  key=len, reverse=True) ##reverse is for ascending descending
      keyword_sentences[key] = values
    return keyword_sentences

  sentences = tokenize_sent(summ_text)
  keyword_sentence_mapping = get_keyword_sentences(summ_keys, sentences)

  print(keyword_sentence_mapping)

  for keyword in keyword_sentence_mapping:
    if len(keyword_sentence_mapping[keyword])>0:  
      print(keyword_sentence_mapping[keyword][0])


  def get_distractors(syn,word):
    distractors=[]
    word=word.lower()
    og_word = word
    if len(word.split())>0:
      word=word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
      return distractors
    for item in hypernym[0].hyponyms():
      name = item.lemmas()[0].name()
      if name == og_word:
        continue
      name = name.replace("_"," ")
      name = " ".join(w.capitalize() for w in name.split())
      if name is not None and name not in distractors:
        distractors.append(name)
    return distractors

  def get_wordsense(sent,word):
      word= word.lower()
    
      if len(word.split())>0:
          word = word.replace(" ","_")

      synsets = wn.synsets(word,'n')
      if synsets:
          wup = max_similarity(sent, word, 'wup', pos='n')
          adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
          lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
          return synsets[lowest_index]
      else:
          return None

  key_distractor_list = {}

  for keyword in keyword_sentence_mapping:
      if len(keyword_sentence_mapping[keyword])>0:
        wordsense = get_wordsense(keyword_sentence_mapping[keyword][0],keyword)
        if wordsense:
            distractors = get_distractors(wordsense,keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors
                #print(key_distractor_list)
  questions=[]
  index = 1
  for quest in key_distractor_list:

    sentence = keyword_sentence_mapping[quest][0]
    pattern = re.compile(quest, re.IGNORECASE)
    output = pattern.sub( " _______ ", sentence)
    print (index,")",output)
    if quest.capitalize() in key_distractor_list[quest]:
      key_distractor_list[quest].remove(quest.capitalize())
    choices = [quest.capitalize()] + key_distractor_list[quest]
    top4choices = choices[:4]
    random.shuffle(top4choices)
    optionchoices = ['a','b','c','d']
    for idx,choice in enumerate(top4choices):
        print ("\t",optionchoices[idx],")"," ",choice)
    print ("\nMore options: ", choices[4:20],"\n\n")
    id=index
    question=output
    options=top4choices
    answer=options.index(quest.capitalize())
    res_dict = dict_of(id,question,options,answer)
    #print(id,options,question,answer)
    questions.append(res_dict)
    #print(res_dict)
    index = index + 1
  print(questions) 
  return questions   
    
mainModel("Mahatma Gandhi, also known as Mohandas Karamchand Gandhi, was an Indian independence leader who played a significant role in India's struggle for freedom from British rule. He was born on October 2, 1869, in Porbandar, Gujarat, India. Gandhi was a lawyer by profession, but he is best known for his non-violent civil disobedience approach to achieving political and social change. He began his political activism in South Africa, where he fought against discriminatory laws against Indians. He later returned to India in 1915 and became the leader of the Indian National Congress, which was at the forefront of the independence movement. Gandhi's philosophy of non-violent resistance, known as Satyagraha, influenced many movements for civil rights and freedom around the world, including the American civil rights movement led by Martin Luther King Jr. He believed that non-violent protest could achieve social and political change without resorting to violence or aggression. Gandhi's most significant contribution to India's struggle for independence was the Salt March of 1930. The British had imposed a salt tax, making it illegal for Indians to produce or sell salt. Gandhi led a 240-mile march to the Arabian Sea to collect salt, which galvanized the Indian people and brought international attention to their cause. Gandhi's legacy continues to inspire people around the world. He believed in the power of the individual to bring about change, and his teachings on non-violence and civil disobedience continue to influence social and political movements to this day. He was assassinated on January 30, 1948, but his ideas and philosophy live on, and he remains one of the most revered and respected figures in modern history.")

##########################PUNCHUATIONS####################

import os
import yaml
import torch
from torch import package

torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml',
                               progress=False)

with open('latest_silero_models.yml', 'r', encoding='utf8') as yaml_file:
    models = yaml.load(yaml_file, Loader=yaml.SafeLoader)
model_conf = models.get('te_models').get('latest')
model_url = model_conf.get('package')

model_dir = "downloaded_model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, os.path.basename(model_url))

if not os.path.isfile(model_path):
    torch.hub.download_url_to_file(model_url,
                                   model_path,
                                   progress=True)

imp = package.PackageImporter(model_path)
model = imp.load_pickle("te_model", "model")
example_texts = model.examples

def apply_te(text, lan='en'):
    return model.enhance_text(text, lan)
print(apply_te("done", lan='en'))

def video_conetext(video_id):
    srt = YouTubeTranscriptApi.get_transcript(video_id) 
    jsonSrt=json.dumps(srt)
    data = json.loads(jsonSrt)
    df = pd.json_normalize(data)

    df_final = df.text
    df_final = " ".join(df_final)
    result = re.sub(r'\[.+\]', '', df_final)
    return result

##################################API##############################

app = Flask(__name__)
CORS(app)

@app.route("/api",methods=['POST','GET'])

def api():
    if request.method=='POST':
        input_json = request.get_json()
        print(request.content_type)
        print(input_json['url'])
        #vurl=request.values.get('url')
        # print(vurl)
        # url=request.form['url']
        url=input_json['url']
        parsed_url=urlparse(url)
        video_id=parse_qs(parsed_url.query)['v'][0]
        print(video_id)
        context=video_conetext(video_id)
        print(context)
        p_context=apply_te(context,lan='en')
        print("\n",p_context)
        questions= mainModel(p_context)
        return jsonify(questions)
        #return jsonify({'Success':'100'})
    
    if request.method=='GET':
        print("get success")
        return jsonify({'Success':'100'})


if __name__ == '__main__':
    app.run( port=os.getenv("PORT", default=5000))
