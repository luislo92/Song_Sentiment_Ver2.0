import requests
from bs4 import BeautifulSoup
import os
import re
from numpy import nan
import wikipedia
import json
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

os.chdir('/Users/luislosada/Columbia Drive/Applied Text and NLP')
df4 = pd.read_csv('pre-clean-dataset.csv')

list_of_genres = wikipedia.WikipediaPage('List of popular music genres').links[17:]

#eliminate (music), (music genre), (genre), (African music genre), (rap), (electronic dance music genre), List of
#drop duplicates and eliminate space.

l_of_gen_clean = []
to_remov = ['styles of music: A–F',
    'styles of music: G–M',
    'styles of music: N–R',
    'styles of music: S–Z']
for gen in list_of_genres:
        gen = re.sub('[(]music[)]',"",gen)
        gen = re.sub('[(]music genre[)]',"",gen)
        gen = re.sub('[(]genre[)]',"",gen)
        gen = re.sub('[(]African music genre[)]',"",gen)
        gen = re.sub('[(]rap[)]',"",gen)
        gen = re.sub('[(]electronic dance music genre[)]',"",gen)
        gen = re.sub('List of',"",gen)
        l_of_gen_clean.append(gen.lstrip(' ').lower())
l_of_gen_clean = [x for x in l_of_gen_clean if x not in to_remov] #remov non genres
df4.insert(loc=5,column = 'genre',value = l_of_gen_clean)

#Elimnate - remix or (remix), eliminate duplicates

def clean_name(df):
    clean = []
    for name in df['name']:
        name = name.lower()
        name = name.replace('(remix)', '')
        name = name.replace('remix', '')
        name = name.replace('- bonus', '')
        name = name.replace('(bonus track)', '')
        name = name.replace('√±', 'ñ')
        name = name.replace('- recorded at spotify studios nyc', "")
        name = name.replace('- live', "")
        name = name.replace('- dababy remix', "")
        name = name.replace('- radio edit', "")
        name = name.replace('- acoustic live', "")
        name = name.replace('- radio mix', "")
        name = name.replace('- the acoustic live session', "")
        name = name.replace('(live acoustic)', "")
        name = name.replace('- acoustic version', "")
        name = name.replace('- mtv unplugged, 2012', "")
        name = name.replace('- remastered 2014', "")
        name = name.replace('- remastered 2015', "")
        name = name.replace('- remastered 2009', "")
        name = name.replace('- remastered', "")
        name = name.replace('- 2004 remaster', "")
        name = name.replace('- 2003 remaster', "")
        clean.append(name)
    return clean

df4['clean_name'] = clean_name(df4)

df4.to_csv('df_with_clean_names.csv')

##################

df5 = pd.read_csv('df_with_tags.csv',index_col=0)
df5.reset_index(drop=True,inplace=True)

genre_col =[]
for string in df5['cleantag']:
    ll = list(str(string).split(","))
    x=True
    for gen in ll:
        if gen in l_of_gen_clean and x == True:
            genre_col.append(gen)
            x = False
    if x == True:
        genre_col.append(nan)

pd.Series(genre_col).value_counts()

df5.insert(loc=5,column = 'genre',value = genre_col)

genre = []
for i in range(len(df5)):
    artist = df5['artists'][i].replace(" ","_").lower()
    track = df5['name'][i].replace(" ","_").lower()
    link = str('https: // theaudiodb.com / api / v1 / json / 195003 / searchtrack.php?s=' + artist + '& t =' + track).replace(' ', '')
    #https: // theaudiodb.com / api / v1 / json / 195003 / searchtrack.php?s = maroon_5 & t = memories
    page = requests.get(link)
    page.status_code == 200 #if true the page was downloaded successfully
    soup = BeautifulSoup(page.content,"html.parser")
    js = json.loads(str(soup.text))
    try:
        genre.append(js['track'][0]['strGenre'])
    except TypeError:
        genre.append(nan)
df5.insert(loc=6,column = 'genre_2',value = genre)

df5.loc[df5['genre'].isna(),'genre'] = pd.Series(df5[df5['genre'].isna()].loc[:,'genre_2'])
df5.drop(columns=['genre_2']).to_csv('pre_tok.csv')

lyr = []
for i in range(len(df5)):
    try:
        lyr.append(nltk.word_tokenize(df5.loc[i, 'lyrics']))
        try:
            if lyr[i][1] == '-' or lyr[i][2] == '-' or lyr[i][3] == '-' or lyr[i][4] == '-' or lyr[i][5] == '-':
                lyr[i] = nan
            elif lyr[i][0] == 'This' and lyr[i][1] == 'is' and lyr[i][3] == 'ranking':
                lyr[i] = nan
        except IndexError:
            lyr[i] = lyr[i]
            continue
    except:
        lyr.append(nan)

l_of_chor = []
chor = []
for j in range(len(lyr)):
    xx = True
    try:
        for k in range(len(lyr[j])):
            if lyr[j][k] == 'Chorus':
                c = k+1
                while lyr[j][c] != "[" and xx==True:
                    chor.append(lyr[j][c])
                    c += 1
                xx=False
    except:
        l_of_chor.append(nan)
        continue
    l_of_chor.append(chor)
    chor = []

df5.drop(columns=['chorus'],inplace=True)
df5.insert(loc=7,column = 'chorus',value = l_of_chor)

dff = df5.dropna(subset = ['chorus']).reset_index(drop=True)

ind_remov = dff.loc[[dff.loc[i,'chorus'] == [] for i in range(len(dff))],'chorus'].index.tolist()
dff = dff.drop(index=ind_remov).reset_index(drop=True)


def clean_tokens (df):
    list_of_artists = list(set([item.lower() for it in [nltk.word_tokenize(art) for art in list(dff['artists'])] for item in it]))
    ff = []
    hh=[]
    lmtzr = WordNetLemmatizer()
    stopwords = list(set(nltk.corpus.stopwords.words('english')+['ap']+['i']+["y'all"]+['m.']+['mme']+['donot']+['rah']+
                         ['&'] + ['de']+['b']+['ca'] + ['of'] + ['us'] + ['the'] + ['at'] + ["in"] + ['and'] + ['be'] +
                         ['it'] + ['what'] + ['sv'] +['lo'] +['d']+['n']+['spotify']+['record']+['studios'] + ['chorus']+
                         ['verse']+['intro']+['outro']))
    for sent in df:
        #print(sent)
        for token in sent:
            tt = token.replace("'s", " ").replace("n't", "not").replace('-','').replace("'ll", "will").replace('my—','my').\
                replace("'cross",'across').replace("'ve",'have').replace("'bout","about").replace("'m","am").replace("'d","would").replace("'re",'are').\
                replace('wantt','want').replace('mr.','mister').replace('ms.','miss').replace('murda','murder').replace('like-','like').replace('smallz','small')
            tt = re.sub(r'^([0-9]|[0-9][0-9]|[0-9][0-9][0-9])$',' ',tt) #remove numbers
            tt = tt.lower()
            if tt == 'wo':
                tt = 'would'
            elif tt == 'gon':
                tt = 'going'
            elif tt == 'wan':
                tt = 'want'
            elif tt == 'na' or tt == "ta":
                tt = 'to'
            elif tt == 'ya':
                tt = 'you'
            elif tt == 'lil':
                tt = 'little'
            elif tt == 'ain':
                tt = 'am'
            elif tt == "'em" or tt == "em":
                tt = 'them'
            elif tt == 'cause' or tt == "'cause":
                tt = 'because'
            elif tt == 't':
                tt = 'not'
            elif tt == 'till' or tt == "'till" or tt == "'til" or tt == "til":
                tt = 'until'
            elif tt.endswith('—') == True:
                tt = tt.split('—')[0]
            elif tt == 'hol':
                tt = 'hold'
            elif tt == 'l':
                tt = 'lost'
            elif tt == 'cali':
                tt = 'california'
            tt = tt.split('_')

            if len(tt)==1:
                if tt[0] not in ['[]','[:’',':','[',']','?',',' ,')','(',' ',';','—','!',"'",'’','.','"',"...",'“','”',"”",'mme',"''",'``',"''",'si','vv','c','”','ii','+','$'] and tt[0] not in stopwords and tt[0] not in list_of_artists:
                    ff.append(tt[0])
            else:
                for t in tt:
                    if t not in ['[]','[:’',':','[',']','?',',' ,')','(',' ',';','—','!',"'",'’','.','"',"...",'“','”'+'mme',"''",'``',"''",'si','vv','c','”','ii','+','$'] and t not in stopwords and t not in list_of_artists:
                        ff.append(t)

        lemmas = [lmtzr.lemmatize(xt, 'v') for xt in ff]
        hh.append(lemmas)
        ff=[]
    return hh

dff.loc[:,'lyrics'] = clean_tokens([nltk.word_tokenize(dff.lyrics[i]) for i in range(len(dff.lyrics))])

dff.loc[:,'chorus'] = clean_tokens(dff['chorus'])

dic = dff.drop(columns=['genre_2','clean_name','cleantag']).to_dict()

with open('clean_js.json', 'w') as fp:
    json.dump(dic, fp)



