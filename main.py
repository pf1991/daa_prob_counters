#prob
from random import choices

# counter functions
from collections import Counter

# get text from url
import requests
from bs4 import BeautifulSoup

from math import sqrt, log
import pandas as pd 
import numpy as np 
from sklearn import metrics
from collections import OrderedDict

# helpers
def counter(words, prob = None, decreasing = None):
    res = {}
    for w in words:

        # init if not exists
        if w not in res:
            res[w] = 0

        # calc prob
        p = None
        if prob is not None and decreasing != True: 
            #fixed probability        
            p = prob
        elif  prob is not None and decreasing == True:
            #decreasing probability
            p = prob**res[w]

        # can count?
        can_count = True
        if p is not None:
            can_count = choices([True, False], [p, 1-p])[0]

        if can_count:
            res[w] = res[w] + 1
    
    return [(k, v) for k, v in sorted(res.items(), reverse=True, key=lambda item: item[1])]


def get_text(url):
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text=True)

    output = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head', 
        'input',
        'script',
    ]

    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    
    return output

def load_words():

    # URLs
    url_pt = 'https://pt.wikipedia.org/wiki/Albert_Einstein'
    url_en = 'https://en.wikipedia.org/wiki/Albert_Einstein'

    # get stop words
    stop_words_pt = None
    stop_words_en = None
    with open('stop_words_pt.txt', 'r') as file:
        stop_words_pt = file.read().replace('\n', ' ')
    with open('stop_words_en.txt', 'r') as file:
        stop_words_en = file.read().replace('\n', ' ')

    stop_words_pt = stop_words_pt.split()
    stop_words_en = stop_words_en.split()

    # get texts
    text_pt = get_text(url_pt).split()
    text_pt = list(filter(lambda v: v not in stop_words_pt and v.isalpha(), text_pt))
    text_en = get_text(url_en).split()
    text_en = list(filter(lambda v: v not in stop_words_en and v.isalpha(), text_en))

    return(text_pt, text_en)

def run_counters(words_list, p_fixed, p_decr):

    # my probabilities
    p_fixed = 0.25 # 1 / 2**2
    p_decr = 1/sqrt(2)  # 1 / 2**1/2

    #count words with fixed and decrease prob
    fixed = counter(words_list, p_fixed)
    decr = counter(words_list, p_decr, True)

    #build dataframe with all results
    df_fixed = pd.DataFrame(fixed, columns=['Word', 'Fixed'])
    df_decremental = pd.DataFrame(decr, columns=['Word', 'Decremental'])
    df = df_fixed.merge(df_decremental, left_on='Word', right_on='Word')
    df['Fixed Aprox.'] = df.Fixed / p_fixed # 2 ** 2 * counter
    df['Decremental Aprox.'] = (p_decr**-1)**df.Decremental - 1
    # df['Decremental Expected.'] = np.floor(np.log(df.Exact)/np.log((p_decr**-1))) + 1

    return df

def count_words(words):

    #pt text
    exact_count = counter(words)

    #init vars
    df = pd.DataFrame(exact_count[0:20], columns=['Word', 'Exact'])

    p_decr = 1/sqrt(2)
    p_fixed = 1/4
    output = {}
    data_frame_collection = []
    
    #count words t times
    for i in range(0,t):
        df_pt = run_counters(words, p_fixed, p_decr)
        df_pt = df.merge(df_pt, left_on='Word', right_on='Word')
        data_frame_collection.append(df_pt)

    #group all counts
    df_concact = pd.concat(data_frame_collection).groupby('Word')
    df_describe = df_concact.describe()

    #merge results and build output table
    for k in df_concact.groups.keys():
        df_group = df_concact.get_group(k)
        describe_row = df_describe.loc[ k , : ]
        df_group = df_concact.get_group(k)
        mad = df_group.mad()
        
        output[k] = {}

        #exact count
        output[k]['Exact'] = {}
        output[k]['Exact']['TRIALS'] = describe_row['Exact']['count']
        output[k]['Exact']['MEAN'] = describe_row['Exact']['mean']
        
        data_labels = ['Fixed', 'Fixed Aprox.', 'Decremental', 'Decremental Aprox.']
        for l in data_labels:
            
            output[k][l] = {}

            #Expected aprox value
            if 'Decremental' == l:
                output[k][l]['EXPECTED_COUNT'] = np.floor(np.log( describe_row['Exact']['mean'])/np.log((p_decr**-1))) + 1
            elif l == 'Fixed':
                output[k][l]['EXPECTED_COUNT'] =  describe_row['Exact']['mean']*p_fixed

            aprox_counter_value_col = l + ' Aprox.'         
            output[k][l]['MEAN'] = describe_row[l]['mean']
            output[k][l]['STD'] = describe_row[l]['std']
            output[k][l]['MIN'] = describe_row[l]['min']
            output[k][l]['MAX'] = describe_row[l]['max']
            output[k][l]['TRIALS'] = describe_row[l]['count']
            #Mean absolute deviation
            output[k][l]['MAD'] = mad[l]

            #calc error
            if 'Aprox.' in l:
                output[k][l]['MAE'] = metrics.mean_absolute_error(df_group['Exact'].tolist(), df_group[l].tolist())

    output = OrderedDict(sorted(output.items(), key=lambda i: i[1]['Fixed']['MEAN'], reverse=True))
    df = pd.concat({k: pd.DataFrame(v).T for k, v in output.items()}, axis=0).fillna('-')
    return df

if __name__== "__main__":

    pt_words, en_words = load_words()

    test_range = [10, 100]
    for t in test_range:
        
        df = count_words(pt_words)
        df.to_csv('pt_results%d.csv' % t)

        df = count_words(en_words)
        df.to_csv('en_results%d.csv' % t)
