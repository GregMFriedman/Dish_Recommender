
# coding: utf-8
import pandas as pd
import numpy as np
import re
import pickle
from pprint import pprint
from __future__ import division
import matplotlib.pyplot as plt 
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
from nltk.util import ngrams
from collections import defaultdict
from operator import itemgetter
from nltk.corpus import stopwords
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.collocations import *
from sklearn.feature_extraction.text import CountVectorizer
import urllib2


def get_restos(dataset, start=100001, num_dfs=50, top_x=10):
    """
    A method to collect restaurants and their reviews to score from a collection of hundreds of thousands of Yelp reviews.
    Input: a dictionary of Yelp reviews, each 10,0000 reviews long.
    Returns a cleaned dictionary with `num_dfs` entries.
    `Top_x` is the number of restaurants in each return dictionary sorted by number of reviews.
    """
    df_dict = {}
    for i in range(start, (start + num_dfs*10000), 10000):
        df = dataset[i]
        df['name'] = df['business_id'].apply(lambda x: id_dict[x])
        df['cats'] = df['name'].apply(lambda x: cat_dict[x])
        df['CityName'] = df['business_id'].apply(lambda x: url_dict[x])
        df['restos'] = df['cats'].apply(lambda x: 1 if 'Restaurants' in x else 0)
        df = df[['name', 'cats', 'stars', 'text', 'date', 'user_id', 'restos', 'CityName']]
        url_list = df.pivot_table('stars', index='CityName', columns='restos', aggfunc='count', margins=True).sort(1, ascending=False)
        urls = url_list.reset_index()['CityName'].tolist()[1:top_x+1]
        df_dict[i] = urls, df
    return df_dict


def screlpy(cityname):
    """
    A method to scrape a restaurant's menu from Yelp.
    Input: a restaurant identifier (in the form of restuarant's name and city combined by a dash)
    Returns a dictionary of the restaurant's menu if on Yelp, otherwise the string 'Menu not on Yelp'
    The dictionaries keys are determined by the headers on the menu page.
    """
    url = get_url(cityname)
    if url:
        site = requests.get(url)
        page = site.text
        soup = BeautifulSoup(page)
        explore = soup.find(class_= 'menu-explore')
        if explore:
            url = 'http://www.yelp.com' + explore['href']
            site = requests.get(url)
            page = site.text
            soup = BeautifulSoup(page)
            main = soup.find(class_='menu-sections')
            menu = {}
            course_list = []
            options = []
            for items in main:
                courses = main.find_all(class_="menu-section-header")
                for course in courses:
                    course_list.append(course.find('h2').text.strip())
                sections = soup.find_all(class_ = 'menu-section')
                for section in sections:
                    food_list = []
                    foods = section.find_all('h3')
                    for food in foods:
                        food_list.append(food.text.strip())
                    options.append(food_list)

            menu = dict(zip(course_list, options))
        else:
            menu = 'Menu not on Yelp'
            print cityname + ' Menu not on Yelp'
    else:
        menu = 'Page Not Found'
    return menu

def get_url(cityname):
    """
    A method to generate the Yelp url given a restaurant and its city.
    Input: the restaurant's city and name combined by a '-'.
    Returns the probable url of the restaunt's Yelp page.

    """
    cityname = cityname.replace(u'\xe9', 'e')
    cityname = cityname.replace(u'\xea', 'e')
    cityname = cityname.replace(u'\xe0', 'a')
    cityname = cityname.replace('&', 'and')
    cityname = cityname.replace('+', 'and')
    cityname = cityname.replace('---', '-')
    i = 0
    url = 'http://www.yelp.com/biz/'+ str(cityname)
    r = requests.get(url)
    try:
        url = 'http://www.yelp.com/biz/'+ str(cityname)
        r = requests.get(url)
    except:
        if (r.status_code != 200) & (i < 6):
            i += 1
            cityname = cityname + '-' + str(i)
        else:
            url = ''
    return url

def get_menus(citynames, menus=None):
    """
    A method to scrape menus for a list of restaurants.
    Input: list of restaurant identifiers & optionally an existing menu.
    Returns a dictionary of menus, whose keys are restaurants.
    """
    if not menus:
        menus = {}
    else:
        menus = menus
    for cityname in citynames:
        menus[cityname] = screlpy(cityname)
        cityname = cityname.replace(u'\xe9', 'e')
        cityname = cityname.replace(u'\xea', 'e')
        cityname = cityname.replace(u'\xe0', 'a')
        cityname = cityname.replace('&', 'and')
        cityname = cityname.replace('+', 'and')
        cityname = cityname.replace('---', '-')
        print 'screlped ' + str(cityname)
    return menus


def word_match(words, menu, vector_dict, threshold=75):
    """
    A method to match single words to menu-items.
    Inputs: a list of all words collected from a restaurant's Yelp reviews, that restaurant's menu,
    a dictionary of 6-dimensional vectors, the first dimension being total reviews in which the word appears and
    the following 5 dimensions are the number of 1 through 5-star reviews in which that word appears.
    Threshold is the minimum match score required to accept the word as a match to a menu item.
    Returns a tuple of length 3: the word, the matched menu item, and the word's vector.

    """
    matches = []
    for word in words:
        for key, value in menu.iteritems():
            for item in value:
                score = fuzz.ratio(word, item)
                if score > threshold:
                    matches.append((word, item, vector_dict[word]))
    if len(matches) < 5:
        for word in words:
            for key, course in menu.iteritems():
                for item in course:
                    short_item = ' '.join(item.split()[-2:])
                    score = fuzz.ratio(word, short_item)
                    if (score > threshold +5) and (word, item, vector_dict[word] not in matches):
                        matches.append((word, item, vector_dict[word]))
    if len(matches) < 5:
        for word in words:
            for key, course in menu.iteritems():
                for item in course:
                    short_item = ' '.join(item.split()[:2])
                    score = fuzz.ratio(word, short_item)
                    if (score > threshold +5) and (word, item, vector_dict[word] not in matches):
                        matches.append((word, item, vector_dict[word]))

    return matches

def best_matches(ngrams, menu, vector_dict, threshold=70):
    """
    A method to match ngrams to menu-items.
    Inputs: a list of all ngrams collected from a restaurant's Yelp reviews, that restaurant's menu,
    a dictionary of 6-dimensional vectors, the first dimension being total reviews in which the ngram appears and
    the following 5 dimensions are the number of 1 through 5-star reviews in which that ngrams appears.
    Threshold is the minimum match score required to accept the ngram as a match to a menu item.
    Returns a dictionary whose key is the ngram and value is the menu-item & the vector of the ngram.
    """
    best_matches = {}
    for ngram in ngrams:
        biggest = 0
        for key, value in menu.iteritems():
            for item in value:
                temp = {}
                score = fuzz.token_set_ratio(ngram, item)
                if score > threshold:
                    score2 = fuzz.partial_ratio(ngram, item)
                    score3 = fuzz.ratio(ngram, item)
                    temp[item] = score2+score3
                    if temp[item] > biggest:
                        biggest = temp[item]
                        best_matches[ngram] = item, vector_dict[ngram]
    if len(best_matches) < 5:
        for ngram in ngrams:
            biggest = 0
            for key, value in menu.iteritems():
                for item in value:
                    temp = {}
                    short_item = ' '.join(item.split()[-2:])
                    score = fuzz.token_set_ratio(ngram, short_item)
                    if score > threshold:
                        score2 = fuzz.partial_ratio(ngram, short_item)
                        score3 = fuzz.ratio(ngram, short_item)
                        temp[item] = score2+score3
                        if temp[item] > biggest:
                            biggest = temp[item]
                            best_matches[ngram] = item, vector_dict[ngram]
        
    return best_matches
            

def get_keyword_counts(column):
    """
    A method to count the number of reviews each ngram is mentioned in.
    Input: all reviews for a given restaurant.
    Returns a DataFrame whose columns are ngrams and whose rows are 1s and 0s,
    1s indicating that an ngram was mentioned at least once in that review, otherwise a 0.
    """
    text = column
    vectorizer = CountVectorizer(ngram_range=(1,3), stop_words='english')
    vectorizer.fit(text)
    features = vectorizer.get_feature_names()
    x = vectorizer.transform(text).toarray()
    counts = pd.DataFrame(x, columns=features)
    singles = counts.applymap(lambda x: 1 if x > 0 else 0)
    return singles


def get_weights(df, column=ex2['stars']):
    """
    A method to convert the total counts to a dictionary of counts by review.
    Inputs: the DataFrame of ngram mentions by review created by get_keyword_counts 
    and a Series of the number of stars in each corresponding review.
    Returns a dictionary of DataFrames whose keys are number of stars, and value the DataFrame
    of ngram mentions for each review of that many stars. 
    """
    singles = df.copy().reset_index() 
    weighted = singles.join(column.reset_index(), lsuffix = '_0')
    totals = weighted.append(weighted.sum(numeric_only=True), ignore_index=True)
    dfs = {}
    for i in range(1,6):
        dfs[i] = weighted[weighted['stars']== i]
    return dfs


def get_totals(df_dict):
    """
    A method to sum the mentions of each ngram over all reviews.
    Input: the dictionary of DataFrames created in `get_weights`
    Returns a dictionary of dictionaries whose outer keys are stars
    and inner keys are each ngram and its review count for that number of stars.
    """
    totals = {}
    for i in range(1,6):
        counts = {}
        df = df_dict[i]
        for (index, value) in df.sum().iteritems():
            counts[index] = value
        totals[i] = counts
    return totals


def get_vectors(totals):
    """
    A method to convert the dictionaries created in `get_totals` to a dictionary of vectors.
    Input: nested dictionary of ngram-counts by star rating.
    Returns a dictionary whose keys are ngrams and values are the 6-D vector of review counts
    for that ngrams described in `best-matches`.
    """
    vectors = {}
    for key in totals[1].keys():
        vectors[key] = sum([totals[1][key],totals[2][key],totals[3][key],totals[4][key],totals[5][key]]), totals[1][key],totals[2][key],totals[3][key],totals[4][key],totals[5][key]
    return vectors



def filter_keys(vector_dict, min_ngram_count = 3, min_word_count = 5):
    """
    A method to filter out matches that only appear in a few reviews for two reasons.  
    The first is because the purpose of the application is provide the user a list of popular
    dishes and rarely mentioned items are by definition not that popular.  The other is because
    mismatches are more frequent among infrequently mentioned ngrams.
    Inputs: dictionaries of ngrams and their vectors, thresholds for required mentions of each ngram.
    Returns a filtered list of words and ngrams to included in the final recommendations.


    """
    bigkeys = filter(lambda x: len(x.split()) > 1, vector_dict.keys())
    bigkeys = filter(lambda x: vector_dict[x][0] > min_ngram_count, bigkeys)
    
    littlekeys = filter(lambda x: len(x.split()) == 1, vector_dict.keys())
    littlekeys = filter(lambda x: vector_dict[x][0] > min_word_count littlekeys)
    
    return littlekeys, bigkeys


def get_recs(vector_dict, menu):
    """
    Method to convert the vectors created in get_vectors to a DataFrame of scores for
    each significant ngram.
    Inputs: vectors of ngrams by review count by rating and corresponding menu
    Returns a DataFrame of scores described in `get_scores` with a row for each ngram.
    """
    recs = []
    matches1 = word_match(filter_keys(vector_dict)[0], menu, vector_dict)
    matches2 = best_matches(filter_keys(vector_dict)[1], menu, vector_dict)
    for match in matches1:
        recs.append((get_scores(match[2]), match[0], match[1]))
    for key, value in matches2.iteritems():
        recs.append((get_scores(value[1]), key, value[0]))
    if recs:
        rdf = pd.DataFrame(recs)
        rdf['Score'] = rdf[0].apply(lambda x: x[1])
        rdf['Trendiness'] = rdf[0].apply(lambda x: x[0])
        rdf['Controversy'] = rdf[0].apply(lambda x: x[3])
        rdf['Buzz'] = rdf[0].apply(lambda x: x[2])
        rdf.rename(columns={1: 'Keyword', 2: 'MenuItem'}, inplace=True)
        rdf.drop(0, axis=1, inplace=True)
        rdf['Score'] = rdf['Score'].apply(lambda x: (x+2)*2.5)
        p_norm = rdf['Controversy'].max()
        b_norm = rdf['Trendiness'].max()
        rdf['Controversy'] = rdf['Controversy'].apply(lambda x: 10*x/p_norm)
        rdf['Trendiness'] = rdf['Trendiness'].apply(lambda x: 10*x/b_norm)
        return rdf.drop_duplicates()
    else:
        return pd.DataFrame()


def get_scores(vector):
    """
    Method that takes a vector of review-count by rating for an ngram and returns 4 scores.
    Input: ngram review-count rating by star.
    Returns 4 calculations:
    Score indicates average rating in which that review appears.
    Weighted is similar to score, but rewards items who have more total positive mentions.
    Polarity measures the variance in star ratings for each item.
    Buzz is just the total number of reviews in which the item is mentioned.
    """
    score = (2*vector[5] + vector[4] - vector[2] - 2*vector[1]) / vector[0]
    weighted = (2*vector[5] + vector[4] - vector[2] - 2*vector[1])
    polarity = np.var(vector[5]*[5] + vector[4]*[4] + vector[3]*[3] + vector[2]*[2] + vector[1]*[1])
    buzz = vector[0]
    
    
    return weighted, score, buzz, polarity


def final_recs(df):
    """
    Method to aggregate the scores of ngram by corresponding menu item.
    Input: DataFrame of scores, whose rows are ngrams.
    Returns a DataFrame of scores whose rows are the corresponding menu-items.
    This is the data the shown to the user.
    """
    if not df.empty:
        rdf = df.groupby('MenuItem').agg({'Score': np.mean,'Trendiness': np.mean, 'Controversy': np.mean, 'Buzz': np.sum}).reset_index()
        b_norm = rdf['Trendiness'].max()
        rdf['Trendiness'] = rdf['Trendiness'].apply(lambda x: 10*np.log(x)/np.log(b_norm) if (x > 1) else 0)
        rdf['Trendiness'] = rdf['Trendiness'].apply(lambda x: np.round(x,3))
        rdf['Score'] = rdf['Score'].apply(lambda x: np.round(x,3))
        rdf['Controversy'] = rdf['Controversy'].apply(lambda x: np.round(x,3))
        
        return rdf.sort(['Trendiness', 'Score'], ascending=False)
    else:
        return 'No popular items =('


def recs_collector(db_dict, rec_dict=None, menu_list=None, start=100001, num_dfs=1):
    """
    Method that gathers menus that corresponds to existing restaurant review data and generates
    scores for each item on each restaurant's menu.
    Input: dictionary of review data by restaurant.  Other inputs are included to be able to
    do this process in chunks.
    Returns a dictionary of 2-item lists of DataFrames.  
    The first DataFrame contains the final recommendations for the user.
    The second DataFrame contains the ngrams scores, which I used for debugging the 
    scoring, filtering and matching algorithms. 
    """
    if not rec_dict:
        rec_dict = {}
    if not menu_list:
        menu_list = []
    for i in range(start, (start + num_dfs*10000), 10000):
        db = db_dict[i]
        menus = get_menus(db[0])
        ex = db[1].copy()
        for key, menu in menus.iteritems():
            if key not in rec_dict:
                if menu == 'Menu not on Yelp':
                    rec_dict[key] = 'Menu not on Yelp'
                    key = key.replace(u'\xe9', 'e')
                    key = key.replace(u'\xea', 'e')
                    key = key.replace(u'\xe0', 'a')
                    key = key.replace('&', 'and')
                    key = key.replace('+', 'and')
                    key = key.replace('---', '-')
                    print str(key) + "'s Menu not on Yelp =("
                else:
                    if key in ex['CityName'].unique():
                        df = ex[ex['CityName']==key]
                        counts = get_keyword_counts(df.text)
                        dfs = get_weights(counts, df['stars'])
                        totals = get_totals(dfs)
                        vectors = get_vectors(totals)
                        rdf = get_recs(vectors, menu)
                        final = final_recs(rdf)
                        rec_dict[key] = [final, rdf]
                        key = key.replace(u'\xe9', 'e')
                        key = key.replace(u'\xea', 'e')
                        key = key.replace(u'\xe0', 'a')
                        key = key.replace('&', 'and')
                        key = key.replace('+', 'and')
                        key = key.replace('---', '-')
                        print 'Rec Made on ' + str(key)
                    else:
                        rec_dict[key] = 'No reviews found'
                        print "Couldn't find reviews on " + str(key)
            if menu not in menu_list: 
                menu_list.append((key, menu))
    return menu_list, rec_dict

def main():
    """
    Creates a picklefile of recommendations used dishes_setup.py 
    """
    with open('mini_dfs.pkl', 'r') as picklefile:
        minis = pickle.load(picklefile)
    db_dict = get_restos(minis)
    menus, recs_db = recs_collector(db_dict)
    menus, recs_db = recs_collector(db_dict, recs_db, menus, start=110001, num_dfs=5)
    with open('recs_db0.pkl', 'w') as picklefile:
        pickle.dump(recs_db, picklefile)
    with open('menus_dict.pkl', 'w') as picklefile:
        pickle.dump(menus, picklefile)

if __name__ == '__main__':
    main()

