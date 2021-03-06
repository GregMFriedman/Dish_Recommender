# coding: utf-8

import json
import urllib
import urllib2
from cnfg import cnfg
import os
import time
import pickle
import requests
import oauth2
from bs4 import BeautifulSoup
from pymongo import MongoClient
import requests
from requests_oauthlib import OAuth1
import cnfg
import time
from datetime import datetime
from random import randint
import os
import inspect, os

import pandas as pd
import numpy as np

import nltk
import unicodedata
from nltk.util import ngrams
from collections import defaultdict
from operator import itemgetter
from nltk.corpus import stopwords
from textblob.sentiments import NaiveBayesAnalyzer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.feature_extraction.text import CountVectorizer


config = cnfg.load(".yelp_config")

OUTPUTDIR = 'output'

DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = '70 Battery Place, NY'
DEFAULT_RADIUS = 10000

term = DEFAULT_TERM
location = DEFAULT_LOCATION

# term = raw_input('What are you in the mood for? ')
# location = raw_input('Where are you thinking? ')

API_HOST = 'api.yelp.com'
SEARCH_LIMIT = 10
SEARCH_PATH = '/v2/search/'
BUSINESS_PATH = '/v2/business/'

# OAuth credential placeholders that must be filled in by users.
CONSUMER_KEY = config["consumer_key"]
CONSUMER_SECRET = config["consumer_secret"]
TOKEN = config['token']
TOKEN_SECRET = config['token_secret']



def remove_accents(input_str):


    if type(input_str) == unicode:
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    else:
        return input_str



def request(host, path, url_params=None):

    """Prepares OAuth authentication and sends the request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        dict: The JSON response from the request.
    Raises:
        urllib2.HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = 'https://{0}{1}?'.format(host, urllib.quote(path.encode('utf8')))

    consumer = oauth2.Consumer(CONSUMER_KEY, CONSUMER_SECRET)
    oauth_request = oauth2.Request(method="GET", url=url, parameters=url_params)

    oauth_request.update(
        {
            'oauth_nonce': oauth2.generate_nonce(),
            'oauth_timestamp': oauth2.generate_timestamp(),
            'oauth_token': TOKEN,
            'oauth_consumer_key': CONSUMER_KEY
        }
    )
    token = oauth2.Token(TOKEN, TOKEN_SECRET)
    oauth_request.sign_request(oauth2.SignatureMethod_HMAC_SHA1(), consumer, token)
    signed_url = oauth_request.to_url()

    conn = urllib2.urlopen(signed_url, None)
    try:
        response = json.loads(conn.read())
    finally:
        conn.close()

    return response



def get_biz_ids(term, location, radius, maxlimit=20):

    """Function to get the biz_ids from the yelp API.
    Args:
        term (str): The search term.
        location (str): The location to search for.
        radius (int): The radius (in meters) for the search.
        maxlimit (int): An optional set of query parameters in the request for maximim results returned. (Max is 20)
    Returns:
        businesses (list): A list of all the biz_ids for the search.
    """

    offset = 0
    businesses = []
    resume = True
    queries = 0;
    while resume:
        if queries > 3:
            break
        url_params = {
            'term': term.replace(' ', '+'),
            'location': location.replace(' ', '+'),
            'limit': maxlimit,
            'radius_filter': radius,
            'offset': offset,
            'sort': 2
        }
        output = request(API_HOST, SEARCH_PATH, url_params = url_params)
        n_records = len(output['businesses'])
        if n_records < 20:
            resume = False
        for i in range(n_records):
            businesses.append(output['businesses'][i]['id']) 
        offset += n_records
        queries += 1
    bizzes = map(lambda x: remove_accents(x), businesses)
    return bizzes



def create_dir(directory):
    """Check to see if directory exists, if not creates a directory
    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_soup_from_url(url):

    """Takes url and returns bs4.BeautifulSoup class.
    Args:
        url (str): A string for the url
    Returns:
        soup (bs4.BeautifulSoup): The beautiful soup object of a given url.
    """    
    soup = BeautifulSoup(requests.get(url).text, 'lxml')
    return soup


def get_business_url(biz_id, base = 'http://www.yelp.com/biz/'):

    """Takes biz_id and returns the url for a business.
    Args:
        biz_id (str): A unique business id for each Yelp business
    Returns:
        business_url (str): The url for the Yelp business.
    """ 

    business_url = base + biz_id
    return business_url


def get_number_reviews(biz_id):

    """Gets the unique number of reviews for a business.
    Args:
        biz_id (str): A unique business id for each Yelp business
    Returns:
        number_reviews (int): The unique number of reviews for a business.
    """ 

    url = get_business_url(biz_id)
    soup = get_soup_from_url(url)
    number_reviews = int(soup.find(itemprop = 'reviewCount').text)
    return number_reviews


def get_all_reviews_for_business(biz_id, max_limit = 20):

    """Gets all of customer reviews for a business.
    Args:
        biz_id (str): A unique business id for each Yelp business
    Returns:
        reviews (list): All the reviews for a given business.
    """ 

    reviews = []
    counter = 0
    url = get_business_url(biz_id)
    new_url = url
    num_reviews = get_number_reviews(biz_id)
    print biz_id, num_reviews
    if num_reviews > 500:
        num_reviews = 500
    
    while counter <= num_reviews:
        soup = get_soup_from_url(new_url)
        review_list = soup.find(class_ = 'review-list')
        for review in review_list.find_all(itemprop='description'):
            reviews.append(review.text)
        counter += max_limit
        new_url = '{0}?start={1}'.format(url, counter)
    return reviews


def get_all_scores_for_business(biz_id, max_limit = 20):

    """Gets all of customer review scores for a business.
    Args:
        biz_id (str): A unique business id for each Yelp business
    Returns:
        scores (list): All the review scores for a given business.
    """ 

    scores = []
    counter = 0
    url = get_business_url(biz_id)
    new_url = url
    num_reviews = get_number_reviews(biz_id)
    if num_reviews > 500:
        num_reviews = 500
    
    while counter <= num_reviews:
        soup = get_soup_from_url(new_url)
        review_list = soup.find(class_ = 'review-list')
        for score in review_list.find_all(class_ = 'rating-very-large'):
            try:
                rating = float(score.find(itemprop = 'ratingValue')['content'][0])
                scores.append(rating)
            except:
                pass
        counter += max_limit
        new_url = '{0}?start={1}'.format(url, counter)
    return scores


def get_one_menu(soup):

    """Creates a dictionary of the menu for each meal with courses as keys and
     lists of dishes for each course as the value: 
    Args:
        soup (BeautifulSoup soup): Scraped site of the menu webpage
    Returns:
        meal (str): Name of the meal 
        menu (dict): A menu dictionary for the given meal
    """ 

    meal = soup.find(class_='breadcrumbs').find('span').text.strip()
    main = soup.find(class_='menu-sections')
    menu = {}
    course_list = []
    options = []
    for items in main:
        courses = main.find_all(class_="menu-section-header")
        for course in courses:
            item = remove_accents(course.find('h2').text.strip())
            item = item.replace('.', '')
            course_list.append(item)
        sections = soup.find_all(class_ = 'menu-section')
        for section in sections:
            food_list = []
            foods = section.find_all('h4')
            for food in foods:
                item = remove_accents(food.text.strip())
                item = item.replace('.', '')
                food_list.append(item)
#             if len(food_list) == 0:
#                 food_list = [remove_accents(section)]
            options.append(food_list)
        menu = dict(zip(course_list, options))
    return meal, menu


def screlpy(biz_id):

    """Creates a dictionary of the restaurant's menu with courses as keys and
     lists of dishes for each course as the value: 
    Args:
        biz_id (str): The unique business id for each restaurant
    Returns:
        menu (dict): A menu dictionary for the given restaurant
    """ 
    menus = {}
    url = get_business_url(biz_id)
    soup = get_soup_from_url(url)
    explore = soup.find(class_= 'menu-explore')
    if explore:
        url = 'http://www.yelp.com' + explore['href']
        soup = get_soup_from_url(url)
        meal, meal_menu = get_one_menu(soup)
        menus[meal] = meal_menu
        other_menus = soup.find(class_="sub-menus")
        if other_menus:
            other_menus = other_menus.find_all('li')
            for other_menu in other_menus:
                if other_menu.a and 'Wine' not in other_menu.a.text:
                    other_url = 'http://www.yelp.com' + other_menu.a['href']
                    other_soup = get_soup_from_url(other_url)
                    meal, meal_menu = get_one_menu(other_soup)
                    menus[meal] = meal_menu
        return menus    
    else:
        return False



def get_menus(biz_ids, menus=None):

    """Creates a dictionary of restaurants and their corresponding menus.
    Args:
        biz_ids (list): A list of unique business ids for each Yelp business
        menus (dict): Existing dictionary to add to instead of starting a new one
    Returns:
        menus (dict): A dictionary of menus for the list of biz_ids
    """

    if not menus:
        menus = {}
    for biz_id in biz_ids:
        # biz_id = biz_id.replace(u'\xe9', 'e')
        # biz_id = biz_id.replace(u'\xea', 'e')
        # biz_id = biz_id.replace(u'\xe0', 'a')
        biz_id = remove_accents(biz_id)
        # biz_id = biz_id.replace('&', 'and')
        # biz_id = biz_id.replace('+', 'and')
        # biz_id = biz_id.replace('---', '-')
        if (biz_id not in menus) and screlpy(biz_id):
            menus[biz_id] = screlpy(biz_id)
            print 'screlped ' + biz_id
        else:
            print biz_id + "'s Menu already scraped"
    else:
        print biz_id + ' Menu not on Yelp'
    return menus



def get_keyword_counts(text):

    """Creates a DataFrame to count appearances of all n-grams across all reviews
    for a given restaurant
    Args:
        text (list): a list of all reviews for a given restaurant
    Returns:
        singles: a DataFrame whose columns are n-grams and rows are 
    a 1 or 0 corresponding to whether or not the n-gram appears in each review.
    """

    vectorizer = CountVectorizer(ngram_range=(1,3), stop_words='english')
    vectorizer.fit(text)
    features = vectorizer.get_feature_names()
    x = vectorizer.transform(text).toarray()
    counts = pd.DataFrame(x, columns=features)
    singles = counts.applymap(lambda x: 1 if x > 0 else 0)
    return singles



def get_vectors(df, ratings):

    """Creates a mapping of keywords to the number of reviews that keyword appears in
    for each star value 
    Args:
        df (DataFrame): DataFrame generated by 'get_keyword_counts'
        ratings (Series): a series of star ratings corresponding to each review
    Returns:
        vectors (dict): a dictionary whose keys are keywords and values are vectors 
    """

    singles = df.copy().reset_index(drop=True) 
    weighted = singles.join(ratings.reset_index(), lsuffix = '_0')
    if 'stars' in weighted.columns:
        weighted.drop('stars', 1, inplace=True)
    weighted.rename(columns={0:'stars'}, inplace=True)
    
    
    totals = {}
    for i in range(1,6):
        counts = {}
        df = weighted[weighted['stars']== i]
        for (index, value) in df.sum().iteritems():
            counts[index] = value
        totals[i] = counts

    vectors = {}
    for k in totals[1].keys():
        vectors[k] = sum([totals[1][k],totals[2][k],totals[3][k],totals[4][k],totals[5][k]]), \
        totals[1][k],totals[2][k],totals[3][k],totals[4][k],totals[5][k]
    return vectors




def vectorize_restaurant(biz_id):

    """Creates a mapping of keywords to the number of reviews that keyword appears in
    for each star value for a given restaurant
    Args:
        biz_id(str): unique identifier of each restaurant 
    Returns:
        vectors (dict): a dictionary whose keys are keywords and values are vectors 
    """
    
    reviews = get_all_reviews_for_business(biz_id) 
    ratings = pd.Series(get_all_scores_for_business(biz_id))
    
    counts = get_keyword_counts(reviews)
    vectors = get_vectors(counts, ratings)
    
    return vectors


def filter_keys(vector_dict, big_threshold=3, little_threshold=5):

    """Filters the keys in vectors down to words and phrases that appear in a substantial
    number of reviews for that restaurant.
    Args:
        vector_dict (dict): a dictionary whose keys are keywords and values are vectors
        big_threshold (int): minimum number of 2-gram and 3-gram mentions
        little_threshold (int): minimum number of single-word mentions
    Returns:
        littlekeys (list): a list of single keywords
        bigkeys (list): a list of 2-grams and 3-grams
    """

    bigkeys = filter(lambda x: len(x.split()) > 1, vector_dict.keys())
    bigkeys = filter(lambda x: vector_dict[x][0] > big_threshold, bigkeys)
    
    littlekeys = filter(lambda x: len(x.split()) ==1, vector_dict.keys())
    littlekeys = filter(lambda x: vector_dict[x][0] > little_threshold, littlekeys)
    
    return littlekeys, bigkeys



def best_matches(bigkeys, menu, vector_dict, threshold=90):

    """Matches 2-gram and 3-gram keyword vectors to menu items 
    Args:
        bigkeys (list): a list of 2-grams and 3-grams
        menu (dict): a dictionary of a menu for a given restaurant
        vector_dict (dict): a dictionary whose keys are keywords and values are vectors
        threshold (int): fuzzy-match score required for keyword matching 
    Returns:
        best_matches (dict): a dictionary whose keys menu items and values are tuples of keyword 
        matches and their corresponding fuzzy match score
    """


    best_matches = {}
    for name, meals in menu.iteritems():
        for meal, course in meals.iteritems():
            for food in course:
                item = food.replace('-', ' ')
                if len(item.split()) > 1 and len(item.split()) < 4:
                    matches = process.extractBests(item, bigkeys, scorer=fuzz.QRatio, score_cutoff=threshold, limit=3)
                    matches += process.extractBests(item, bigkeys, scorer=fuzz.UWRatio, score_cutoff=threshold, limit=3)
                    vectors = []
                    for match in matches:
                        vectors.append(vector_dict[match[0]])
                    best_matches[item] = dict(zip(matches, vectors))
                if len(item.split()) > 3:
                    descriptor = item.split()[-2:]
                    matches = process.extractBests(item, bigkeys, scorer=fuzz.QRatio, score_cutoff=threshold, limit=3)
                    matches += process.extractBests(item, bigkeys, scorer=fuzz.UWRatio, score_cutoff=threshold, limit=3)
                    vectors = []
                    for match in matches:
                        vectors.append(vector_dict[match[0]])
                    best_matches[item] = dict(zip(matches, vectors))
    return best_matches

def word_match(words, menu, vector_dict, threshold=93, best_matches=None):

    """Matches single-word keyword vectors to menu items 
    Args:
        littlekeys (list): a list of single keywords
        menu (dict): a dictionary of a menu for a given restaurant
        vector_dict (dict): a dictionary whose keys are keywords and values are vectors
        threshold (int): fuzzy-match score required for keyword matching 
        best_matches (dict): dict created in best_matches to add to
    Returns:
        best_matches (dict): a nested dictionary whose outer keys are menu items, inner keys
        are tuples of keyword matches and their corresponding fuzzy match score, and values
        are the review vectors of each keyword 
    """

    if not best_matches:
        best_matches = {}
    for name, meals in menu.iteritems():
        for meal, value in meals.iteritems():
            for food in value:
                item = remove_accents(food)
                if len(item.split()) == 1:
                    matches = process.extractBests(item, words, scorer=fuzz.QRatio, score_cutoff=threshold, limit=3)
                    vectors = []
                    for match in matches:
                        vectors.append(vector_dict[match[0]])
                    best_matches[item] = dict(zip(matches, vectors))
    return best_matches     


def get_keyword_dict(matches):

    """A mapping of menu items to keywords and number of reviews that keyword appeared in
     for debugging purposes
    Args: 
        best_matches (dict): dictionary of menu items, keywords and their vectors 
    Returns:
        best_matches (dict): a dictionary whose keys menu items and values are tuples of keyword 
        matches and their corresponding fuzzy match score
    """

    rows = []
    for key, match_list in matches.iteritems():
        for match, vector in match_list.iteritems():
            row = [match[0]] + list(vector)
            row.insert(0, key)
            rows.append(row[:3])
    match_dict = defaultdict(list)
    
    for row in rows:
        match_dict[row[0]].append(row[1:])
    return matches

def get_scores(vector):

    """Convert vectors to more relatable scores.
    Args: 
        vector(tup): a 6-D tuple--a total review count and a review count for all 5 star ratings
    Returns:
        buzz (int): total reviews
        weighted (float): a score favoring higher keyword counts
        score (float): average rating for the appearance of given keyword
        polarity (float): a score indicating higher variance among reviews
    """

    score = np.round( sum([i*vector[i] for i in range(1,6)]) / vector[0], 2)
    # weighted = np.round((2*vector[5] + vector[4] - vector[2] - 2*vector[1]),2)
    # polarity = np.round(np.var(vector[5]*[5] + vector[4]*[4] + vector[3]*[3] + vector[2]*[2] + vector[1]*[1]),2)
    buzz = vector[0]
    
    
    # return buzz, weighted, score, polarity
    return buzz, score


def rec_df(best_matches):

    """Create DataFrame of aggregate scores for each menu item or an empty list if 
     no matches are found for the menu items.
    Args: 
        best_matches (dict): dictionary of menu items, keywords and their vectors 
    Returns:
        rdf (DataFrame): a DataFrame whose rows are dishes and columns are scores
    """

    rows = []
    for key, match_list in best_matches.iteritems():
        for match, vector in match_list.iteritems():
            row = list(match) + list(get_scores(vector))
            row.insert(0, key)
            rows.append(row)
    rec_df = pd.DataFrame(rows)
    try:
        #rec_df.rename(columns={0:'Dish', 1: 'Keyword', 2: 'Match_Score', 3: 'Tot_Reviews', 4: 'Trendiness', 5:'Score', 6:'Controversy'}, inplace=True)
        rec_df.rename(columns={0:'Dish', 1: 'Keyword', 2: 'Match_Score', 3: 'Tot_Reviews', 4: 'Score'}, inplace=True)

        # rec_df['Score'] = rec_df['Score'].apply(lambda x: (x+2)*1.25)
        # rdf = rec_df.drop(['Trendiness', 'Controversy'],axis=1).pivot_table(index='Dish',  aggfunc={'Tot_Reviews':sum, 
        #                                                               'Score': np.mean, 'Match_Score': np.mean})
        rdf = rec_df.pivot_table(index='Dish',  aggfunc={'Tot_Reviews':sum, 'Score': np.mean, 'Match_Score': np.mean})
        rdf['Match_Score'] = np.round(rdf['Match_Score'],1)
        rdf['Score'] = np.round(rdf['Score'],2)
        rdf.reset_index(inplace=True)
        return rdf
    except:
        return []


def add_keywords(df):

    """Adds a column of keywords and counts for debugging purposes
    Args: 
        df (DataFrame): a DataFrame whose rows are dishes and columns are scores
    Returns:
        df (DataFrame): same df with added column of keywords and corresponding review counts
    """

    df['Keywords'] = df['Dish'].apply(lambda x: match_dict[x])
    return df


def add_course_details(rdf, menu, biz_id):

    """Adds the meal and course for each dish in the reccomendation dictionary.
    Args:
        rdf (DataFrame): A DataFrame of dishes
        menus (dict): dict of scraped menus
    Returns:
        rec (dict): A augmented dictionary of recommendations
    """

    
    rec = rdf.T.to_dict()
    #menu = menus[biz_id]
    for number, item in rec.iteritems():
        if number != '_id':
            dish = item['Dish']
            for meal, courses in menu.iteritems():
                for course, dishes in courses.iteritems():
                    if dish in dishes:
                        item['Course'] = course
                        item['Meal'] = meal
    return rec


def get_rec(biz_id, menus=None, rec_dict=None, folder='all_recs'):

    """ Encapsulates the full recommendation process from biz_id to dish reccomendation for that biz_id
    Args:
        biz_id (str): unique identifier for restaurant
        menus (dict): dict of scraped menus
        rec_dict (dict): dictionary of existing recommendations to avoid redundancies
        folder (str): path to the folder storing the recommendations 
    Returns:
        rec (dict): A dictionary of dishes and scores for a given biz_id
    """
    
    start_time = time.time()
    if biz_id in rec_dict:
        return rec_dict[biz_id]
    if biz_id in menus:
        menu = menus[biz_id]
    else:
        menu = screlpy(biz_id)
        if menus:
            menus[biz_id] = menu
            print biz_id + ' added to menus'
            
    #grab up to 500 reviews for a restaurant and return ratings vectors for each n-gram (n from 1-3)
    vectors = vectorize_restaurant(biz_id)
    print(biz_id + ' vectorized')


    #filter down to ngrams that appear multiple times throughout all reviews for that restaurant

    littlekeys, bigkeys = filter_keys(vectors)


    #find ngrams that are close matches with items on that restaurant's menu

    matches = best_matches(bigkeys, menu, vectors)
    word_matches = word_match(littlekeys, menu, vectors, best_matches=matches)

    #return a pandas DataFrame of menu items and their scores

    rdf = rec_df(word_matches)

    if (len(rdf) > 1):
        print biz_id + ' ' + str(len(rdf))

        create_dir(folder)

        rec = add_course_details(rdf, menu, biz_id)

        print ("--- %s seconds ---") % (time.time() - start_time)

        with open(folder + '/' + biz_id + '_recs.json', 'w') as fp:
            json.dump(rec, fp)

        return rec

    else:
        return []


def get_rec_dict(biz_ids, menus=None, rec_dict=None):



    for biz_id in biz_ids:
        
        rec = get_rec(biz_id, menus=menus, rec_dict=rec_dict)
        rec_dict[biz_id] = rec
            
    with open('rec_dict.pkl', 'w') as picklefile:
        pickle.dump(rec_dict, picklefile)

    print 'Recommendations Completed'



def get_stats(biz_id, menus=None, stats=None):

    """Creates a dictionary of relevant info for a given restaurant
    Args: 
        biz_id(str): unique identifier of each restaurant 
    Returns:
        stats(dict): a dictionary containing category, coordinates, city, neighborhood, rating,
        image_url, review_count, the restaurant's menu if on Yelp and the last time that menu was updated
    """


    business_path = BUSINESS_PATH + biz_id
    info = request(API_HOST, business_path)
    if not menus:
        menus = {}
    if not stats:
        stats = {}
        stats['category'] = info['categories'][0]
        stats['long'] = info['location']['coordinate']['longitude']
        stats['lat'] = info['location']['coordinate']['latitude']
        stats['city'] = info['location']['city']
        stats['neighborhood'] = info['location']['neighborhoods'][0]
        stats['image_url'] = info['image_url']
        stats['name'] = info['name']
        stats['rating'] = info['rating']
        stats['id'] = info['id']
        stats['review_count'] = info['review_count']
        if 'menu_date_updated' in info.keys():
            stats['menu_update'] = datetime.fromtimestamp(int(info['menu_date_updated'])).strftime('%Y-%m-%d')
        if biz_id in menus:
            stats['menu'] = menus[biz_id]
        else:
            stats['menu'] = screlpy(biz_id) 
    else:
        stats = stats
    return stats



def get_stats_dict(biz_ids, stats_dict=None):

    """Creates information dictionaries by restaurant IDs from a list of restuaraunts IDs
    Args: 
        biz_ids(str): list of unique restaurant identifiers
    Returns:
        stats_dict(dict): dictionary of info dictionaries whose keys are restaurant names
    """

    if not stats_dict:
        stats_dict = {}
    for biz_id in biz_ids:
        stats_dict[biz_id] = get_stats(biz_id)
    return stats_dict


def get_neighborhoods(stats_dict):


    """Finds all unique neighborhoods in a dictionary of restaurant stats
    Args: 
        stats_dict(dict): dictionary of info dictionaries whose keys are restaurant names
    Returns:
        neighborhoods(list): list of unique neighborhoods in stats_dict
    """

    neighborhoods = []
    for biz_id, stats in stats_dict.iteritems():
        if stats['neighborhood'] not in neighborhoods:
            neighborhoods.append(stats['neighborhood'])
    return neighborhoods


def get_dish_descriptions(soup):






    meal = soup.find(class_='breadcrumbs').find('span').text.strip()
    main = soup.find(class_='menu-sections')
    menu = {}
    course_list = []
    options = []
    for items in main:
        courses = main.find_all(class_="menu-section-header")
        for course in courses:
            item = remove_accents(course.find('h2').text.strip())
            item = item.replace('.', '')
            course_list.append(item)
        sections = main.find_all(class_='menu-section')
        for section in sections:
            food_list = []
            foods = section.find_all(class_='menu-item-details')
            for food in foods:
                dish = [remove_accents(food.h4.text).strip()]
                if food.find(class_="menu-item-details-description"):
                    ingredients = food.p.text.split(',')
                    ingredients = map(lambda x: remove_accents(x), ingredients)
                    #print dish, ingredients
                    dish += ingredients
                food_list.append(dish)
            options.append(food_list)
        menu = dict(zip(course_list, options))
    return meal, menu

def screlpy_descriptions(biz_id):

    """Creates a dictionary of the restaurant's menu with courses as keys and
     lists of dishes and their descriptions for each course as the value: 
    Args:
        biz_id (str): The unique business id for each restaurant
    Returns:
        menu (dict): A menu dictionary for the given restaurant
    """ 
    menus = {}
    url = get_business_url(biz_id)
    soup = get_soup_from_url(url)
    explore = soup.find(class_= 'menu-explore')
    if explore:
        url = 'http://www.yelp.com' + explore['href']
        soup = get_soup_from_url(url)
        meal, meal_menu = get_dish_descriptions(soup)
        menus[meal] = meal_menu
        other_menus = soup.find(class_="sub-menus")
        if other_menus:
            other_menus = other_menus.find_all('li')
            for other_menu in other_menus:
                if other_menu.a and 'Wine' not in other_menu.a.text:
                    other_url = 'http://www.yelp.com' + other_menu.a['href']
                    other_soup = get_soup_from_url(other_url)
                    meal, meal_menu = get_dish_descriptions(other_soup)
                    menus[meal] = meal_menu
        print biz_id + ' descriptions screlped'
        return menus    
    else:
        return False


def main():

    try:
        with open('rec_dict.pkl', 'r') as picklefile:
            rec_dict = pickle.load(picklefile)
    except:
        rec_dict = {}
        print 'new rec_dict created'

    term = DEFAULT_TERM
    location = DEFAULT_LOCATION
    radius = DEFAULT_RADIUS

    start_time = time.time()
    term = raw_input('What are you in the mood for? ')
    location = raw_input('Where are you thinking? ')

    location = location.replace(" ","")
    print ("Getting your restaurant reviews...")
    biz_ids = get_biz_ids(term, location, radius)

    try:
        with open('all_menus.pkl', 'r') as picklefile:
            menus = pickle.load(picklefile)
        menus = get_menus(biz_ids, menus)

    except:
        menus = get_menus(biz_ids)
    
    with open('all_menus.pkl', 'w') as picklefile:
        pickle.dump(menus, picklefile)
    print ('Menus scraped')

    # folder = ('_').join(location.replace(",", ' ').split  ())
    # sub_folder = ('_').join(term.split())
    # create_dir(folder+'/'+sub_folder)

    biz_ids = filter(lambda(x): x in menus, biz_ids)
    get_rec_dict(biz_ids, menus=menus, rec_dict=rec_dict)

if __name__ == '__main__':
    main()