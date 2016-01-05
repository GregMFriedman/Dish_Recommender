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

from pprint import pprint

from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import scale
from scipy import sparse


config = cnfg.load(".yelp_config")

OUTPUTDIR = 'output'

DEFAULT_TERM = 'restaurants'
DEFAULT_LOCATION = 'SoHo, NY'
DEFAULT_RADIUS = 10000

API_HOST = 'api.yelp.com'
SEARCH_LIMIT = 20
SEARCH_PATH = '/v2/search/'
BUSINESS_PATH = '/v2/business/'

# OAuth credential placeholders that must be filled in by users.
CONSUMER_KEY = config["consumer_key"]
CONSUMER_SECRET = config["consumer_secret"]
TOKEN = config['token']
TOKEN_SECRET = config['token_secret']


NUM_CLUSTERS = 15

DRINK_MENUS = [u'Bar Menu',
 u'Beer',
 u'Beer Menu',
 u'Beers',
 u'Beverage Menu',
 u'Beverages Menu',
 u'Brunch Menu',
 u'Cocktail Menu',
 u'Cocktails',
 u'Cocktails & Drinks',
 u'Cocktails and Drinks',
 u'Cull & Pistol Happy Hour Menu',
 u'Dessert',
 u'Dessert Menu',
 u'Dinner Menu',
 u'Drink Menu',
 u'Drinks',
 u'Drinks Menu',
 u'Happy Hour',
 u'Libations Menu',
 u'Other Drink Offerings',
 u'Treats Menu',
 u'Vino/Cocktails',
 u'Wine Menu']


def remove_accents(input_str):

	"""Cleans unicode strings with foreign characters and returns the English equivalent.
    Args:
        input_str (str): Any string, unicode or not
    Returns:
        output_str (str): Cleaned string if necessary, otherwise original string
    """ 

	if type(input_str) == unicode:
	    nfkd_form = unicodedata.normalize('NFKD', input_str)
	    output_str = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
	    return output_str
	else:
	    output_str = input_str
	    return output_str



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


def get_dish_descriptions(soup):

	"""Creates a nested dictionary of the menu for each meal with courses as outer
	keys, dish names as inner keys and descriptions of those dishes as the values.
	If there is no description available, the value defaults to the name of the dish. 
	Args:
	    soup (BeautifulSoup soup): Scraped site of the menu webpage
	Returns:
	    meal (str): Name of the meal 
	    menu (dict): A description dictionary for the menu of the given meal
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



def create_menu_strings(description_dict):

	"""Converts a dictionary of menu descriptions into strings for clustering
	entire restaurants (and all their menus) 
	Args:
	    description_dict (dict): dictionary of menu descriptions
	Returns:
	    menu_list (list): A list of mini-dictionaries with keys of biz_id
	    and dishes, whose values are both strings.  
	""" 

	menu_list = []
	for biz_id, menu in description_dict.iteritems():
	    temp = {}
	    dish_string = ''
	    for meal, courses in menu.iteritems():
	        if meal not in DRINK_MENUS:
	            for course, foods in courses.iteritems():
	                for food in foods:
	                    for word in food:
	                        dish_string += ' ' + word.strip()
	    if dish_string:
	        temp['biz_id'] = biz_id
	        temp['dishes'] = dish_string
	        menu_list.append(temp)

	return menu_list



def create_meal_strings(description_dict):

	"""Converts a dictionary of menu descriptions into strings for clustering
	by each menu (rather than each restaurant)
	Args:
	    description_dict (dict): dictionary of menu descriptions
	Returns:
	    meal_list (list): A list of mini-dictionaries with keys of biz_id
	    and dishes and meal, all of whose values are strings.  
	""" 

	meal_list = []
	for biz_id, menu in description_dict.iteritems():
	    for meal, courses in menu.iteritems():
	        temp = {}
	        dish_string = ''
	        for course, foods in courses.iteritems():
	            for food in foods:
	                for word in food:
	                    dish_string += ' ' + word.strip()
	        temp['meal'] = meal
	        temp['biz_id'] = biz_id
	        temp['dishes'] = dish_string
	        meal_list.append(temp)
	return meal_list




def cluster_restaurants(menu_list, num_clusters=NUM_CLUSTERS):
    
    """Takes a list of dish descriptions for a given restaurant or menu and
    clusters them.
    Args:
        menu_list (list): a list whose elements are dictionaries of biz_ids and dish 
        descriptions for one restaurant menu or an aggregation of all the menus 
        for one restaurant.
        NUM_CLUSTERS (int): number of clusters 
    Returns:
    	km (sklearn Kmeans clustering object): only returned for convenience in later
    	functions
    	vectorizer(TfidfVectorizer): also returned for convenience
        clusters (Numpy Array): an array whose indexes correspond to each description in
        dish_list and whose values are the cluster that description was assigned to.

    """ 
    #setup the dish vectors
    dish_list = []
    for dish in menu_list:
    	dish_list.append(dish['dishes'])
    vectorizer = TfidfVectorizer(stop_words = "english", ngram_range=(1,2))
    matrix = vectorizer.fit_transform(dish_list)
    word_array = matrix.toarray()
    df = pd.DataFrame(word_array, columns = vectorizer.get_feature_names())
    
    # run Kmeans Clustering
    km = KMeans(n_clusters=num_clusters, init = 'k-means++', max_iter = 100, n_init = 1)
    scale(df,with_mean=False)
    clusters = km.fit_predict(df)
    return vectorizer, km, clusters




def get_keyword_dict(vectorizer, km, num_clusters=NUM_CLUSTERS):
    
    """Creates a dictionary of lists of most common words by cluster to
    get a better idea of what clustered restaurant menus have in common.
    Args:
        vectorizer(TfidfVectorizer): in order to get the keywords
        km (sklearn Kmeans clustering object): in order to 'locate' cluster
        centers 
        NUM_CLUSTERS (int): number of clusters 
    Returns:
        keyword_dict(dict): a dictionary of most common words by cluster.
    """ 
    
    keyword_dict = {}
    terms = vectorizer.get_feature_names()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        keywords = []
        for ind in order_centroids[i, :50]:
            keywords.append(terms[ind])
        keyword_dict[i] = keywords

    return keyword_dict




def add_cluster_info(menu_list, clusters, keyword_dict, info_dict=None):

	"""Adds the meal and course for each dish in the reccomendation dictionary.
	Args:
	    menu_list (list): a list whose elements are dictionaries of biz_ids and dish 
	    clusters (Numpy Array): an array whose indexes correspond to each description in
	    dish_list and whose values are the cluster that description was assigned to.
	    keyword_dict(dict): a dictionary of most common words by cluster.
	Returns:
	    clustered_menus (dict): A augmented dictionary of recommendations
	"""

	clustered_menus = {}

	for i in range(len(clusters)):
	    
	    menu_list[i]['cluster'] = clusters[i]
	    menu_list[i]['cluster_words'] = keyword_dict[clusters[i]]
	    biz_id = menu_list[i]['biz_id']
	    clustered_menus[biz_id] = menu_list[i]
	    
	    if info_dict:
	    	if biz_id in info_dict:
		    	info_dict[biz_id]['cluster'] = clustered_menus[biz_id]['cluster']
		    	info_dict[biz_id]['cluster_words'] = clustered_menus[biz_id]['cluster_words']

	if info_dict:
		with open('info_dict_test.pkl', 'w') as picklefile:
			pickle.dump(info_dict, picklefile)

	else:
		with open('clustered_menus.pkl', 'w') as picklefile:
			pickle.dump(clustered_menus, picklefile)


def main():

	try:
		with open('info_dict.pkl', 'r') as picklefile:
			info_dict = pickle.load(picklefile)

	except:

		biz_ids = get_biz_ids(term=DEFAULT_TERM, location=DEFAULT_LOCATION, radius=DEFAULT_RADIUS)
		info_dict = get_stats_dict(biz_ids)

	try:
		with open('menu_descriptions.pkl', 'r') as picklefile:
			descriptions = pickle.load(picklefile)
 	except:
 		descriptions = {}
 		biz_ids = info_dict.keys()
 		for biz_id in biz_ids:
 			descriptions[biz_id] = screlpy_descriptions(biz_id)

 	
 	start_time = time.time()

 	menu_list = create_menu_strings(descriptions)

 	vectorizer, km, clusters = cluster_restaurants(menu_list)

 	keyword_dict = get_keyword_dict(vectorizer, km)

 	add_cluster_info(menu_list, clusters, keyword_dict, info_dict=info_dict)

 	print 'Clusters Created!'

 	print ("--- %s seconds ---") % (time.time() - start_time)


if __name__ == '__main__':
    main()