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
            if meal not in drink_menus:
                for course, foods in courses.iteritems():
                    for food in foods:
                        for word in food:
                            dish_string += ' ' + word.strip()
        if dish_string:
            temp['biz_id'] = biz_id
            temp['dishes'] = dish_string
            menu_list.append(temp)
    return menu_list