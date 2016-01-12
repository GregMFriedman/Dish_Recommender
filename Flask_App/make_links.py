from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_json import FlaskJSON, JsonError, json_response, as_json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pickle
import numpy as np
import pandas as pd
import cnfg
from sqlalchemy import Column, ForeignKey, Integer, String
import requests
from requests_oauthlib import OAuth1
import cnfg
import time
from random import randint
import os
import inspect, os
import json


app = Flask(__name__)

config = cnfg.load(".psql_config")
connection = config['connection']

@app.route('/')
@app.route('/restaurants/json/')
def recs_json():
	with open('/Users/gregoryfriedman/Dish_Recommender/recs_master.json', 'r') as f:
		rec = f.read()
		r = json.loads(rec)
	return jsonify(r)


@app.route('/restaurants/')
def displayDishes():
	return render_template('dishes.html')


# @app.route('/restaurant/<restaurant_id>/menu/')
# def showMenu(restaurant_id):
#     restaurant = session.query(Restaurant).filter_by(Index=restaurant_id).one()
#     items = session.query(Dish).filter_by(Restos_Index=restaurant_id).all()
#     return render_template('menu2.html', items=items, restaurant=restaurant)

@app.route('/')
@app.route('/foodmap/json/')
def dishes_json():
	with open('/Users/gregoryfriedman/Dish_Recommender/dishes.json', 'r') as f:
		rec = f.read()
		r = json.loads(rec)
	return jsonify(r)

@app.route('/foodmap/')
def showRestaurants():
	with open('../info_dict_clustered.pkl', 'r') as picklefile:
		info_master = pickle.load(picklefile)

	clusters = {}

	for biz_id, info in info_master.iteritems():

		if 'cluster' in info:

			name = info['cluster']
			words = info['cluster_words'][:10]

			clusters[name] = words


	return render_template('restaurant_map.html', clusters=clusters)

@app.route('/foodmap/<biz_id>')
def showMenu(biz_id):
	with open('../info_dict_clustered.pkl', 'r') as picklefile:
		info_master = pickle.load(picklefile)

	d = info_master[biz_id]
	
	
	if 'cluster_words' in d:
		keywords = d['cluster_words'][:10]
	else:
		keywords = ['hi']
	pic = d['image_url']
	name = d['name']
	dishes = d['dish_scores']
	

	return render_template('menu3.html', dishes=dishes, keywords=keywords, pic=pic, name=name)


@app.route('/dishmap/')
def showDishes():
	return render_template('dish_map.html')


@app.route('/foodmap/json/<biz_id>')
@as_json
def get_value(biz_id):
    with open('../info_dict_clustered.pkl', 'r') as picklefile:
        info_master = pickle.load(picklefile)
    for biz_id, info in info_master.iteritems():

    
	    if 'dish_scores' in info:
	        
	        if 0 in info['dish_scores']:
	            
	            print biz_id
	        
	            scores_json = json.dumps(info['dish_scores'])
	            info['dish_scores'] = scores_json
	    
	    if 'cluster' in info:
	        info['cluster'] = str(info['cluster'])

    d = info_master[biz_id]
    return jsonify(d)



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)