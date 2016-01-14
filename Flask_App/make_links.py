from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import numpy as np
import pandas as pd
import json



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
			words = map(lambda x: str(x.capitalize()), words)

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

	

	return render_template('menu.html', dishes=dishes, keywords=keywords, pic=pic, name=name)


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