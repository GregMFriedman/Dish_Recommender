from flask import Flask, render_template, request, redirect, url_for
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import Column, ForeignKey, Integer, String

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import relationship

from sqlalchemy import create_engine

from dishes_setup0 import Dish, Restaurant

Base = declarative_base()

app = Flask(__name__)

config = cnfg.load("ds/metis/ds5_Greg/projects/04-fletcher/.yelp/.db_config")
connection = config['connection']

engine = create_engine(connection)
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

with open('../city_list0.pkl', 'r') as picklefile:
     cities = pickle.load(picklefile)

@app.route('/')
@app.route('/city/')
def showCities():
	return render_template('cities.html', cities=cities)


@app.route('/city/<city>/')
def getRestos(city):
	restaurants = session.query(Restaurant).filter_by(City=city).all() 
	return render_template('index.html', restaurants=restaurants, City=city)


@app.route('/restaurant/<int:restaurant_id>/')
@app.route('/restaurant/<int:restaurant_id>/menu/')
def showMenu(restaurant_id):
    restaurant = session.query(Restaurant).filter_by(Index=restaurant_id).one()
    items = session.query(Dish).filter_by(Restos_Index=restaurant_id).all()
    return render_template('menu2.html', items=items, restaurant=restaurant)




if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)

