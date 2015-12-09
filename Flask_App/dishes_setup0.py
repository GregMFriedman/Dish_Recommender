import os
import sys

from sqlalchemy import Column, ForeignKey, Integer, String

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import relationship

from sqlalchemy import create_engine

from sqlalchemy.orm import mapper, sessionmaker

from sqlalchemy import UniqueConstraint

import pickle

Base = declarative_base()

with open('../resto_info0.pkl', 'r') as picklefile:
     rdb = pickle.load(picklefile)
with open('../master_dict0.pkl', 'r') as picklefile:
     ind = pickle.load(picklefile)

class Restaurant(Base):
	__tablename__ = 'Restos0'

	Index = Column(Integer, primary_key=True)
	City = Column(String)
	Latitude = Column(String)
	Longitude = Column(String)
	Name = Column(String)
	CityName = Column(String,  nullable= False)
	Rating = Column(String)
	Review_Count = Column(Integer)
	RestID = Column(Integer, unique=True)

class Dish(Base):
	__tablename__ = 'Menus0'

	
	Index = Column(Integer, primary_key = True)
	MenuItem = Column(String, nullable= False)
	Trendiness = Column(String)
	Buzz = Column(String)
	Score = Column(String)
	Controversy = Column(String)
	Id = Column(String)
	City = Column(String)
	Rating = Column(String)
	Restos_Index = Column(Integer, ForeignKey('Restos0.Index'))
	Resto = Column(String)
	restaurant = relationship(Restaurant, primaryjoin='Restaurant.Index==Dish.Restos_Index', order_by=Index)
	



if __name__ == "__main__":

	config = cnfg.load("ds/metis/ds5_Greg/projects/04-fletcher/.yelp/.db_config")
	connection = config['connection']

	engine = create_engine(connection, echo=True)
	Base.metadata.create_all(engine)
	Base.metadata.bind = engine
	Session = sessionmaker(bind=engine)
	session = Session()


	g = 0
	r = ind[0]
	resto = Restaurant(Index=0, City=r['city'], Latitude=r['latitude'], Longitude=r['longitude'], Name=r['name'], CityName=r['CityName'], Rating=r['stars'], Review_Count=r['review_count'], RestID=0)
	dishes = r['menu']
	for j in range(len(dishes)):
		print len(dishes)
		rec = dishes[j]
		dish = Dish(Index=g, MenuItem=rec['MenuItem'], Trendiness=rec['Trendiness'], Buzz=rec['Buzz'], Score=rec['Score'], Controversy=rec['Controversy'], Id= rec['Id'], City=rec['City'], Rating=rec['Rating'], Resto=rec['Resto'] , restaurant=resto)
		g = g + 1
		session.add(dish)
		session.commit()

	for i in range(1, len(ind)):
	    r = ind[i]
	    resto = Restaurant(Index=i, City=r['city'], Latitude=r['latitude'], Longitude=r['longitude'], Name=r['name'], CityName=r['CityName'], Rating=r['stars'], Review_Count=r['review_count'], RestID=i)
	    dishes = r['menu']
	    for j in range(len(dishes)):
	        print len(dishes)
	        rec = dishes[j]
	        dish = Dish(Index=g, MenuItem=rec['MenuItem'], Trendiness=rec['Trendiness'], Buzz=rec['Buzz'], Score=rec['Score'], Controversy=rec['Controversy'], Id= rec['Id'], City=rec['City'], Rating=rec['Rating'], Resto=rec['Resto'] , restaurant=resto)
	        g = g + 1
	        session.add(dish)
	        session.commit()

    	
	
	# for i in range(len(recs_table)):
	#     rec = recs_table[i]
	#     g = ind[i]
	#     item = Dish(Index=i, MenuItem=rec['MenuItem'], Trendiness=rec['Trendiness'], Buzz=rec['Buzz'], Score=rec['Score'], Controversy=rec['Controversy'], Id= rec['Id'], City=rec['City'], Rating=rec['Rating'], Resto=rec['Resto'], RestID=rec['RestID'], restaurant=rest_dict[rec['RestID']])
	#     session.add(item)
	#     session.commit()

	

	
# 	