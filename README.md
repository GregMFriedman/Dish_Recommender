# Dish Recommender

Here's what my application does:

### find_restaurants.py

Finds restaurants based on user input.  
Scrapes Yelp reviews to find mentions of menu items at those restaurants. 
Scores items based on average rating of review in which those items are mentioned.

### scrape_deeper.py

From the restaurants whose dishes have been scored, gather geolocational data, and scrape the menu for
the descriptions of the dishes

### cluster_restaurants.py

Use k-means to cluster restaurants by ingredients in their menu descriptions.


### make_links.py

Flask app that displays restaurants by cluster on a google map and colors them by their categorization on Yelp
If you hover over a restaurant, you'll see the restaurant's name and a link to a table of its scored menu items, 
which the user can sort by score or number of reviews.  

I am working on getting that page up in the cloud.  Hopefully to come soon!
For now, here are some screenshots:

#### Map Page:

![Display Restaurants](/images/DisplayingClusters.png)

#### Sample Restaurant Landing Page:

![Dish Scores](/images/DispalyingDishes.png)

