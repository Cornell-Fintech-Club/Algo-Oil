# Package used to execute HTTP POST request to the API
import json
import urllib.request
import pandas as pd
from dateutil import rrule
from datetime import date, datetime, timedelta
import time

# curl -XPOST -H "Content-type: application/json" -d '{
#     "type": "filterArticles",
#     "queryString": "title:WTI OR description:WTI",
#     "from": 0,
#     "size": 50
# }' 'https://api.newsfilter.io/public/actions?token=8a9ea1e0669e43cc8e3bf11bfde7883b83d6242d3ae34deaa6805dcb3242537f'

# API endpoint
API_KEY = '8a9ea1e0669e43cc8e3bf11bfde7883b83d6242d3ae34deaa6805dcb3242537f'
API_ENDPOINT = "https://api.newsfilter.io/public/actions?token={}".format(API_KEY)

def articles_perweek (queryString, titles, descriptions):
  # Define the filter parameters
  # queryString = "title:WTI OR title:oil OR description:WTI OR symbols:WTI AND publishedAt:[2021-01-01 TO 2023-11-11]"
  # "symbols:NFLX AND publishedAt:[2020-02-01 TO 2020-05-20]"

  payload = {
  "type": "filterArticles",
  "queryString": queryString,
  "from": 0,
  "size": 50
  # max size for one api call is 50
  }

  # Format your payload to JSON bytes
  jsondata = json.dumps(payload)
  jsondataasbytes = jsondata.encode('utf-8')

  # Instantiate the request
  req = urllib.request.Request(API_ENDPOINT)

  # Set the correct HTTP header: Content-Type = application/json
  req.add_header('Content-Type', 'application/json; charset=utf-8')
  # Set the correct length of your request
  req.add_header('Content-Length', len(jsondataasbytes))

  # Send the request to the API
  response = urllib.request.urlopen(req, jsondataasbytes)

  # Read the response
  res_body = response.read()
  # Transform the response into JSON
  articles = json.loads(res_body.decode("utf-8"))
  # Get titles only
  a = articles['articles']
  i = 0
  while i < len(a):
    article = a[i]
    title = article['title']
    description = article['description']
    titles.append(title)
    descriptions.append(description)
    i = i + 1
  return titles, descriptions

#print(titles)
#print(articles)

# Run the function
titles = []
descriptions = []

now = datetime.now()
start = date(2021, 1, 1)

for dt in rrule.rrule(rrule.WEEKLY, dtstart=start, until=now):
  dt_plusweek = dt + timedelta(days=7)
  queryString = "title:WTI OR title:oil OR title:petroleum OR description:WTI OR description:oil OR description:petroleum OR symbols:WTI AND publishedAt:[" + dt.strftime('%Y-%m-%d') + " TO " + dt_plusweek.strftime('%Y-%m-%d') + "]"
  titles, descriptions = articles_perweek(queryString, titles, descriptions)
  time.sleep(1)

# print(titles)

# Export to csv for annotations
df = pd.DataFrame(descriptions, titles)#
df.to_csv('article_titles_descriptions.csv')