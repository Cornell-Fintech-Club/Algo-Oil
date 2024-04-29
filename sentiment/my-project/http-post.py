from datetime import datetime, timedelta
import json
import urllib.request
import pandas as pd
from dateutil import rrule
import time

# API endpoint
API_KEY = '8a9ea1e0669e43cc8e3bf11bfde7883b83d6242d3ae34deaa6805dcb3242537f'
API_ENDPOINT = "https://api.newsfilter.io/public/actions?token={}".format(API_KEY)

def collect_and_split_data(query_string, start_date, end_date):
    titles = []
    descriptions = []
    dates = []

    for dt in rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date):
        dt_plusweek = dt + timedelta(days=7)
        payload = {
            "type": "filterArticles",
            "queryString": query_string.format(dt.strftime('%Y-%m-%d'), dt_plusweek.strftime('%Y-%m-%d')),
            "from": 0,
            "size": 50
        }
        jsondata = json.dumps(payload)
        jsondataasbytes = jsondata.encode('utf-8')

        req = urllib.request.Request(API_ENDPOINT)
        req.add_header('Content-Type', 'application/json; charset=utf-8')
        req.add_header('Content-Length', len(jsondataasbytes))

        response = urllib.request.urlopen(req, jsondataasbytes)
        res_body = response.read()
        articles = json.loads(res_body.decode("utf-8"))

        # API call
        for article in articles['articles']:
            titles.append(article['title'])
            descriptions.append(article['description'])
            try:
                # parse with timezone offset
                dates.append(datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%S%z').date())
            except ValueError:
                # otherwise parse as UTC time
                dates.append(datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%S.%fZ').date())
        time.sleep(1)
    
    # Split data based on date split
    split_date = start_date + timedelta(days=int((end_date - start_date).days * 0.7))
    train_data = pd.DataFrame({'Title': titles[:int(len(titles) * 0.7)], 'Description': descriptions[:int(len(descriptions) * 0.7)], 'Date': dates[:int(len(dates) * 0.7)]})
    test_data = pd.DataFrame({'Title': titles[int(len(titles) * 0.7):], 'Description': descriptions[int(len(descriptions) * 0.7):], 'Date': dates[int(len(dates) * 0.7):]})
    
    return train_data, test_data

# query strings
wti_query_string = "title:WTI AND description:WTI AND publishedAt:[{} TO {}]"
oil_query_string = "title:oil AND description:oil AND publishedAt:[{} TO {}]"

# stard & end dates
start_date = datetime(2023, 12, 1)
end_date = datetime.now()

# WTI query
wti_train_data, wti_test_data = collect_and_split_data(wti_query_string, start_date, end_date)

# oil query
oil_train_data, oil_test_data = collect_and_split_data(oil_query_string, start_date, end_date)

# merge
train_combined = pd.concat([wti_train_data, oil_train_data]).sort_values(by='Date')
test_combined = pd.concat([wti_test_data, oil_test_data]).sort_values(by='Date')

# reoder
train_combined = train_combined[['Date', 'Title', 'Description']]
test_combined = test_combined[['Date', 'Title', 'Description']]

# save
train_combined.to_csv('train_oil.csv', index=False)
test_combined.to_csv('test_oil.csv', index=False)
