import json
import requests
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import time


def single_query():
    response = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    if response.status_code == 200:
        return response.json()
    else:
        return none

def query_many(table, iters):
    for i in range(iters):
        print "Querying:", i
        content = single_query()
        try:
            # maybe check event_id
            eid = content['object_id']
            if not table.find_one({"object_id": eid}):
                table.insert_one(content)
                print "inserted!"
            else:
            	print "dupe!"
        except DuplicateKeyError:
            print 'DUPS!'
        time.sleep(10)

if __name__ == "__main__":
	db_cilent = MongoClient()
	db = db_cilent['fraudly']
	table = db['fraud']
	query_many(table, 1000)