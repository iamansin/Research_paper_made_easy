from neo4j import GraphDatabase
import os 
# from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging
import sys
load_dotenv()


handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logging.getLogger("neo4j").addHandler(handler)
logger =logging.getLogger("neo4j").setLevel(logging.DEBUG)
NEO4J_URI="neo4j+ssc://659de33b.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="GbaCybkq06KMac9xPd5mOKgWbUy5NHawuU4cu91bwIc"
AURA_INSTANCEID="659de33b"
AURA_INSTANCENAME="Instance01"

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Successfull")
except Exception as e:
    print(e)    