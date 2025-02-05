from neo4j import GraphDatabase
import os 
# from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging
import sys
load_dotenv()



try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Successfull")
except Exception as e:
    print(e)    
