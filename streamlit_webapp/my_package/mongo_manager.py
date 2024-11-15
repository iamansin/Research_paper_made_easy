from pymongo import MongoClient

class MongoManager:
    def __init__(self, collection_name="admin_user", mongo_uri="mongodb://localhost:27017" ,db_name="Research_paper_app"):
        self.mongo_uri = mongo_uri
        self.client = None
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = None
        self.connect()

    def connect(self):
        #Establishes a connection to the MongoDB database and initializes the collection..
        try:
            self.client = MongoClient(self.mongo_uri)
            self.collection = self.client[self.db_name][self.collection_name]
            print(f"Connected to MongoDB database: {self.db_name}, collection: {self.collection_name}")
        except Exception as e:
            print(f"An error occurred while connecting to MongoDB: {e}")

    def insert_document(self, document):
        #Inserts a single document into the collection.
        try:
            result = self.collection.insert_one(document)
            print(f"Document inserted with id: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"An error occurred while inserting document: {e}")
            return None

    def insert_documents(self, documents):
        #Inserts multiple documents into the collection..
        try:
            result = self.collection.insert_many(documents)
            print(f"Documents inserted with ids: {result.inserted_ids}")
            return result.inserted_ids
        except Exception as e:
            print(f"An error occurred while inserting documents: {e}")
            return None

    def find_document(self, query):
        #Finds a single document based on the query.
        try:
            document = self.collection.find_one(query)
            return document
        except Exception as e:
            print(f"An error occurred while finding document: {e}")
            return None

    def find_documents(self, query):
        #Finds multiple documents based on the query.
        try:
            documents = list(self.collection.find(query))
            return documents
        except Exception as e:
            print(f"An error occurred while finding documents: {e}")
            return None

    def update_document(self, query, update):
        #Updates a single document based on the query.
        try:
            result = self.collection.update_one(query, update)
            print(f"Documents matched: {result.matched_count}, Documents modified: {result.modified_count}")
            return result.modified_count
        except Exception as e:
            print(f"An error occurred while updating document: {e}")
            return None

    def delete_document(self, query):
        #Deletes a single document based on the query.
        try:
            result = self.collection.delete_one(query)
            print(f"Documents deleted: {result.deleted_count}")
            return result.deleted_count
        except Exception as e:
            print(f"An error occurred while deleting document: {e}")
            return None

    def close_connection(self):
        #Closes the MongoDB connection.
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

