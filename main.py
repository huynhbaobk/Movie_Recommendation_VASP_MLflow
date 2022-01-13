import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
from datetime import date, datetime
import operator
import time
import grpc
from bs4 import BeautifulSoup

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

# convert list of numbers to list (eg. "[1,2,3]" to [1,2,3])
def convert_to_list_num(my_list):
    my_list = my_list.split(',')
    my_list[0] = my_list[0].replace("[","")
    my_list[-1] = my_list[-1].replace("]","")
    return my_list

with open('movielens_dataset/movDB_to_movML_dict.json') as f:
    movDB_to_movML_dict = json.load(f)


with open('movielens_dataset/movML_to_movDB_dict.json', 'rb') as f:
    movML_to_movDB_dict = json.load(f)


def get_suggestions():
    movDB_to_movML_dict.keys()
    titles = list(dict(movDB_to_movML_dict).keys())
    list_suggestions = [title.capitalize() for title in titles]
    return list_suggestions

# create an instance
app = Flask(__name__)


### Read all user_ids with movie_ids 
def read_purchases_txt(directory, p=''):
    print("Reading purchases_txt" + p)
    purchases_txt = pd.read_json(directory + 'purchases_txt' + p + '.json')
    # purchases_txt['userId'] = purchases_txt.userId.apply(str)
    return purchases_txt

### Read training movies (just predict in this set)
def read_items_sorted(directory, p='_pu5'):
    print("Reading items_sorted" + p)
    items_sorted = pd.read_json(directory + 'items_sorted' + p + '.json')
    items_sorted['itemid'] = items_sorted.itemid.apply(str)
    return items_sorted

def train_tokenizer(directory):
    items_sorted = read_items_sorted(directory)
    toki = tf.keras.preprocessing.text.Tokenizer()
    toki.fit_on_texts(items_sorted.itemid.to_list())
    _, num_movies = toki.texts_to_matrix(['xx']).shape
    print("Number of training movies", num_movies)
    return toki, num_movies


dataset_path = '/home/baohuynh/baohuynh/recommendation/The-Movie-Cinema/movielens_dataset/'
toki, num_movies = train_tokenizer(dataset_path)
id_movies_2_index_vector_dict = toki.word_index

data_movies = pd.read_csv('/home/baohuynh/baohuynh/recommendation/The-Movie-Cinema/movielens_dataset/movies.csv')
data_ratings = pd.read_csv('/home/baohuynh/baohuynh/recommendation/The-Movie-Cinema/movielens_dataset/ratings.csv')

movies_name2id_dict = {}
movies_id2name_dict = {}
for ml_search_movie_id,movie_name in zip(list(data_movies["movieId"]),list(data_movies["title"])):
    movies_name2id_dict[movie_name] = ml_search_movie_id
    movies_id2name_dict[ml_search_movie_id] = movie_name


movie_ids_of_users =  read_purchases_txt(dataset_path)
movie_ids_of_users_dict = dict(zip(movie_ids_of_users.userId, movie_ids_of_users.itemids))


def get_key(dictionary, val):
    for key, value in dictionary.items():
         if val == value:
             return key
    return "key doesn't exist"


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/recommend",methods=["POST"])
def recommend():
    # # getting data from AJAX request
    title = request.form['title']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()


    ##########################################################################################################################33
    global data_movies 
    global data_ratings 

    if request.method == 'POST':
        db_search_movie_name = title.lower()
        db_search_movie_year = release_date[-4:]
        try:
            # print('[INFO] Search movie: ', db_search_movie_name)
            db_search_id_name, ml_search_movie_name =  movDB_to_movML_dict[db_search_movie_name]
            print(f'[INFO] Recommend for {ml_search_movie_name} ID: {db_search_id_name}')
        
            ml_search_movie_id = movies_name2id_dict[ml_search_movie_name]
                       
            userId = 1
            rating = 5

            # print(movie_ids_of_users_dict)
            movie_ids_of_users_dict[userId] = movie_ids_of_users_dict[userId] + ',' +str(ml_search_movie_id)
            user_data =  movie_ids_of_users_dict[userId]
            # user_data = ['648,2018,1022,9,714,81,349' + ',' + str(ml_search_movie_id)]
            
            movie_ids_of_users.itemids[get_key(movie_ids_of_users.userId, userId)] = str(movie_ids_of_users.itemids[get_key(movie_ids_of_users.userId, userId)]) + ',' + str(ml_search_movie_id)
            movie_ids_of_users.to_json("/home/baohuynh/baohuynh/recommendation/The-Movie-Cinema/movielens_dataset/purchases_txt.json")

            # user_data = data_ratings[data_ratings.userId == userId]
            print(f'[INFO] User data for ID_{userId}: {user_data}')
            
            new_rating = pd.DataFrame({"userId":[userId],
                    "movieId":[ml_search_movie_id],
                    "rating":[rating],
                    "timestamp": [int(time.time())]})
            print(new_rating)
            
            data_ratings = data_ratings.append(new_rating, ignore_index=True)
            data_ratings.to_csv('/home/baohuynh/baohuynh/recommendation/The-Movie-Cinema/movielens_dataset/ratings.csv', index=False)
            
            
            user_input_vec = toki.texts_to_matrix([user_data])[0]


            print('[INFO] Recommend vector input: ', user_input_vec)
            print('[INFO] Lenght of input: ', len(user_input_vec))
            # Convert the Tensor to a batch of Tensors and then to a list
            image_tensor = tf.expand_dims(user_input_vec, 0)
            image_tensor = image_tensor.numpy().tolist()
            # Optional: define a custom message lenght in bytes
            MAX_MESSAGE_LENGTH = 20000000

            # Optional: define a request timeout in seconds
            REQUEST_TIMEOUT = 5

            # Open a gRPC insecure channel
            channel = grpc.insecure_channel(
                "localhost:8500",
                options=[
                    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                ],
            )

            # Create the PredictionServiceStub
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

            # Create the PredictRequest and set its values
            req = predict_pb2.PredictRequest()
            req.model_spec.name = 'recommendation_model'
            req.model_spec.signature_name = ''

            # Convert to Tensor Proto and send the request
            # Note that shape is in NHWC (num_samples x height x width x channels) format
            tensor = tf.make_tensor_proto(image_tensor)
            req.inputs["input_1"].CopyFrom(tensor)  # Available at /metadata

            # Send request
            response = stub.Predict(req, REQUEST_TIMEOUT)

            print('[INFO] Sending input to serve model')
            # Handle request's response
            output_tensor_proto = response.outputs["output_1"]  # Available at /metadata
            shape = tf.TensorShape(output_tensor_proto.tensor_shape)

            result = tf.reshape(output_tensor_proto.float_val, shape)
            result = np.array(result)
            print("Result ", result)
            print(np.array(user_input_vec) != 0)
            user_input_not = np.invert(np.array(user_input_vec) != 0)
            print("result_user_input_notnot ", user_input_not)
            print('[INFO] Recived result: ', result[:,user_input_not])
            result = result[:,user_input_not]
            top_movies_index = result.argsort()[:,::-1]
            # top_movies_recommned = []
            movie_cards={}

            top_movie_count = 0
            for i in top_movies_index[0]:
                # print(toki.index_word[i])
                movie_rec_name_ml = movies_id2name_dict[int(toki.index_word[i])]
                print(movie_rec_name_ml+"__score: "+str(result[0][i]))

                if movie_rec_name_ml in movML_to_movDB_dict:
                    # print("[INFO] Added recommendation list")

                    movie_rec_id_db, movie_rec_name_db = movML_to_movDB_dict[movie_rec_name_ml]
                    # print(movie_rec_id_db)
                
                    re = 'https://api.themoviedb.org/3/movie/' + str(movie_rec_id_db) \
                            +'?api_key=332fc08736785fea6eaeb4e722aa9e73&language=en-US'
                    req = requests.get(re)
                    movie_rec_info = json.loads(req.text)

                    if movie_rec_info['poster_path'] != None:

                        rec_posters = 'https://image.tmdb.org/t/p/original' + str(movie_rec_info['poster_path'])
                        rec_movies = movie_rec_info['title']
                        rec_movies_org = movie_rec_info['original_title']
                        rec_vote = movie_rec_info['vote_average']
                        rec_year = movie_rec_info['release_date'][:4]
                        
                        # print('[INFO] Search movie successfully: ', movie_rec_info['poster_path'])
            
                        # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
                        movie_cards[rec_posters] = [rec_movies,rec_movies_org,rec_vote,rec_year]
                        top_movie_count += 1
                        if top_movie_count == 10:
                            print("STOP ", top_movie_count)
                            break
                        # time.sleep(0.5)
                else:
                    print("[WARNING] Could not search the movie info")


            ########################################################################################################################################
            
            # movie_cards = dict(e for i, e in enumerate(movie_cards.items()) if 0 <= i < 10)
            # movie_cards = dict(sorted(movie_cards.items(), key=operator.itemgetter(1), reverse=True)[:10])
            print('[INFO] TOP Recommendation: ', movie_cards)
            # passing all the data to the html file

        except Exception as e:
                print('[ERROR] There was the error in recommend function')
                print('[ERROR] Exception: ', e)
                # return "0"
        return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
                                vote_count=vote_count,release_date=release_date,runtime=runtime,
                                status=status,genres=genres,movie_cards=movie_cards)

if __name__ == '__main__':
    app.run(debug=True)
