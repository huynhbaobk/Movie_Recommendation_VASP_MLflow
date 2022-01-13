# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from utils import *
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

set_seed(42)
from urllib.parse import urlparse
import mlflow
import mlflow.keras

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import time
import subprocess
import shutil

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def preprocess_data(data_path):
    m = pd.read_csv(data_path + '/' + 'movies.csv')
    m = m[m.movieId.notnull()].reindex()
    m['itemid']=m.movieId.apply(lambda x: str(int(x)))
    m['product_name'] = m['title']
    items = m[['itemid','product_name','genres']]
    items.to_json(data_path + '/' +'items.json')

    interactions = pd.read_csv(data_path + '/' + 'ratings.csv')
    interactions = interactions[interactions.rating>=4.]
    interactions = interactions.sort_values(['userId','timestamp'])
    interactions['itemid'] = interactions['movieId'].apply(str)
    interactions['userid'] = interactions['userId'].apply(str)
    interactions['amount'] = 1
    interactions['date'] = interactions['timestamp']
    interactions[['itemid','userid','amount','date']]
    interactions.to_json(data_path + '/' + "purchases.json")
    interactions['itemids'] = interactions[['userid','itemid']].groupby(['userid'])['itemid'].transform(lambda x: ','.join(x))
    iii = interactions[['userId','itemids']].drop_duplicates()
    iii.to_json(data_path + '/' +'purchases_txt.json')

    purchases=pd.read_json(data_path + '/' +'purchases.json')
    purchases['userid'] = purchases.userid.apply(str)
    purchases['itemid'] = purchases.itemid.apply(str)
    purchases_item_counts = purchases[['userid','itemid']]
    purchases_user_counts = purchases[['userid','itemid']]
    purchases_user_count = purchases.groupby(['userid']).size().to_frame('nr_of_purchases').reset_index()
    purchases_user_count = purchases_user_count.sort_values(by=['nr_of_purchases'], ascending=False)
    pu5=purchases_user_count[purchases_user_count.nr_of_purchases>=5]
    purchases_pu5 = purchases[purchases.userid.isin(pu5.userid)]
    purchases_item_count_pu5 = purchases_pu5.groupby(['itemid']).size().to_frame('nr_of_purchases').reset_index()
    purchases_item_count_pu5 = purchases_item_count_pu5.sort_values(by=['nr_of_purchases'], ascending=False)
    purchases_pu5.to_json(data_path + '/' +'purchases_pu5.json') 

    purchases_pu5['itemids'] = purchases_pu5[['userid','itemid']].groupby(['userid'])['itemid'].transform(lambda x: ','.join(x))
    iii = purchases_pu5[['userId','itemids']].drop_duplicates()
    iii['userid']=iii['userId'].apply(str)
    iii = iii[['userid','itemids']]
    iii.to_json(data_path + '/' +'purchases_txt_pu5.json')

    iii['userid'].to_frame().to_json(data_path + '/' +'users_pu5.json')
    items[items.itemid.isin(purchases_item_count_pu5.itemid)].to_json(data_path + '/' +"items_pu5.json")
    purchases_item_count_pu5.to_json(data_path + '/' +"items_sorted_pu5.json")
    pu5.to_json(data_path + '/' +"users_sorted_pu5.json")

    users = pd.read_json(data_path + '/' +'users_pu5.json')
    shuffled_users = users.sample(frac=1., random_state=42)
    test_users = shuffled_users.iloc[:600]
    val_users = shuffled_users.iloc[600:1200]
    train_users = shuffled_users.iloc[1200:]

    test_users.to_json(data_path + '/' +"test_users.json")
    val_users.to_json(data_path + '/' +"val_users.json")
    train_users.to_json(data_path + '/' +"train_users.json") 

    print(len(train_users),len(val_users),len(test_users))


class DiagonalToZero(tf.keras.constraints.Constraint):
    def __call__(self, w):
        """Set diagonal to zero"""
        q = tf.linalg.set_diag(w, tf.zeros(w.shape[0:-1]), name=None)
        return q

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a basket."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), stddev=1.)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VASP(Model):
    class Model(tf.keras.Model):
        def __init__(self, num_words, latent=128, hidden=512, items_sampling=1.):
            """
            num_words             nr of items in dataset (size of tokenizer)
            latent                size of latent space
            hidden                size of hid close_fds=True)
den layers
            items_sampling        Large items datatsets can be very gpu memory consuming in EASE layer.
                                  This coefficient reduces number of ease parametrs by taking only
                                  fraction of items sorted by popularity as input for model.
                                  Note: This coef should be somewhere around coverage@100 achieved by full
                                  size model.
                                  For ML20M this coef should be between 0.4888 (coverage@100 for full model)
                                  and 1.0
                                  For Netflix this coef should be between 0.7055 (coverage@100 for full
                                  model) and 1.0
            """
            super(VASP.Model, self).__init__()

            self.sampled_items = int(num_words * items_sampling)

            assert self.sampled_items > 0
            assert self.sampled_items <= num_words

            self.s = self.sampled_items < num_words

            # ************* ENCODER ***********************
            self.encoder1 = tf.keras.layers.Dense(hidden)
            self.ln1 = tf.keras.layers.LayerNormalization()
            self.encoder2 = tf.keras.layers.Dense(hidden)
            self.ln2 = tf.keras.layers.LayerNormalization()
            self.encoder3 = tf.keras.layers.Dense(hidden)
            self.ln3 = tf.keras.layers.LayerNormalization()
            self.encoder4 = tf.keras.layers.Dense(hidden)
            self.ln4 = tf.keras.layers.LayerNormalization()
            # self.encoder5 = tf.keras.layers.Dense(hidden)
            # self.ln5 = tf.keras.layers.LayerNormalization()
            # self.encoder6 = tf.keras.layers.Dense(hidden)
            # self.ln6 = tf.keras.layers.LayerNormalization()
            # self.encoder7 = tf.keras.layers.Dense(hidden)
            # self.ln7 = tf.keras.layers.LayerNormalization()

            # ************* SAMPLING **********************
            self.dense_mean = tf.keras.layers.Dense(latent,
                                                    name="Mean")
            self.dense_log_var = tf.keras.layers.Dense(latent,
                                                       name="log_var")

            self.sampling = Sampling(name='Sampler')

            # ************* DECODER ***********************
            self.decoder1 = tf.keras.layers.Dense(hidden)
            self.dln1 = tf.keras.layers.LayerNormalization()
            self.decoder2 = tf.keras.layers.Dense(hidden)
            self.dln2 = tf.keras.layers.LayerNormalization()
            self.decoder3 = tf.keras.layers.Dense(hidden)
            self.dln3 = tf.keras.layers.LayerNormalization()
            # self.decoder4 = tf.keras.layers.Dense(hidden)
            # self.dln4 = tf.keras.layers.LayerNormalization()
            # self.decoder5 = tf.keras.layers.Dense(hidden)
            # self.dln5 = tf.keras.layers.LayerNormalization()

            self.decoder_resnet = tf.keras.layers.Dense(self.sampled_items,
                                                        activation='sigmoid',
                                                        name="DecoderR")
            self.decoder_latent = tf.keras.layers.Dense(self.sampled_items,
                                                        activation='sigmoid',
                                                        name="DecoderL")

            # ************* PARALLEL SHALLOW PATH *********

            self.ease = tf.keras.layers.Dense(
                self.sampled_items,
                activation='sigmoid',
                use_bias=False,
                kernel_constraint=DiagonalToZero(),  # critical to prevent learning simple identity
            )

        def call(self, x, training=None):
            sampling = self.s
            if sampling:
                sampled_x = x[:, :self.sampled_items]
                non_sampled = x[:, self.sampled_items:] * 0.
            else:
                sampled_x = x

            z_mean, z_log_var, z = self.encode(sampled_x)
            if training:
                d = self.decode(z)
                # Add KL divergence regularization loss.
                kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                kl_loss = tf.reduce_mean(kl_loss)
                kl_loss *= -0.5
                self.add_loss(kl_loss)
                self.add_metric(kl_loss, name="kl_div")
            else:
                d = self.decode(z)

            if sampling:
                d = tf.concat([d, non_sampled], axis=-1)

            ease = self.ease(sampled_x)

            if sampling:
                ease = tf.concat([ease, non_sampled], axis=-1)

            return d * ease

        def decode(self, x):
            e0 = x
            e1 = self.dln1(tf.keras.activations.swish(self.decoder1(e0)))
            e2 = self.dln2(tf.keras.activations.swish(self.decoder2(e1) + e1))
            e3 = self.dln3(tf.keras.activations.swish(self.decoder3(e2) + e1 + e2))
            # e4 = self.dln4(tf.keras.activations.swish(self.decoder4(e3) + e1 + e2 + e3))
            # e5 = self.dln5(tf.keras.activations.swish(self.decoder5(e4) + e1 + e2 + e3 + e4))

            dr = self.decoder_resnet(e2)
            dl = self.decoder_latent(x)

            return dr * dl

        def encode(self, x):
            e0 = x
            e1 = self.ln1(tf.keras.activations.swish(self.encoder1(e0)))
            e2 = self.ln2(tf.keras.activations.swish(self.encoder2(e1) + e1))
            e3 = self.ln3(tf.keras.activations.swish(self.encoder3(e2) + e1 + e2))
            e4 = self.ln4(tf.keras.activations.swish(self.encoder4(e3) + e1 + e2 + e3))
            # e5 = self.ln5(tf.keras.activations.swish(self.encoder5(e4) + e1 + e2 + e3 + e4))
            # e6 = self.ln6(tf.keras.activations.swish(self.encoder6(e5) + e1 + e2 + e3 + e4 + e5))
            # e7 = self.ln7(tf.keras.activations.swish(self.encoder7(e6) + e1 + e2 + e3 + e4 + e5 + e6))

            z_mean = self.dense_mean(e4)
            z_log_var = self.dense_log_var(e4)
            z = self.sampling((z_mean, z_log_var))

            return z_mean, z_log_var, z

    def create_model(self, latent=256, hidden=512, ease_items_sampling=1., summary=False):
        self.model = VASP.Model(self.dataset.num_words, latent, hidden, ease_items_sampling)
        self.model(self.split.train_gen[0][0])
        if summary:
            self.model.summary()
        self.mc = MetricsCallback(self)

    def compile_model(self, lr=0.00002, fl_alpha=0.25, fl_gamma=2.0):
        """
        lr         learning rate of Nadam optimizer
        fl_alpha   alpha parameter of focal crossentropy
        fl_gamma   gamma parameter of focal crossentropy
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Nadam(lr),
            loss=lambda x, y: tfa.losses.sigmoid_focal_crossentropy(x, y, alpha=fl_alpha, gamma=fl_gamma),
            metrics=['mse', cosine_loss]
        )

    def train_model(self, epochs=150):
        self.model.fit(
            self.split.train_gen,
            validation_data=self.split.validation_gen,
            epochs=epochs,
            callbacks=[self.mc]
        )
        print(self.model.metrics)


if __name__ == "__main__":
    # while True:
        dataset_path = '/home/baohuynh/baohuynh/recommendation/The-Movie-Cinema/movielens_dataset/'

        preprocess_data(dataset_path)

        warnings.filterwarnings("ignore")
        np.random.seed(40)

        dataset = Data(d=dataset_path, pruning='u5')
        dataset.splits = []
        dataset.create_splits(1, 2000, shuffle=False, generators=False)
        dataset.split.train_users = pd.read_json(dataset_path + "train_users.json").userid.apply(str).to_frame()
        dataset.split.validation_users = pd.read_json(dataset_path + "val_users.json").userid.apply(str).to_frame()
        dataset.split.test_users = pd.read_json(dataset_path + "test_users.json").userid.apply(str).to_frame()
        dataset.split.generators()


        with mlflow.start_run():
            m = VASP(dataset.split, name="VASP_ML20_1")
            m.create_model(latent=2048, hidden=4096, ease_items_sampling=0.33)
            m.model.summary()
            print("=" * 80)
            print("Train for 50 epochs with lr 0.00005")
            m.compile_model(lr=0.00005, fl_alpha=0.25, fl_gamma=2.0)
            m.train_model(50)
            print("=" * 80)
            print("Than train for 20 epochs with lr 0.00001")
            m.compile_model(lr=0.00001, fl_alpha=0.25, fl_gamma=2.0)
            m.train_model(25)
            print("=" * 80)
            print("Than train for 20 epochs with lr 0.000001")
            m.compile_model(lr=0.00001, fl_alpha=0.25, fl_gamma=2.0)
            m.train_model(25)

            m.mc.plot_history().figure.savefig("training_result.png")

            test_metrics = m.test_model()
            print('[INFO] Test metrics: ', test_metrics)

            mlflow.log_param("fl_alpha", 0.25)
            mlflow.log_param("fl_gamma", 2.0)
            mlflow.log_metric("Recall_20", test_metrics[0])
            mlflow.log_metric("Recall_50", test_metrics[1])
            mlflow.log_metric("NCDG_100", test_metrics[2])
            mlflow.log_artifact("training_result.png")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                print('MODEL REGISTRY')
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(m.model, "model", registered_model_name="RecommendationModel")
            else:
                print('FILE')
                mlflow.keras.log_model(m.model, "model")

        time.sleep(10)

        client = MlflowClient()
        # Parametrizing the right experiment path using widgets
        experiment_name = 'Default'
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_ids = [experiment.experiment_id]
        print("Experiment IDs:", experiment_ids)

        # Setting the decision criteria for a best run
        query = "metrics.NCDG_100 > 0.1"
        runs = client.search_runs(experiment_ids, query, ViewType.ALL)

        # Searching throught filtered runs to identify the best_run and build the model URI to programmatically reference later
        accuracy_high = None
        best_run = None
        for run in runs:
            if (accuracy_high == None or run.data.metrics['NCDG_100'] > accuracy_high):
                accuracy_high = run.data.metrics['NCDG_100']
                best_run = run
        print('Highest Accuracy: ', accuracy_high)
        run_id = best_run.info.run_id
        print('Run ID: ', run_id)

        model_uri = "runs:/" + run_id + "/model"
        print('model_uri', model_uri)

        # Check if model is already registered
        model_name = "BestRecommmendationModel"
        try:
            registered_model = client.get_registered_model(model_name)
        except:
            registered_model = client.create_registered_model(model_name)

        # Create the model source
        model_source = f"{best_run.info.artifact_uri}/model"
        print('model_source', model_source)

        # Archive old production model
        max_version = 0
        for mv in client.search_model_versions("name='BestRecommmendationModel'"):
            # print('search model', mv)
            current_version = int(dict(mv)['version'])
            if current_version > max_version:
                max_version = current_version
            if dict(mv)['current_stage'] == 'Production':
                version = dict(mv)['version']
                client.transition_model_version_stage(model_name, version, stage='Archived')

        # Create a new version for this model with best metric (accuracy)
        client.create_model_version(model_name, model_source, run_id)
        # Check the status of the created model version (it has to be READY)
        status = None
        while status != 'READY':
            for mv in client.search_model_versions(f"run_id='{run_id}'"):
                status = mv.status if int(mv.version)==max_version + 1 else status
            time.sleep(5)

        # Promote the model version to production stage
        client.transition_model_version_stage(model_name, max_version + 1, stage='Production')


        ### Deploy tensorflow serving
        print("[INFO] Deploy tensorflow serving")
        os.system('kill $(lsof -t -i:8501)') 

        dest = '/home/baohuynh/baohuynh/recommendation/The-Movie-Cinema/model_deploy/1/'
        shutil.copytree(model_source.replace('file://', "") + '/data/model/', dest, dirs_exist_ok=True)
        

        subprocess.Popen(["bash","runserver.sh"], close_fds=True)

        print("[INFO] Sleeping...")
        # time.sleep(5)
        

        # subprocess.Popen(["python","main.py"], close_fds=True)