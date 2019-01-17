def select_N_random_races(N_FILTERED_RACES):
    import pickle
    import random
    import numpy as np
    from context import datasources_path, pickles_path

    races_path = datasources_path+"Images\\"
    dict_race_pictures_train = pickle.load(open(pickles_path+"dict_race_pictures_train.p", "rb" ))
    dict_race_pictures_test = pickle.load(open(pickles_path+"dict_race_pictures_test.p", "rb" ))

    L_races = pickle.load(open(pickles_path+"L_races.p", "rb" ))
    N_RACES = len(L_races)
    L_races_indexes = list(range(0,N_RACES))
    L_filtered_indexes = random.sample(L_races_indexes, N_FILTERED_RACES)
    L_filtered_races = [L_races[i] for i in L_filtered_indexes]
    L_filtered_races = list(np.sort(L_filtered_races))

    return(L_filtered_races)


def preprocess_for_vgg16(my_image_path):
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.vgg16 import preprocess_input

    img = load_img(my_image_path, target_size=(224, 224))  # Charger l'image
    img = img_to_array(img)  # Convertir en tableau numpy
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
    img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16
    return img


def build_train_and_test_datasets(L_filtered_races,label_encoder_name,bool_augmented):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import pickle
    from context import datasources_path, pickles_path

    dict_train_val_test = {}

    L_picture_paths_train = []
    L_picture_paths_test = []

    if bool_augmented == False:
        dict_race_pictures_train = pickle.load(open(pickles_path+"dict_race_pictures_train.p", "rb" ))
        dict_race_pictures_test = pickle.load(open(pickles_path+"dict_race_pictures_test.p", "rb" ))
        race_path = datasources_path+"Images\\"

    elif bool_augmented == True:
        dict_race_pictures_train = pickle.load(open(pickles_path+"dict_augmented_race_pictures_train.p", "rb" ))
        dict_race_pictures_test = pickle.load(open(pickles_path+"dict_augmented_race_pictures_test.p", "rb" ))
        race_path = datasources_path+"Images_augmented\\"

    for race in L_filtered_races:
        L_picture_names_train = dict_race_pictures_train[race]
        L_picture_names_test = dict_race_pictures_test[race]

        L_race_picture_paths_train = [race_path+race+"\\"+picture for picture in L_picture_names_train]
        L_race_picture_paths_test = [race_path+race+"\\"+picture for picture in L_picture_names_test]

        L_picture_paths_train+= L_race_picture_paths_train
        L_picture_paths_test+= L_race_picture_paths_test

    #Building the data train list :
    train_list = []
    for picture_path in L_picture_paths_train:
        img_for_vgg16 = preprocess_for_vgg16(picture_path)
        train_list.append(img_for_vgg16[0])
    X_train = np.asarray(train_list)

    #Building the data test list :
    test_list = []
    for picture_path in L_picture_paths_test:
        img_for_vgg16 = preprocess_for_vgg16(picture_path)
        test_list.append(img_for_vgg16[0])
    X_test = np.asarray(test_list)

    #Buiding the train and test labels :
    y_train_raw = []
    y_test_raw = []
    for race in L_filtered_races:
        for picture in dict_race_pictures_train[race]:
            y_train_raw.append(race)
        for picture in dict_race_pictures_test[race]:
            y_test_raw.append(race)


    labelencoder = LabelEncoder()
    L_encoded_races = labelencoder.fit_transform(L_filtered_races)
    pickle.dump(labelencoder,open(pickles_path+label_encoder_name+".p", "wb"))

    #using label encoders for the races :
    y_train_raw = labelencoder.transform(y_train_raw)
    y_test_raw = labelencoder.transform(y_test_raw)

    #converting the encoded values in string in order to use them in the CountVectorizer :
    y_train_raw = ['idx_'+str(idx) for idx in y_train_raw]
    y_test_raw = ['idx_'+str(idx) for idx in y_test_raw]

    #using the CountVectorizer :
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    y_train = vectorizer.fit_transform(y_train_raw).toarray()
    feature_names = vectorizer.get_feature_names()
    y_test = vectorizer.transform(y_test_raw).toarray()

    dict_train_val_test['X_train'] = X_train
    dict_train_val_test['X_test'] = X_test
    dict_train_val_test['y_train'] = y_train
    dict_train_val_test['y_test'] = y_test

    return dict_train_val_test


def build_train_validation_and_test_datasets(L_filtered_races,label_encoder_name,vectorizer_name):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import pickle
    from context import datasources_path, pickles_path
    import cv2

    dict_train_val_test = {}

    L_picture_paths_train = []
    L_picture_paths_validation = []
    L_picture_paths_test = []

    dict_data = pickle.load(open(pickles_path+"dict_data.p", "rb" ))
    dict_race_pictures_train = dict_data['train_data_with_validation']
    dict_race_pictures_validation = dict_data['validation_data']
    dict_race_pictures_test = dict_data['test_data']

    race_path = datasources_path+"Images\\"

    for race in L_filtered_races:
        L_picture_names_train = dict_race_pictures_train[race]
        L_picture_names_validation = dict_race_pictures_validation[race]
        L_picture_names_test = dict_race_pictures_test[race]

        L_race_picture_paths_train = [race_path+race+"\\"+picture for picture in L_picture_names_train]
        L_race_picture_paths_validation = [race_path+race+"\\"+picture for picture in L_picture_names_validation]
        L_race_picture_paths_test = [race_path+race+"\\"+picture for picture in L_picture_names_test]

        L_picture_paths_train+= L_race_picture_paths_train
        L_picture_paths_validation+= L_race_picture_paths_validation
        L_picture_paths_test+= L_race_picture_paths_test

    #Building the data train list :
    raw_train_list = []
    train_list = []
    train_labels = []
    for picture_path in L_picture_paths_train:
        train_labels.append(picture_path.split('\\')[-1])
        img_raw = cv2.imread(picture_path)
        raw_train_list.append(img_raw)
        img_for_vgg16 = preprocess_for_vgg16(picture_path)
        train_list.append(img_for_vgg16[0])
    X_train = np.asarray(train_list)
    X_train_raw = np.asarray(raw_train_list)

    #Building the data validation list :
    raw_val_list = []
    val_list = []
    val_labels = []
    for picture_path in L_picture_paths_validation:
        val_labels.append(picture_path.split('\\')[-1])
        img_raw = cv2.imread(picture_path)
        raw_val_list.append(img_raw)
        img_for_vgg16 = preprocess_for_vgg16(picture_path)
        val_list.append(img_for_vgg16[0])
    X_val = np.asarray(val_list)
    X_val_raw = np.asarray(raw_val_list)

    #Building the data test list :
    raw_test_list = []
    test_list = []
    test_labels = []
    for picture_path in L_picture_paths_test:
        test_labels.append(picture_path.split('\\')[-1])
        img_raw = cv2.imread(picture_path)
        raw_test_list.append(img_raw)
        img_for_vgg16 = preprocess_for_vgg16(picture_path)
        test_list.append(img_for_vgg16[0])
    X_test = np.asarray(test_list)
    X_test_raw = np.asarray(raw_test_list)

    #Buiding the train, validation and test labels :
    y_train_raw = []
    y_val_raw = []
    y_test_raw = []
    for race in L_filtered_races:
        for picture in dict_race_pictures_train[race]:
            y_train_raw.append(race)
        for picture in dict_race_pictures_validation[race]:
            y_val_raw.append(race)
        for picture in dict_race_pictures_test[race]:
            y_test_raw.append(race)


    labelencoder = LabelEncoder()
    L_encoded_races = labelencoder.fit_transform(L_filtered_races)
    pickle.dump(labelencoder,open(pickles_path+label_encoder_name+".p", "wb"))

    #using label encoders for the races :
    y_train_raw = labelencoder.transform(y_train_raw)
    y_val_raw = labelencoder.transform(y_val_raw)
    y_test_raw = labelencoder.transform(y_test_raw)

    #converting the encoded values in string in order to use them in the CountVectorizer :
    y_train_raw = ['idx_'+str(idx) for idx in y_train_raw]
    y_val_raw = ['idx_'+str(idx) for idx in y_val_raw]
    y_test_raw = ['idx_'+str(idx) for idx in y_test_raw]

    #using the CountVectorizer :
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    y_train = vectorizer.fit_transform(y_train_raw).toarray()
    feature_names = vectorizer.get_feature_names()
    y_val = vectorizer.transform(y_val_raw).toarray()
    y_test = vectorizer.transform(y_test_raw).toarray()
    #dumping the vectorizer object :
    pickle.dump(vectorizer,open(pickles_path+vectorizer_name+".p", "wb"))

    dict_train_val_test['train_labels'] = train_labels
    dict_train_val_test['val_labels'] = val_labels
    dict_train_val_test['test_labels'] = test_labels

    dict_train_val_test['X_train_raw'] = X_train_raw
    dict_train_val_test['X_val_raw'] = X_val_raw
    dict_train_val_test['X_test_raw'] = X_test_raw

    dict_train_val_test['X_train'] = X_train
    dict_train_val_test['X_val'] = X_val
    dict_train_val_test['X_test'] = X_test

    dict_train_val_test['y_train'] = y_train
    dict_train_val_test['y_val'] = y_val
    dict_train_val_test['y_test'] = y_test
    pickle.dump(dict_train_val_test,open(pickles_path+"dict_train_val_test.p", "wb"))

    return dict_train_val_test

def predict_dog_race(img):
    import numpy as np
    import pandas as pd
    from keras.models import model_from_json
    from keras import backend as K
    import pickle
    from context import pickles_path

    df_prediction = pd.DataFrame(columns=['prediction_score','index','race'])
    L_prediction = []

    if isinstance(img,np.ndarray):
        if img.shape==(224, 224, 3):
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        elif img.shape==(1, 224, 224, 3):
            img = img
        else:
            return("BAD IMAGE SHAPE! SHAPE MUST BE (224, 224, 3) or (1, 224, 224, 3)")

        # load json and create model
        json_file = open('CNN_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("CNN_model.h5")
        print("Loaded model from disk")
        prediction = loaded_model.predict(img)
        K.clear_session()
        print(prediction)

        #Loading the label_encoder object :
        labelencoder = pickle.load(open(pickles_path+"label_encoder_final_model.p", "rb" ))

        #Mapping indexes, scores and races with pandas
        for index,score in enumerate(prediction[0]):
            dict_prediction_unsorted = {'prediction_score':score, 'index':index, 'race':labelencoder.inverse_transform([index])[0]}
            df_prediction = df_prediction.append(dict_prediction_unsorted, ignore_index=True)

        #Sorting the final races by prediction scores :
        df_prediction.sort_values(by='prediction_score', ascending=False, inplace=True)
        for race, score in zip(df_prediction['race'], df_prediction['prediction_score']):
            L_prediction.append((race, score))

        print('Prediction over!')
        return(L_prediction)
    else:
        return("BAD PICTURE TYPE ! PICTURE MUST BE A numpy.ndarray")
