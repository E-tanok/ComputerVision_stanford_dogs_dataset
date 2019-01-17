def values_to_col(myDataFrame,myColumnList,bool_with_old_col_name):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    """
    This function goal is to treat categorical features in a pandas DataFrame
    list of columns:
    From a categorical column 'CC' which contains 'N' attributes
    [att1, att2, att3,..., attn ] we create N new vectors/features/columns :
        - row to row, if the category was present at the intersection of 'CC'
        and the row,then the value at the intersection of the row and the new
        column is 1
        - else, the value at the intersection of the row and the new column is 0
    The relation between rows and columns is kept


    RESULT : The entry DataFrame with the new categorical vectors :
    2 new columns are also created :
        - 'created_columns' : a column with a list of all the new created columns
        - 'dict_mapping' : a column with a dictionary which maps the old columns
        with the columns they generated

    PARAMS :
    - 'myDataFrame' refers to the DataFrame we interest in
    - 'myColumnList' refers to the list of columns (the list can have only one
    value but it must be a list) we want to vectorize
    - 'bool_with_old_col_name' is a boolean attribute that specify if we want to
    keep the old columns names or not :
        --> example : with old names, new columns are :
        CC_att1, CC_att2, CC_att3,..., CC_attn
        --> example : without old names : att1, att2, att3,..., attn
    """
    created_columns = []
    dict_mapping = {}
    for column in myColumnList:
        #Missing values filling
        myDataFrame[column].fillna('none', inplace=True)
        newFeatures = []
        corpus = myDataFrame[column]
        vectorizer = CountVectorizer(min_df=1,max_df=1.0)

        #Construction of the row/words Matrix
        X = vectorizer.fit_transform(corpus).toarray()
        feature_names = vectorizer.get_feature_names()

        for feature in feature_names:
            if bool_with_old_col_name==True:
                newFeatureName = '%s_%s'%(column,feature)
            else:
                newFeatureName = feature

            newFeatures.append(newFeatureName)
            created_columns.append(newFeatureName)

            if column in dict_mapping :
                dict_mapping[column].append(newFeatureName)
            else:
                dict_mapping[column] = [newFeatureName]

        #Construction of the row/words DataFrame
        myfeaturedf = pd.DataFrame(X,columns=newFeatures)
        myDataFrame = pd.concat([myDataFrame, myfeaturedf], axis=1, join_axes=[myfeaturedf.index])
        myDataFrame['created_columns']=[created_columns]*len(myDataFrame)
        myDataFrame['dict_mapping']=[dict_mapping]*len(myDataFrame)

    return myDataFrame

def percent_of_total(myDataFrame,myColumnList):
    """
    This function goal is to convert each continuous columns of a determined
    list into a column were the values are the percentage of the sum of all
    columns included in the list.

    RESULT : The entry DataFrame with columns (included in 'myColumnList')
    converted into percentage of their sum.

    PARAMS :
    - 'myDataFrame' refers to the entry myDataFrame.
    - 'myColumnList' refers to the list  of columns with which we want to focus
    the analysis
    """
    myDataFrame['total'] = myDataFrame[myColumnList].sum(1)
    for column in myColumnList:
        myDataFrame[column] = 100*(myDataFrame[column]/ myDataFrame['total'])
    myDataFrame.drop('total',inplace=True,axis=1)
    return myDataFrame

def group_by_frequency(myDataFrame,myColumn):
    import numpy as np
    """
    This function goal is to build an aggregated DataFrame which contains the occurences of the catagorical terms contained in
    'myColumn' args.

    RESULT : an aggregated DataFrame with the occurences of each values.
    - The DataFrame is sorted by descending occurences.
    - It also contains :
        - rank of each category in terms of occurences.
        - cumsum of occurences from the first value to the last one.
        - percent of total occurences covered by the upper categories at a given row.

    PARAMS :
    - 'myDataFrame' : the entry DataFrame
    - 'myColumn' : the column concerned by the frequencies count
    """
    grouped = myDataFrame.copy()
    grouped['occurences'] = 1
    grouped = grouped[[myColumn,'occurences']].groupby(myColumn).sum()
    grouped.sort_values(by='occurences', ascending=False, inplace=True)
    grouped['rank'] = range(1,len(grouped)+1)
    grouped['cumsum'] = np.cumsum(grouped['occurences'])
    grouped['percent_of_total'] = grouped['cumsum']/grouped['occurences'].sum()

    return grouped

def class_my_files(myPath):
    """
    This function goal is to build a dictionnary of all the files available in
    a given repository, based on the files extensionss.

    RESULT : a dictionnary which maps all files to their extensions

    PARAMS :
    - 'myPath' : the path of the repository in which you want to map files.
    """
    from os import listdir
    import re

    L_files = listdir(myPath)
    dict_extensions = {}
    extensions = [r'.csv',r'.xls$',r'.xlsx',r'.json',r'.txt',r'.p$','.jpg']

    for ext in extensions :
        regex = re.compile(ext)
        selected_files = list(filter(regex.search, L_files))

        clean_ext = re.sub('\.|\$','',ext)
        dict_extensions[clean_ext] = selected_files

    return dict_extensions

def convert_in_list(myDataFrame,myColumn):
    from ast import literal_eval
    """
    This function goal is to convert a pandas column into a "list" datatype column
    IMPORTANT : The column values must match with the python lists pattern in order to be read and converted correctly.

    RESULT : The same column, with each value converted into an array : that's also possible to loop over the array values

    PARAMS :
    - myDataFrame : the entry DataFrame
    - myColumn : String, the column to convert
    """

    myDataFrame[myColumn] = myDataFrame[myColumn].apply(literal_eval)
    return myDataFrame
