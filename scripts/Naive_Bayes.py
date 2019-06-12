#### Diwakar Mohan

import numpy as np
from nltk.tokenize import RegexpTokenizer
import time
import random
from collections import Counter
import sys
import pickle

tokenizer = RegexpTokenizer(r'\w+')
def getWordFreq(review,wordfreq,isStemmedData):
    if not isStemmedData:
        raw = review.lower()
        raw = raw.replace('<br /><br />', ' ')
        tokens = tokenizer.tokenize(raw)
        words = ' '.join(tokens)
        word_list = words.split()
        word_count = len(word_list)
    else:
        word_list = review.split()
        word_count = len(word_list)
        
    key = "wordcount"
    for word in word_list:
        wordfreq[word] += 1

    wordfreq[key] += word_count

def getWordFreqBest(review,wordfreq,isStemmedData):
    if not isStemmedData:
        raw = review.lower()
        raw = raw.replace('<br /><br />', ' ')
        tokens = tokenizer.tokenize(raw)
        words = ' '.join(tokens)
        word_list = words.split()
        word_count = len(word_list)
    else:
        word_list = review.split()
        word_count = len(word_list)
        
    key = "wordcount"
    for i in range(word_count-1):
        word = word_list[i] + '_' + word_list[i+1]
        wordfreq[word] += 1

    wordfreq[key] += word_count
    
def getRating(review, catagory, vocab_len,num_cat, totalExamples,isStemmedData):
    if not isStemmedData:
        raw = review.lower()
        raw = raw.replace('<br /><br />', ' ')
        tokens = tokenizer.tokenize(raw)
        words = ' '.join(tokens)
        word_list = words.split()
    else:
        word_list = review.split()
        
    key = "wordcount"
        
    logthetaK = 0.0
    for word in word_list:
        freq = catagory[word]                
        if logthetaK == 0.0:
            logthetaK = (np.log(freq + 1.0 ) - np.log(catagory[key] + vocab_len))
        else:
            logthetaK += (np.log(freq + 1.0 ) - np.log(catagory[key] + vocab_len))

    with np.errstate(divide='ignore'):
        prob = np.log(num_cat) + logthetaK - np.log(totalExamples)
    return prob

def getRatingBest(review, catagory, vocab_len,num_cat, totalExamples,isStemmedData):
    if not isStemmedData:
        raw = review.lower()
        raw = raw.replace('<br /><br />', ' ')
        tokens = tokenizer.tokenize(raw)
        words = ' '.join(tokens)
        word_list = words.split()
    else:
        word_list = review.split()

    word_list_len = len(word_list)
    key = "wordcount"
        
    logthetaK = 0.0

    for i in range(word_list_len-1):
        word = word_list[i] + '_' + word_list[i+1]
        freq = catagory[word]                
        if logthetaK == 0.0:
            logthetaK = (np.log(freq + 1.0 ) - np.log(catagory[key] + vocab_len))
        else:
            logthetaK += (np.log(freq + 1.0 ) - np.log(catagory[key] + vocab_len))

    with np.errstate(divide='ignore'):
        prob = np.log(num_cat) + logthetaK - np.log(totalExamples)
    return prob

def getRandomAccuracy(test_text_file,test_label_file):
    match_random = 0
    count_random = 0
    for line in test_text_file:
        predict_class = random.randint(1,8)
        if predict_class > 4:
            predict_class += 2
    
        label_line_test = test_label_file.readline()
        catagory_class = int(label_line_test)
        if catagory_class == predict_class:
            match_random += 1
        count_random += 1

    accuracy = (match_random * 100.0) / count_random
    return accuracy
    
def getMaxAccuracy(test_text_file,test_label_file,ratings):
    predict_class = np.argmax(ratings) + 1
    match_max = 0
    count_max = 0
    for line in test_text_file:
        label_line_test = test_label_file.readline()
        catagory_class = int(label_line_test)
        if catagory_class == predict_class:
            match_max += 1
        count_max += 1

    accuracy = (match_max * 100.0) / count_max
    return accuracy

def trainModel(train_text_file,train_label_file,isStemmedData):
    dict_list = [Counter() for x in range(10)]
    ratings = [0 for x in range(10)]
    labels = []
    vocab = Counter()
    vlen = 0

    for line in train_text_file:
        label_line = train_label_file.readline()
        getWordFreq(line, dict_list[int(label_line)-1],isStemmedData)
        ratings[int(label_line)-1] += 1
        labels.append(int(label_line))

    for i in range(10):
        vocab += dict_list[i]

    vlen = len(vocab)
    m = np.size(labels)

    return dict_list,ratings,labels,vlen,m

def trainModelBest(train_text_file,train_label_file,isStemmedData):
    dict_list = [Counter() for x in range(10)]
    ratings = [0 for x in range(10)]
    labels = []
    vocab = Counter()
    vlen = 0

    for line in train_text_file:
        label_line = train_label_file.readline()
        getWordFreqBest(line, dict_list[int(label_line)-1],isStemmedData)
        ratings[int(label_line)-1] += 1
        labels.append(int(label_line))

    for i in range(10):
        vocab += dict_list[i]

    vlen = len(vocab)
    m = np.size(labels)

    return dict_list,ratings,labels,vlen,m

def predictTrainAccuracy(train_text_file,dict_list,ratings,labels,vlen,m,isStemmed):
    match_train = 0
    count_train = 0
    for line in train_text_file:
        prob_array = []
        for i in range(np.size(ratings)):
            prob1 = getRating(line, dict_list[i], vlen, ratings[i], m,isStemmed)
            prob_array.append(prob1)
        
        index = np.argmax(prob_array)
        if (labels[count_train] == (index+1)):
            match_train += 1
        count_train += 1

    accuracy = (match_train * 100.0) / count_train
    return accuracy

def predictTrainAccuracyBest(train_text_file,dict_list,ratings,labels,vlen,m,isStemmed):
    match_train = 0
    count_train = 0
    for line in train_text_file:
        prob_array = []
        for i in range(np.size(ratings)):
            prob1 = getRatingBest(line, dict_list[i], vlen, ratings[i], m,isStemmed)
            prob_array.append(prob1)
        
        index = np.argmax(prob_array)
        if (labels[count_train] == (index+1)):
            match_train += 1
        count_train += 1

    accuracy = (match_train * 100.0) / count_train
    return accuracy

def predictTestAccuracy(test_text_file,test_label_file,dict_list,ratings,vlen,m,isStemmed,outfile):
    is_out_file = False
    if outfile != '':
        out_file_fp = open(outfile,'w')
        is_out_file = True

    confusion_matrix = np.matrix(np.zeros([8,8]))
    match_test = 0
    count_test = 0
    accuracy = 0

    for line in test_text_file:
        prob_array_test = []
        for i in range(np.size(ratings)):
            prob1_test = getRating(line, dict_list[i], vlen, ratings[i],m,isStemmed)
            prob_array_test.append(prob1_test)
        
        index = np.argmax(prob_array_test)
        if is_out_file:
            out_file_fp.write(str(index+1)+'\n')
        else:
            label_line_test = test_label_file.readline()
            catagory_class = int(label_line_test)            
            if (catagory_class == (index+1)):
                match_test += 1
            count_test += 1

            if index > 4:
                index = index - 2
            if catagory_class > 4:
                catagory_class = catagory_class - 2
            confusion_matrix[index,catagory_class-1] += 1

    if is_out_file:
        out_file_fp.close()
    else:
        accuracy = (match_test * 100.0) / count_test
        
    return accuracy, confusion_matrix

def predictTestAccuracyBest(test_text_file,test_label_file,dict_list,ratings,vlen,m,isStemmed,outfile):
    is_out_file = False
    if outfile != '':
        out_file_fp = open(outfile,'w')
        is_out_file = True

    confusion_matrix = np.matrix(np.zeros([8,8]))
    match_test = 0
    count_test = 0
    accuracy = 0

    for line in test_text_file:
        prob_array_test = []
        for i in range(np.size(ratings)):
            prob1_test = getRatingBest(line, dict_list[i], vlen, ratings[i],m,isStemmed)
            prob_array_test.append(prob1_test)
        
        index = np.argmax(prob_array_test)
        if is_out_file:
            out_file_fp.write(str(index+1)+'\n')
        else:
            label_line_test = test_label_file.readline()
            catagory_class = int(label_line_test)            
            if (catagory_class == (index+1)):
                match_test += 1
            count_test += 1

            if index > 4:
                index = index - 2
            if catagory_class > 4:
                catagory_class = catagory_class - 2
            confusion_matrix[index,catagory_class-1] += 1

    if is_out_file:
        out_file_fp.close()
    else:
        accuracy = (match_test * 100.0) / count_test
        
    return accuracy, confusion_matrix
        
#==================================================
#read the dataset from the files
nargs = 0
for arg in sys.argv:
    nargs += 1

if nargs == 1:
    train_text_file = open( 'imdb_train_text.txt', 'r')
    train_label_file = open('imdb_train_labels.txt', 'r')

    #=============================================
    
    print("(a): TRAIN MODEL")
    DICT_LIST,RATINGS,LABELS,VLEN,M = trainModel(train_text_file,train_label_file,False)
    DICT_BOOL = False
    print("\n")

    print("(a): STORING MODEL DATA TO FILE MODEL1")
    model1_file = open("MODEL1","wb")
    MODEL1_DATA = [DICT_LIST,RATINGS,LABELS,VLEN,M,DICT_BOOL]
    pickle.dump(MODEL1_DATA,model1_file)
    model1_file.close()
    print("\n")
    
    #=============================================

    print("(a): TRAINING DATA PREDICTION")
    train_text_file.seek(0)
    train_label_file.seek(0)
    
    accuracy = predictTrainAccuracy(train_text_file,DICT_LIST,RATINGS,LABELS,VLEN,M,False)
    print("(a): Training data accuracy=", accuracy)

    train_text_file.close()
    train_label_file.close()
    print("\n")
 
    #=============================================
    
    print("(a): TEST DATA PREDICTION")
    test_label_file = open('imdb_test_labels.txt', 'r')
    test_text_file = open( 'imdb_test_text.txt', 'r')

    accuracy,confusion_mat = predictTestAccuracy(test_text_file,test_label_file,DICT_LIST,RATINGS,VLEN,M,False,'')
    print("(a): Testing data accuracy =", accuracy)
    print("(c): Confusion_matrix=",confusion_mat)
    print("\n")

    #==================================================

    print("(b): RANDOM DATA PREDICTION")
    test_text_file.seek(0)
    test_label_file.seek(0)

    accuracy = getRandomAccuracy(test_text_file,test_label_file)
    print("(b): Test data accuracy random =", accuracy)
    print("\n")

    #==================================================

    print("(b): MAX OCCURANCE CLASS PREDICTION")
    test_text_file.seek(0)
    test_label_file.seek(0)

    accuracy = getMaxAccuracy(test_text_file,test_label_file,RATINGS)
    print("(b): Test data accuracy max =", accuracy)

    test_label_file.close()
    test_text_file.close()
    print("\n")
    
    #==================================================
    
    trans_train_text_file = open( 'imdb_train_text_trans.txt', 'r')
    trans_train_label_file = open('imdb_train_labels.txt', 'r')
    
    print("(d): TRAIN MODEL FOR STEMMED DATA")
    DICT_LIST_TRANS,RATINGS_TRANS,LABELS_TRANS,VLEN_TRANS,M_TRANS = trainModel(trans_train_text_file,trans_train_label_file,True)
    DICT_BOOL_TRANS = True
    print("\n")

    print("(d): STORING MODEL DATA TO FILE MODEL2")
    model2_file = open("MODEL2","wb")
    MODEL2_DATA = [DICT_LIST_TRANS,RATINGS_TRANS,LABELS_TRANS,VLEN_TRANS,M_TRANS,DICT_BOOL_TRANS]
    pickle.dump(MODEL2_DATA,model2_file)
    model2_file.close()
    print("\n")
    
    print("(d): STEMMED TRAINING DATA PREDICTION")
    trans_train_text_file.seek(0)
    trans_train_label_file.seek(0)    
    accuracy = predictTrainAccuracy(trans_train_text_file,DICT_LIST_TRANS,RATINGS_TRANS,LABELS_TRANS,VLEN_TRANS,M_TRANS,True)
    print("(d): Stemmed training data accuracy=", accuracy)
    print("\n")
    
    print("(d): STEMMED TEST DATA PREDICTION")
    
    trans_test_label_file = open('imdb_test_labels.txt', 'r')
    trans_test_text_file = open( 'imdb_test_text_trans.txt', 'r')
    
    accuracy,confusion_mat = predictTestAccuracy(trans_test_text_file,trans_test_label_file,DICT_LIST_TRANS,RATINGS_TRANS,VLEN_TRANS,M_TRANS,True,'')
    print("(d): Stemmed testing data accuracy =", accuracy)
    print("\n")

    #==================================================

    trans_train_text_file.seek(0)
    trans_train_label_file.seek(0)
    trans_test_text_file.seek(0)
    trans_test_label_file(0)
    
    print("(e): TRAIN MODEL USING BEST ALGO")
    DICT_LIST_BEST,RATINGS_BEST,LABELS_BEST,VLEN_BEST,M_BEST = trainModelBest(trans_train_text_file,trans_train_label_file,True)
    DICT_BOOL_BEST = True
    print("\n")

    print("(d): STORING MODEL DATA TO FILE MODEL3")
    model3_file = open("MODEL3","wb")
    MODEL3_DATA = [DICT_LIST_BEST,RATINGS_BEST,LABELS_BEST,VLEN_BEST,M_BEST,DICT_BOOL_BEST]
    pickle.dump(MODEL3_DATA,model3_file)
    model3_file.close()
    print("\n")
    
    print("(d): BEST TRAINING DATA PREDICTION")
    trans_train_text_file.seek(0)
    trans_train_label_file.seek(0)    
    accuracy = predictTrainAccuracyBest(trans_train_text_file,DICT_LIST_BEST,RATINGS_BEST,LABELS_BEST,VLEN_BEST,M_BEST,True)
    print("(d): Best training data accuracy=", accuracy)
    print("\n")
    
    print("(d): BEST TEST DATA PREDICTION")
    trans_test_label_file = open('imdb_test_labels.txt', 'r')
    trans_test_text_file = open( 'imdb_test_text_trans.txt', 'r')
    accuracy,confusion_mat = predictTestAccuracyBest(trans_test_text_file,trans_test_label_file,DICT_LIST_BEST,RATINGS_BEST,VLEN_BEST,M_BEST,True,'')
    print("(d): Best testing data accuracy =", accuracy)
    print("\n")
    
    
    trans_train_text_file.close()
    trans_train_label_file.close()
    trans_test_label_file.close()
    trans_test_text_file.close()    
else:
    model_ = sys.argv[1]
    input_ = sys.argv[2]
    output_ = sys.argv[3]
    is_best_model = sys.argv[4]

    BEST_MODEL = False
    if is_best_model == "true":
        BEST_MODEL = True    

    print("LOAD MODEL DATA FROM MODEL FILE")
    model_file = open(model_,"rb")
    data = pickle.load(model_file)
    model_file.close()

    DICT_LIST_MODEL,RATINGS_MODEL,LABEL_MODEL,VLEN_MODEL,M_MODEL,MODEL_BOOL = data
    print("\n")
    
    input_file = open(input_,'r')
    print("GENERATING OUTPUT FILE")
    if BEST_MODEL:
        accuracy,confusion_mat = predictTestAccuracyBest(input_file,'',DICT_LIST_MODEL,RATINGS_MODEL,VLEN_MODEL,M_MODEL,MODEL_BOOL,output_)
    else:
        accuracy,confusion_mat = predictTestAccuracy(input_file,'',DICT_LIST_MODEL,RATINGS_MODEL,VLEN_MODEL,M_MODEL,MODEL_BOOL,output_)

    input_file.close()
        

    
    
