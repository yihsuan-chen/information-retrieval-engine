#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import re
import copy
import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter     # count th frequency of the items in a list
from porter_stemming import PorterStemmer


# function for tokenization
# The regular expression rules refered from O'Reilly's Regex Cookbook
# https://www.safaribooksonline.com/library/view/regular-expressions-cookbook/9781449327453/
def tokeniser(doc):
    doc1 = doc.replace('\n', ' ')
    rule = r'''(?x)
        "[A-Z\s]+?"                                               #double quatation (only for uppercase characters)
        | \s\‘[\w\s,]+?\’\s                                       #inverted commas
        | \s\'[\w\s,]+?\'\s                                       #single quotation mark
        | [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+          #email
        | (?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})                  #IP address
        | http[s]?://(?:[\w+&@#/%-=~_|$,]+)                       #url
        | \d+(?:\.\d+)?%                                          #percentages, e.g. 82%
        | \$\d+(?:,\d+)?(?:\.\d+)?                                #currency e.g. $12.40
        | [\w]+(?:\.[\w]+)+\.*                                    #abbreviations
        | [A-Z][A-Za-z]+(?:\s[A-Z][a-z]+)*                        #seperated words with capital
        | [a-zA-Z0-9]+(?:-[a-zA-Z]+)+                             #words with optional hyphen
        | [A-Za-z]{2,}                                            #words with at least 2 characters
        | \d+(?:,\d+)?(?:\.\d+)? 
        '''
    token0 = re.findall(rule, doc1)
    token1 = [w.lower().replace(',', '').replace('-', ' ').replace('"', '').replace("'", '').replace("’", '').replace("‘", '').strip() for w in token0]
    token2 = [w for w in token1 if len(w) > 1]
    return token2


# function for remove stop-words
def rm_stopword(token, stopwordfile):
    # read the given stopword file line by line
    with open(stopwordfile, 'r', encoding='utf8') as infile:
        stopword_list = infile.readlines()
    stopword_set = set([x.strip('\n') for x in stopword_list])
    # remove the token in the stop word set
    token_nostopword = [w for w in token if w not in stopword_set]
    return token_nostopword


# Stemming using PorterStemmer
# The code is modified from: https://tartarus.org/martin/PorterStemmer/python.txt
def Stemming(doc_list):
    # call PorterStemmer class
    p = PorterStemmer(doc_list)
    output = []
    i = 0
    for word in p.doc:
        if word.isalpha():
            output.append(p.stem(word, 0, len(word) - 1))
        else:
            output.append(word)
        i += 1
    return output


# main function for this engine
def main():

    # Input argument in the terminal
    task = sys.argv[1]
    arg1 = sys.argv[2]
    arg2 = sys.argv[3]
    arg3 = sys.argv[4:]

    # Task A: document indexing
    if task == 'index':
        print("===START TO INDEX===")
        # read the documents as a dictionary
        collect_doc = {}
        for doc in os.listdir(arg1):
            fullpath = os.path.join(arg1, doc)
            # read the file
            with open(fullpath, 'r', encoding='utf8') as infile:
                # remove ',' and '.txt' of filename and set as the key for the document
                collect_doc[doc.replace(',', '').replace('.txt', '')] = infile.read()

        # text-preprocessing steps
        doc_tokenized = {}
        for key in collect_doc.keys():
            step1 = tokeniser(collect_doc[key])
            step2 = rm_stopword(step1, arg3[0])
            step3 = Stemming(step2)
            step4 = rm_stopword(step3, arg3[0])
            doc_tokenized[key] = step4

        # count document freq (Counter funtion return a dictionary for the frequency of the items in the list)
        word_set = list(chain.from_iterable([set(value) for value in doc_tokenized.values()]))
        doc_freq = Counter(word_set)

        # calculate tf
        collect_tf = {}
        for doc in doc_tokenized.keys():
            collect_tf[doc] = Counter(doc_tokenized[doc])

        # final indexing format
        tfidf_output = []
        N = len(collect_doc)
        for w in doc_freq.keys():
            text = w
            idf = np.log(N / (1 + doc_freq[w]))  # idf calculation
            for doc in collect_tf.keys():
                if w in collect_tf[doc].keys():
                    text += (',' + doc + ',' + str(collect_tf[doc][w]))
            text += (',' + str(round(idf, 3)))
            tfidf_output.append(text)

        # save indexing output
        index_path = os.path.join(arg2, "./index.txt")
        # if the index_dir does not exist, create a new directory
        if not os.path.isdir(arg2):
            os.makedirs(arg2)
        out_file = open(index_path, 'w', encoding='utf8')
        out_file.write('\n'.join(tfidf_output))
        out_file.close()
        print("Indexing is done!")

    # Task B: Query Processing (Vector space model)
    elif task == 'search':
        print('===START TO SEARCH===')
        # loading index file
        index_path = os.path.join(arg1, "./index.txt")
        with open(index_path, 'r', encoding='utf8') as infile:
            tfidf_output1 = infile.readlines()

        # normalize the query
        query = ' '.join(arg3)
        step1 = tokeniser(query)
        step3 = Stemming(step1)
        q_final = step3

        #
        match_doc = {}
        q_vector = []
        target_word = []
        info = []
        for w in tfidf_output1:
            doc_info = w.split(',')
            term = doc_info[0]
            idf = float(doc_info[-1])
            if term in q_final:
                j = 0
                for q in q_final:
                    if term == q:
                        j += 1
                q_vector.append(idf * j)
                target_word.append(term)
                for k in range(1, len(doc_info) - 1, 2):
                    match_doc[doc_info[k]] = []
                info.append(w)

        # make another dictionary for saving the document total weight
        doc_total_w = copy.deepcopy(match_doc)

        if len(target_word) == 0:
            print("There're no matched collections!")
        else:
            print('Normlized Query: ', target_word)

            # map useful weights and calculate tf-idf for match_doc
            for t in info:
                doc_info = t.split(',')
                idf = float(doc_info[-1])
                for d in match_doc.keys():
                    found = False
                    for j in range(1, len(doc_info) - 1, 2):
                        if doc_info[j] == d:
                            match_doc[d].append(float(doc_info[j + 1]) * idf)
                            found = True
                    if not found:
                        match_doc[d].append(0)

            # count document total weight for match_doc
            for w in tfidf_output1:
                doc_info = w.split(',')
                idf = float(doc_info[-1])
                for j in range(1, len(doc_info) - 1, 2):
                    d = doc_info[j]
                    if d in doc_total_w.keys():
                        doc_total_w[d].append(float(doc_info[j + 1]) * idf)

            # calculate cosine similarity
            q_len = np.sqrt(np.sum(np.array(q_vector) * np.array(q_vector)))
            cosine_sim = {}
            for d in match_doc.keys():
                w_len = np.sqrt(np.sum(np.array(doc_total_w[d]) * np.array(doc_total_w[d])))
                dot = np.dot(np.array(match_doc[d]), np.array(q_vector))
                cosine_sim[d] = dot / (q_len * w_len)

            # make a dataframe and sort the results by scores for final output
            df_cosine = pd.DataFrame(list(cosine_sim.items()), columns=['doc', 'score'])
            df_final = df_cosine.sort_values(by=['score'], ascending=False).reset_index(drop=True)

            # finalize the results for output (only show top-k result)
            top_k = int(arg2)
            print('Below are the top-'+str(arg2), 'results:')
            out_file = open("./query_output.txt", 'w', encoding='utf8')
            for index, row in df_final.iterrows():
                result = row["doc"] + ',' + str(round(row["score"], 3))
                if index < top_k:
                    out_file.write(result + '\n')
                    print(result)
            out_file.close()

    # Task C: Query Processing (Search and ranked based on Probabilistic Model)
    elif task == 'search_p':
        print('===START TO SEARCH (Probabilistic Model)===')
        # loading index file
        index_path = os.path.join(arg1, "./index.txt")
        with open(index_path, 'r', encoding='utf8') as infile:
            tfidf_output1 = infile.readlines()

        # query processing
        query = ' '.join(arg3)
        step1 = tokeniser(query)
        step3 = Stemming(step1)
        q_final = step3
        N = len(tfidf_output1)
        query_set = set(q_final)
        match_doc = {}
        target_word = []
        info = []
        q_weight = []
        for w in tfidf_output1:
            doc_info = w.split(',')
            term = doc_info[0]
            if term in query_set:
                target_word.append(term)
                n = 0
                for k in range(1, len(doc_info) - 1, 2):
                    match_doc[doc_info[k]] = []
                    n += 1
                info.append(w)
                q_weight.append(np.log((N + 0.5) / (n + 0.5)))

        if len(target_word) == 0:
            print("There're no matched collections!")
        else:
            print('Normlized Query: ', target_word)

            # map useful weights for match_doc
            k = 0
            for t in info:
                doc_info = t.split(',')
                for d in match_doc.keys():
                    found = False
                    for j in range(1, len(doc_info) - 1, 2):
                        if doc_info[j] == d:
                            match_doc[d].append(q_weight[k])
                            found = True
                    if not found:
                        match_doc[d].append(0)
                k += 1

            # print(target_word, '\n', q_weight)

            # sum scores
            final_score = {}
            for y in match_doc.keys():
                final_score[y] = sum(match_doc[y])

            # finalize the results for output
            top_k = int(arg2)
            df_p = pd.DataFrame(list(final_score.items()), columns=['doc', 'score'])
            df_p2 = df_p.sort_values(by=['score'], ascending=False).reset_index(drop=True)
            print('Below are the top-' + str(arg2), 'results:')
            out_file = open("./search_prop_model.txt", 'w', encoding='utf8')
            for index, row in df_p2.iterrows():
                result = row["doc"] + ',' + str(round(row["score"], 3))
                if index < top_k:
                    out_file.write(result + '\n')
                    print(result)
            out_file.close()

    else:
        print("The specified task is invalid!! Please use 'index' or 'search' or 'search_p'")


if __name__ == "__main__":
    main()
