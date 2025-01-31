# WEEK 2 Task: Search engine
# 2nd version, issue #9 almost fixed

import nltk
from sklearn.feature_extraction.text import CountVectorizer

# DATA
wikipedia = ""
file_name = "enwiki-20181001-corpus.1000-articles.txt"
try:
    file = open(file_name, "r", encoding='utf-8')
    wikipedia = file.read()
    file.close()
except OSError:
    print("Error reading the file", file_name)

# Split into lists of strings (each article is a string)
documents = wikipedia.split('</article>')

# Create a dictionary (article name: article contents) if needed
dict = {}
for d in documents:
    alku = d.find("<article name=")
    loppu = d.find('">')
    name = d[16:loppu]
    text = d[loppu + 2:]
    dict[name] = text

def read_article(title):
    if title in dict.keys():
        print("\n*****************************************")
        print("\n" + title)
        print(dict.get(title))
        print("*****************************************\n")
    else:
        print("No such article found.")

# CREATING THE MATRIX
# use scikit's count vectorizer to convert documents to a matrix of tokens
# binary=true --> all non zero counts are set to 1
# -->(doesn't measure token counts but whether the term is in the document or not)

# standard vectorizer: only words with 2 or more characters
# term vocabulary size: 107 001
#cv = CountVectorizer(lowercase=True, binary=True)

# Issue 10: index all words containing alpha-numerical characters (including stop words)
# this vectorizer includes also 1-character words (like 'a') and doesn't remove any stopwords
# terms vocabulary size: 107 343
stopwords = []  # add here any stopwords you want to remove from the list of tokens
cv = CountVectorizer(lowercase=True, stop_words=stopwords, token_pattern='(?u)\\b\\w+\\b',
                     max_df=1.0, min_df=1, binary=True)

# print a sparse matrix
# cv learns the vocabulary dictionary and returns a document-term matrix
sparse_matrix = cv.fit_transform(documents)

# dense version of the same matrix:
dense_matrix = sparse_matrix.todense()

# transpose into term-document matrix:
# rows=terms, columns=documents (ordered by terms for faster lookup)
td_matrix = dense_matrix.T   # .T transposes the matrix
terms = cv.get_feature_names()  # list of all terms

# term-to-index vocabulary {"term" : <index>}
t2i = cv.vocabulary_

# SIMPLE QUERY PARSER

# operator replacements
operators = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

# rewrites the operators (and, or, not), or
# if the token is not an operator, returns the row for the term in term-document matrix:
def rewrite_token(t):
    if t in operators.keys():
        newtoken = operators.get(t)
    elif t in terms:
        newtoken = 'td_matrix[t2i["{:s}"]]'.format(t)
    else:
        newtoken = "NOTFOUND"

    #return operators.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))
    return newtoken

def unknown_words(tokens):
    idx = tokens.index("NOTFOUND")

    if len(tokens) == 1:  # only 1 word in query (NOTFOUND)
        return -1 # no matches

    # when query includes "NOT NOTFOUND"
    if idx > 0:  # NOTFOUND is not first token
        if tokens[idx - 1] == "1 -":
            if len(tokens) == 2:  # only NOT NOTFOUND
                return -2 # matches everything

            if len(tokens) > 3:  # there are other words (not notfound and/or ...)
                # remove "NOT NOTFOUND" and  | or & sign from either side to run the rest of the query
                if idx == 1:
                    tokens = tokens[idx + 2:]
                    if len(tokens) == 1:
                        return tokens[0]
                    else:
                        return " ".join(tokens)

                if idx > 2:
                    tokens = tokens[:idx - 2]
                    if len(tokens) == 1:
                        return tokens[0]
                    else:
                        return " ".join(tokens)

    # NOTFOUND and/or other word
    if len(tokens) > 2:
        if idx < (len(tokens) - 1):
            if tokens[idx + 1] == "&":  # NOTFOUND and ...
                return -1 # no matches
            if tokens[idx + 1] == "|":  # NOTFOUND or ...
                tokens = tokens[idx + 2:]  # remove "NOTFOUND |" and make the other part of the query
                if len(tokens) == 1:
                    return tokens[0]
                else:
                    return " ".join(tokens)

        if idx > 1:
            if tokens[idx - 1] == "&":  # ... & NOTFOUND
                return -1 # no matches
            if tokens[idx - 1] == "|":  # ... | NOTFOUND
                tokens = tokens[:idx - 1]
                if len(tokens) == 1:
                    return tokens[0]
                else:
                    return " ".join(tokens)

# splits query into tokens, rewrites all tokens and joins them together:
def rewrite_query(query):
    # query in a list form:
    tokens = [rewrite_token(t) for t in query.split()]

    if "NOTFOUND" in tokens:
        tokens = unknown_words(tokens)

        if tokens == -1:
            return -1

        if tokens == -2:
            return -2

        else:
            print("One of the words was not found.")
            return tokens

    else: # all terms exist, query is fine
        return " ".join(tokens)

def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query))) # Eval runs the string as a Python command
    print()

def show_doc(query):
    # matching documents as a matrix of one row:
    q = rewrite_query(query)

    if q == -1:
        print("Word not found. No matches.")

    elif q == -2:
        print("Word not found. Search query matches every article.")
        count = 0
        for d in documents:
            print("<Matching article:", d[15:200] + "...")
            print()
            count += 1
            if count > 4:
                print("Showing the first five of", len(documents), "articles.")
                return

    else:
        hits_matrix = eval(rewrite_query(query))  # runs the query

        # the y-coordinates (doc indexes) of the non-zero elements converted to a list:
        hits_list = list(hits_matrix.nonzero()[1])

        #   print(hits_list)
        count = 0
        for doc_idx in hits_list[:5]:
            print("<Matching article:", documents[doc_idx][15:200] + "...")
            print()
            count += 1
            if count > 4:
                print("Showing the first five of", len(hits_list), "articles. \n")

        answer = str(input("Would you like to read one of the articles? (y/n) "))
        if answer == "y":
            title = str(input("Type the title of the article (with correct capitalization): "))
            read_article(title)


# MAKING QUERIES
print("Welcome to the search engine!")
print("Data: 1000 Wikipedia articles")
print("You can use boolean operators (and, or, not) in your query.")
# Ask the user to type a query
query = str(input("Enter a query: "))
while query != "":
    # hits_matrix = eval(rewrite_query(query))
    try:
        #test_query(query)
        show_doc(query)
        query = str(input("Enter a new query (enter blank if you want to quit): "))
    except KeyError:
        query = str(input("Your search failed. Try again: "))
else:
    print("Goodbye!")

