import nltk
from nltk.corpus import conll2000
from sklearn.svm import LinearSVC
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import re

# Natural Language Toolkit: code_classifier_chunker

class ConsecutiveNPChunkTagger(nltk.TaggerI): # [_consec-chunk-tagger]

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) # [_consec-use-fe]
                train_set.append( (featureset, tag) )
                history.append(tag)
        #self.classifier = nltk.MaxentClassifier.train( # [_consec-use-maxent]
        #    train_set, algorithm='megam', trace=0)
        #chk =[k for k in train_set if None in k[0].values() ]
        #print(chk[1])
        self.classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
        #self.ht=train_set

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): # [_consec-chunker]
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


def npchunk_features(sentence, i, history):
    lmtzr = WordNetLemmatizer()
    wordr, pos = sentence[i]
    word=solve_for_name(wordr, pos)
    #lem=lmtzr.lemmatize(wordr)
    subtree= [y for x,y in sentence[0:i]]
    subtree2= [y for x,y in sentence[i:]]
    if i == 0:
        prevpos2,prevword2 = "<none>", "<none>"
        prevword, prevpos = "<START>", "<START>"
    elif i ==1:
        prevpos2,prevword2 = "<none>", "<none>"
        prevword, prevpos = sentence[i-1]
    else:
        prevword, prevpos = sentence[i-1]
        prevword2, prevpos2 = sentence[i-2]
    if i == len(sentence)-2:
        nextword2, nextpos2 = "<none>", "<none>"
        nextword, nextpos = sentence[i+1]
    elif i == len(sentence)-1:
        nextword, nextpos = "<none>", "<none>"
        nextword2, nextpos2 = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
        nextword2, nextpos2 = sentence[i+2]
    if sentence[i-1] !=None and sentence[i-1]=="DT":
        prr=sentence[i-1][0]
    else:
        prr="n"
    if tags_since_dt(sentence, i) is None:
        tagsin='No-t'
    else:
        tagsin=tags_since_dt(sentence, i)
    return { "lemm": "%s+%s+%s" % (lmtzr.lemmatize(prevword),lmtzr.lemmatize(wordr),lmtzr.lemmatize(nextword)),
             "lem21": "%s+%s" % (lmtzr.lemmatize(prevword2),lmtzr.lemmatize(prevword)),
             "lem10": "%s+%s" % (lmtzr.lemmatize(prevword),lmtzr.lemmatize(wordr)),
             "lem01": "%s+%s" % (lmtzr.lemmatize(wordr),lmtzr.lemmatize(nextword)),
             "shape": get_shape(wordr),
             "pos21": "%s+%s" % (prevpos2,pos),
             "pos10": "%s+%s" % (prevpos,pos),
             "pos01": "%s+%s" % (pos,nextpos),
             "pos210": "%s+%s+%s" % (prevpos2,prevpos,pos),
             "word.lower": wordr.lower(),
             "suffix3": wordr.lower()[-3:],
             "suffix2": wordr.lower()[-2:],
             "suffix1": wordr.lower()[-1:], 
             "prevpospos+nextpos": "%s+%s+%s" % (prevpos, pos, nextpos), 
             "pos+nextpos+nextpos2": "%s+%s+%s" % (pos, nextpos,nextpos2),
             "1grFnP": "%s+%s+%s" % (prevword, wordr, nextword), 
             "2grpast": "%s+%s+%s" % (prevword2,prevword, wordr),   
             "2grfuture": "%s+%s+%s" % (wordr, nextword,nextword2),
             "tags-since-dt": tagsin,
             "subtree": "".join(subtree)
             }


def solve_for_name(word,tag):

  if (len( wn.synsets(word))>1):
    return "".join( sorted([ w.pos()+w.name()  for w in wn.synsets(word)]))
  else:
    return tag

def get_shape(word):
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        shape = 'number'
    elif re.match('\W+$', word):
        shape = 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        shape = 'upcase'
    elif re.match('[a-z]+$', word):
        shape = 'downcase'
    elif re.match('\w+$', word):
        shape = 'mixedcase'
    else:
        shape = 'other'
    return shape




def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
        return '+'.join(sorted(tags))


test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])


#train and saving the classifier
nltk.config_megam('megam.exe')
chunker = ConsecutiveNPChunker(train_sents)

from pickle import dump
output = open("chunkModelSVMFeat1.pkl", "wb")
dump(chunker, output,-1)
output.close()
results =chunker.evaluate(test_sents)

print()
#print(results.incorrect())
print("-----------------------------")
#print(results.missed())
print("----------------------------")
#print(chunker.tagger.classifier.show_most_informative_features(n=20,show="all"))
#print(chunker.classifier.explain(featureset,columns=4))
#print (chunker.tagger.classifier.explain(chunker.tagger.ht[1][0], columns=4))

print(results)


# from sklearn import cross_validation

# cv = cross_validation.KFold(len(train_sents), n_folds=10, shuffle=False, random_state=None)

# for traincv, testcv in cv:
#     chunker = ConsecutiveNPChunker(train_sents[traincv[0]:traincv[len(traincv)-1]])
#     #classifier = nltk.NaiveBayesClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
#     print ('accuracy:', chunker.evaluate( train_sents[testcv[0]:testcv[len(testcv)-1]]))




#print(chunker.evaluate(test_sents))
#print(chunker.show_most_informative)
