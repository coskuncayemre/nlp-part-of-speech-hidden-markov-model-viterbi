import re

"""
    Emre COSKUNCAY
    21526806
    BBM497-Assignment 2    

"""

file = open("metu.txt", "r", encoding='utf-8')

trainArray = []
testArray = []
resultArray = []

initialDict = {}
transitionDict = {}
transitionProb = {}
emissionDict = {}

unknownWords = 0
totalInitial = 0
totalWord = 0
totalTag = 0
totalUniqueBigram = 0
totalSuccess = 0

#reads text , creates train and test array
def readText():
    global totalInitial
    i = 1
    for line in file.readlines():
        if(i< 3961): #3961
            trainArray.append(languageModel(line,1))
            totalInitial += 1
        else:
            testArray.append(languageModel(line,1))
        i = i + 1
    return i

#count all words' frequencies not number of unique words
def countDict(dict):
    total = 0
    for number in dict.values():
        total = total + number
    return  total

#s is sentence,n is wanted language model such as unigram,bigram
def languageModel(sentence, n):
    sentence = sentence.lower()
    sentence = re.sub(r'[\s]', ' ', sentence)
    tokens = [token for token in sentence.split(" ") if token != ""]
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

readText()

#calculates frequency and probability of first word of the sentence
def initialCount():
    global totalx
    #calculates frequency
    for i in trainArray:
        firstPair = i[0].split("/")
        initialTag = firstPair[1]
        if not initialTag in initialDict.keys():
            initialDict[initialTag] = 1
        else:
            initialDict.update({initialTag : initialDict.get(initialTag) + 1})
    #then calculates probability
    for tag in initialDict:
        initialDict.update({tag : initialDict[tag] / totalInitial})

initialCount()

#calculates probability of transition between two tags.
def createBigram():
    for sentences in trainArray:
        newSent = ""
        for word in sentences:
            pair = word.split("/")
            tag = pair[1]
            newSent = newSent + " " + tag
        tempList = languageModel(newSent, 2)
        for tagPair in tempList:
            global totalUniqueBigram
            totalUniqueBigram += 1
            if not tagPair in transitionDict.keys():
                transitionDict[tagPair] = 1
            else:
                transitionDict.update({tagPair: transitionDict.get(tagPair) + 1})

createBigram()

#creates nested dictionary for calculating emission frequency
def getTag():
    for sentence in trainArray:
        for wordTag in sentence:
            global  totalWord
            pair = wordTag.split("/")
            word = pair[0]
            tag = pair[1]
            totalWord = totalWord + 1
            #checks if tag is not in the dict.
            if not tag in emissionDict.keys():
                emissionDict[tag] = {}
                emissionDict[tag].update({word: 1})
            else:
                tempDict = emissionDict.get(tag)
                #checks the word had been seen before or not
                if word in tempDict.keys():
                    tempDict.update({word: tempDict.get(word) + 1})
                else:
                    tempDict[word] = 1


getTag()

# frequency of transition between two tags changes to probability
def calculateTransitionProbability():
    counterDict= {}
    #calculates frequency
    for bigramPair in transitionDict:
        pairWords = bigramPair.split(" ")
        firstWord = pairWords[0]
        value = transitionDict[bigramPair]
        if not firstWord in counterDict.keys():
            counterDict[firstWord] = value
        else:
            counterDict.update({ firstWord : counterDict[firstWord] + value})
    #calculates probability with divide by total number of first tag
    for bigramPair in transitionDict:
        tagPair = bigramPair.split(" ")
        firstTag = tagPair[0]
        transitionDict.update({bigramPair: transitionDict[bigramPair] / counterDict.get(firstTag)})

# frequency of emission pairs change to probability
def calculateEmissionProbability():
    for tag in emissionDict:
        tag_own_word_counter = 0
        for word in emissionDict[tag]:
            tag_own_word_counter += emissionDict[tag][word]
        for word in emissionDict[tag]:
            emissionDict[tag].update({ word : emissionDict[tag][word] / tag_own_word_counter})
        emissionDict[tag].update({"totalNumber": float(tag_own_word_counter)})

calculateEmissionProbability()
calculateTransitionProbability()

def ownViterbi(obs,states,initProb,transProb,emit_p):
    global unknownWords
    path = [{}]
    result= [{}]
    pathArray =[{}]
    #calculates probability of the first word of the sentence
    for tag in states:
        value = emit_p.get(tag).get(obs[0])
        # good turing smoothing
        if value is None:
            unknownWords += 1
            value = len(emit_p.get(tag)) / (unknownWords * emit_p.get(tag).get("totalNumber"))
        initTag = initProb.get(tag,0)
        firstProb = initTag * value
        path[0][tag] = firstProb
        result[0][tag] = firstProb
    pathArray[0][obs[0]] = path[0]

    for i in range(1,len(obs)):
        path.append({})
        pathArray.append({})
        result.append({})
        for y in states:
            temp = {}
            value = emit_p.get(y).get(obs[i])
            # good turing smoothing
            if value is None:
                unknownWords += 1
                value = len(emit_p.get(y)) / (unknownWords * emit_p.get(y).get("totalNumber"))
            for y0 in states:
                lasCall = result[i - 1][y0] * transProb.get(y0).get(y, 0) * value
                coklu = y0 + "-" + y
                temp.update({coklu: lasCall})
            maxPair = max(temp, key=temp.get)
            result[i][y] = temp[maxPair]
            path[i][maxPair] = temp[maxPair]
            pathArray[i][obs[i]] = path[i]

    #backtrace
    pathArray.reverse()
    sonuc=[]
    sonucTag=[]
    for i in range(len(pathArray)):
        for word in pathArray[i].values():
            if i == 0:
                enBuyuk = max(word, key=word.get)
                key = list(pathArray[i])[0]
                tagger = enBuyuk.split("-")
                sonuc.append(key + "/" + tagger[1].capitalize())
                sonucTag.append(tagger[1])
                nextWord = tagger[0]
            else:
                for x in word.keys():
                    tagger = x.split("-")
                    if len(tagger) > 1:
                        if tagger[1] == nextWord:
                            key = list(pathArray[i])[0]
                            sonuc.append(key + "/" + nextWord.capitalize())
                            sonucTag.append(nextWord)
                            nextWord = tagger[0]
                            break
                    else:
                        key = list(pathArray[i])[0]
                        sonuc.append(key.capitalize() + "/" + nextWord.capitalize())
                        sonucTag.append(nextWord)
                        break
    sonuc.reverse()
    sonucTag.reverse()

    return sonucTag,(" ".join(sonuc))

# creates nested dictionary of transition probability to use in Viterbi.
def createNestedTransition():
    for bigramPair in transitionDict:
        tagPair = bigramPair.split(" ")
        firstTag = tagPair[0]
        secondTag = tagPair[1]
        value = transitionDict[bigramPair]
        if not firstTag in transitionProb.keys():
            transitionProb[firstTag] = {}
            transitionProb[firstTag].update({secondTag: value})
        else:
            tempDict = transitionProb.get(firstTag)
            tempDict.update({ secondTag : value})


createNestedTransition()

#calculates accurancy of each sentence that was predicted
def accurancy(realTags,predictTags):
    global totalSuccess
    global totalTag
    success = 0
    for i in range(len(realTags)):
        if realTags[i] == predictTags[i]:
            success += 1
        totalTag +=1
    totalSuccess += success

    #print(realTags)
    #print(predictTags)



# splits each test sentence and adds these to some arrays and tuples and calls viterbi function.
def startToPredict():
    global resultArray
    realTagArray = []
    predictionArray = []
    for sentence in testArray:
        states = []
        observations = []
        tempRealArray = []
        for i in range(len(sentence)):
            wordTagPair = sentence[i].split("/")
            word = wordTagPair[0]
            tag = wordTagPair[1]
            tempRealArray.append(tag)
            states.append(tag)
            observations.append(word)
        states = tuple(states)
        observations=tuple(observations)
        predict,predictSentence = ownViterbi(observations,
                       states,
                       initialDict,
                       transitionProb,
                       emissionDict)
        accurancy(tempRealArray, predict)
        predictionArray.append(predict)
        realTagArray.append(tempRealArray)
        resultArray.append(predictSentence)
    #print(totalSuccess)
    #print(totalTag)
    print("Total Accurancy =  %" + str((totalSuccess/totalTag)*100))

startToPredict()

with open("output.txt", "w") as text_file:
    for i in resultArray:
        print("{}".format(i), file=text_file)