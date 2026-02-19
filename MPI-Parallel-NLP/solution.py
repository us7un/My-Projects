############### AUTHOR NAMES ################
# Student 1: Üstün Yılmaz
# Student 2: Enes Hamza Üstün
#######################################################


from mpi4py import MPI
import string
import argparse


# Tokenizes sentences (lines) from the text file
def tokenizeSentences(textLocation):
    with open(textLocation, 'r') as textFile:
        paragraph = textFile.read() # read the text file
        sentenceList = [sentence.strip() for sentence in paragraph.split('\n')] # split into sentences (lines)
        return sentenceList # return list of sentences
    
# Tokenizes vocabulary words from the vocab file
def tokenizeVocab(vocabLocation):
    with open(vocabLocation, 'r') as vocabFile:
        vocabList = [line.strip().lower() for line in vocabFile.readlines()] # read vocab file line by line
        return vocabList # return list of vocab words
    
# Tokenizes stopwords from the stopwords file
def tokenizeStopWords(stopWordsLocation):
    with open(stopWordsLocation, 'r') as stopWordsFile:
        stopWordsList = [line.strip().lower() for line in stopWordsFile.readlines()] # read stop words file line by line
        return stopWordsList # return list of stop words
    
# Makes a given sentence (string) lowercase
def makeLowercase(sentence):
    updatedSentence = sentence.lower() # make sentence lowercase
    return updatedSentence # return updated sentence

# Removes punctuation from a given sentence (string)
def removePunctuation(sentence):
    translator = str.maketrans('', '', string.punctuation) # translator to remove punctuation
    updatedSentence = sentence.translate(translator) # remove punctuation
    return updatedSentence # return updated sentence

# Filters out stop words from a given sentence (string)
def filterWords(sentence, stopWordsList):
    sentenceWords = [word for word in sentence.split() if word] # tokenize sentence
    filteredWords = [word for word in sentenceWords if word not in stopWordsList] # remove stop words
    return filteredWords # return list of filtered words

# Preprocesses a given sentence (string) by making it lowercase, removing punctuation, and filtering stop words
def preprocessSentence(sentence, stopWordsList):
    updatedSentence = makeLowercase(sentence) # make lowercase
    updatedSentence = removePunctuation(updatedSentence) # remove punctuation
    filteredWords = filterWords(updatedSentence, stopWordsList) # filter stop words
    return filteredWords # return list of preprocessed words

# Counts term frequency for each word in vocabList from sentenceList after preprocessing
def countTermFrequency(sentenceList, vocabList, stopWordsList):
    termFrequencies = {word.lower(): 0 for word in vocabList} # initialize term frequencies
    for sentence in sentenceList:
        preprocessed = preprocessSentence(sentence, stopWordsList) # preprocess sentence
        for word in preprocessed:
            if word in termFrequencies:
                termFrequencies[word] += 1 # increment term frequency
    return termFrequencies # return term frequencies

# Counts document frequency for each word in vocabList from sentenceList after preprocessing
def countDocumentFrequency(sentenceList, vocabList, stopWordsList):
    documentFrequencies = {word.lower(): 0 for word in vocabList} # initialize document frequencies
    for sentence in sentenceList:
        preprocessed = preprocessSentence(sentence, stopWordsList) # preprocess sentence
        uniqueWords = set(preprocessed) # get unique words in the sentence
        for word in uniqueWords:
            if word in documentFrequencies:
                documentFrequencies[word] += 1 # increment document frequency
    return documentFrequencies # return document frequencies

# Counts term frequency for each word in vocabList from sentenceList without preprocessing
def countTermFrequencyW(sentenceList, vocabList):
    termFrequencies = {word.lower(): 0 for word in vocabList} # initialize term frequencies
    for sentence in sentenceList:
        for word in sentence:
            if word in termFrequencies:
                termFrequencies[word] += 1 # increment term frequency
    return termFrequencies # return term frequencies

# Counts document frequency for each word in vocabList from sentenceList without preprocessing
def countDocumentFrequencyW(sentenceList, vocabList):
    documentFrequencies = {word.lower(): 0 for word in vocabList} # initialize document frequencies
    for sentence in sentenceList:
        uniqueWords = set(sentence) # get unique words in the sentence
        for word in uniqueWords:
            if word in documentFrequencies:
                documentFrequencies[word] += 1 # increment document frequency
    return documentFrequencies # return document frequencies

# Chunks sentenceList into numWorkers chunks for distribution
def chunkSentences(sentenceList, numWorkers):
    chunkSize = len(sentenceList) // numWorkers # size of each chunk
    remaining = len(sentenceList) % numWorkers # divide remaining "equally" among workers
    chunks = []
    current = 0 # current sentence index
    for i in range(numWorkers):
        extra = 1 if i < remaining else 0 # give remaining sentences to first r workers
        end = current + chunkSize + extra # end index for chunk
        chunks.append(sentenceList[current:end]) # add chunk to list
        current = end # update current index
    return chunks # return list of chunks

# END-TO-END PATTERN OF PREPROCESSING AND TERM FREQUENCY COUNTING
def endToEnd(sentenceList, vocabList, stopWordsList):
    comm = MPI.COMM_WORLD # initialize MPI communicator
    rank = comm.Get_rank() # get process rank
    size = comm.Get_size() # get number of processes (n-1 workers + 1 master)

    # MASTER PROCESS
    if rank == 0:
        chunks = chunkSentences(sentenceList, size - 1) # chunk sentences
        totalTermFrequencies = {word.lower(): 0 for word in vocabList} # initialize total term frequencies
        for i in range(1, size):
            comm.send(chunks[i - 1], dest=i) # send chunk to worker

        for i in range(1, size):
            workerTermFrequencies = comm.recv(source=i) # receive term frequencies from worker
            for word in totalTermFrequencies:
                totalTermFrequencies[word] += workerTermFrequencies[word] # add worker frequencies to total
        
        print("Term Frequencies:")
        print(totalTermFrequencies) # print total term frequencies
        return totalTermFrequencies # return total term frequencies

    # WORKER PROCESSES
    else:
        receivedChunk = comm.recv(source=0) # receive chunk from master
        workerTermFrequencies = countTermFrequency(receivedChunk, vocabList, stopWordsList) # count term frequencies
        comm.send(workerTermFrequencies, dest=0) # send term frequencies to master
        return None # only master returns result

# LINEAR PIPELINE PATTERN OF PREPROCESSING AND TERM FREQUENCY COUNTING
def linearPipeline(sentenceList, vocabList, stopWordsList):
    comm = MPI.COMM_WORLD  # initialize MPI communicator
    rank = comm.Get_rank()  # get process rank

    # MASTER PROCESS
    if rank == 0:
        chunk_size = 10
        chunks = chunkSentences(sentenceList, chunk_size)
        for i in range(chunk_size):
            comm.send(chunks[i], dest=1)  # send chunk consecutively to worker 1

        comm.send("DONE", dest= 1)  # end signal
        termFrequencies = comm.recv(source=4)  # receive term frequencies from worker 4
        print("Term Frequencies:")
        print(termFrequencies)
        return termFrequencies

    # WORKER PROCESSES
    else:
        if rank == 1: # worker 1
            while True:
                receivedChunk = comm.recv(source=0) # get the chunk
                newChunk = [] #initialize processed chunk
                if receivedChunk == "DONE": # end condition
                    comm.send("DONE", dest=2)  # propagate end signal
                    break
                for sentence in receivedChunk:
                    updated_sentence = makeLowercase(sentence) # make the sentences lowercase
                    newChunk.append(updated_sentence)
                comm.send(newChunk, dest=2) # send the processed chunk to next worker

        elif rank == 2: # worker 2
            while True:
                receivedChunk = comm.recv(source=1) # get the chunk
                if receivedChunk == "DONE": # end condition
                    comm.send("DONE", dest=3)  # propagate end signal
                    break
                newChunk = [] #initialize processed chunk
                for sentence in receivedChunk:
                    updated_sentence = removePunctuation(sentence) # remove the punctuations
                    newChunk.append(updated_sentence)
                comm.send(newChunk, dest=3) # send the processed chunk to next worker

        elif rank == 3: # worker 3
            while True:
                receivedChunk = comm.recv(source=2) # get the chunk
                if receivedChunk == "DONE": # end condition
                    comm.send("DONE", dest=4)  # propagate end signal
                    break
                newChunk = [] #initialize processed chunk
                for sentence in receivedChunk:
                    updated_sentence = filterWords(sentence,stopWordsList) # remove the stop words from sentences
                    newChunk.append(updated_sentence)
                comm.send(newChunk, dest=4) # send the processed chunk to next worker

        elif rank == 4:
            termFrequencies = {word.lower(): 0 for word in vocabList}  # initialize term frequencies
            while True:
                receivedChunk = comm.recv(source=3) # get the chunk
                if receivedChunk == "DONE":
                    comm.send(termFrequencies, dest=0) # send the final result to the manager
                    break
                for sentence in receivedChunk:
                    for word in sentence:
                        if word in termFrequencies:
                            termFrequencies[word] += 1  # count the term frequencies

                

# PARALLEL PIPELINES PATTERN OF PREPROCESSING AND TERM FREQUENCY COUNTING
def parallelPipelines(sentenceList, vocabList, stopWordsList):
    comm = MPI.COMM_WORLD # initialize MPI communicator
    rank = comm.Get_rank() # get process rank
    size = comm.Get_size() # get number of processes (size = 1 + 4*i where i is number of pipelines)

    # MASTER PROCESS
    if rank == 0:
        numPipelines = (size - 1) // 4 # number of pipelines
        chunks = chunkSentences(sentenceList, numPipelines) # chunk sentences for each pipeline
        totalTermFrequencies = {word.lower(): 0 for word in vocabList} # initialize total term frequencies
        for i in range(numPipelines):
            comm.send(chunks[i], dest=1 + i * 4) # send chunk to first worker of each pipeline
        
        for i in range(numPipelines):
            pipelineTermFrequencies = comm.recv(source=4 + i * 4) # receive term frequencies from last worker of each pipeline
            for word in totalTermFrequencies:
                totalTermFrequencies[word] += pipelineTermFrequencies[word] # add pipeline frequencies to total
        
        print("Term Frequencies:")
        print(totalTermFrequencies) # print total term frequencies
        return totalTermFrequencies # return total term frequencies
    
    # WORKER PROCESSES
    else:
        pipelinePosition = (rank - 1) % 4 # position of worker in pipeline (from 0 to 3)

        # Position 0: Receive chunk and make lowercase
        if pipelinePosition == 0:
            receivedChunk = comm.recv(source=0) # get chunk from master
            chunkAgainForPipeline = chunkSentences(receivedChunk, 10) # chunk again for pipeline

            for subchunk in chunkAgainForPipeline:
                lowerCaseSubchunk = [makeLowercase(sentence) for sentence in subchunk] # make lowercase
                comm.send(lowerCaseSubchunk, dest=rank + 1) # send to next worker
            
            comm.send("DONE", dest=rank + 1) # end signal
            return None # only master returns result
        
        # Position 1: Remove punctuation
        elif pipelinePosition == 1:
            while True:
                receivedSubchunk = comm.recv(source=rank - 1) # receive from previous worker
                if receivedSubchunk == "DONE":
                    comm.send("DONE", dest=rank + 1) # propagate end signal
                    break
                punctuationRemovedSubchunk = [removePunctuation(sentence) for sentence in receivedSubchunk] # remove punctuation
                comm.send(punctuationRemovedSubchunk, dest=rank + 1) # send to next worker
            
            return None # only master returns result
        
        # Position 2: Stopword removal
        elif pipelinePosition == 2:
            while True:
                receivedSubchunk = comm.recv(source=rank - 1) # receive from previous worker
                if receivedSubchunk == "DONE":
                    comm.send("DONE", dest=rank + 1) # propagate end signal
                    break
                stopwordRemovedSubchunk = [filterWords(sentence, stopWordsList) for sentence in receivedSubchunk] # remove stopwords
                comm.send(stopwordRemovedSubchunk, dest=rank + 1) # send to next worker

            return None # only master returns result
        
        # Position 3: Term frequency counting
        elif pipelinePosition == 3:
            pipelineTermFrequencies = {word.lower(): 0 for word in vocabList} # initialize pipeline term frequencies
            while True:
                receivedSubchunk = comm.recv(source=rank - 1) # receive from previous worker
                if receivedSubchunk == "DONE":
                    break
                for sentenceWords in receivedSubchunk:
                    for word in sentenceWords:
                        if word in pipelineTermFrequencies:
                            pipelineTermFrequencies[word] += 1 # increment term frequency
            comm.send(pipelineTermFrequencies, dest=0) # send to master
            
            return None # only master returns result

# END-TO-END PARALLEL PATTERN OF PREPROCESSING AND TERM & DOCUMENT FREQUENCY COUNTING
def endToEndParallel(sentenceList, vocabList, stopWordsList):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # MASTER PROCESS
    if rank == 0:
        chunks = chunkSentences(sentenceList, size - 1)  # chunk sentences
        totalTermFrequencies = {word.lower(): 0 for word in vocabList}  # initialize total term frequencies
        totalDocFrequencies = {word.lower(): 0 for word in vocabList}
        for i in range(1, size):
            comm.send(chunks[i - 1], dest=i)  # send chunk to worker

        for i in range(size - 1):
            tag, data = comm.recv(source=MPI.ANY_SOURCE) # receive from any worker
            if tag == "TF":
                for word in totalTermFrequencies:
                    totalTermFrequencies[word] += data[word]  # add worker frequencies to total
            elif tag == "DF":
                for word in totalDocFrequencies:
                    totalDocFrequencies[word] += data[word]  # add worker frequencies to total

        # PRINT AND RETURN RESULTS
        print("Term Frequencies:")
        print(totalTermFrequencies)
        print("Document Frequencies:")
        print(totalDocFrequencies)
        return totalTermFrequencies, totalDocFrequencies
    
    # WORKER PROCESSES
    else:
        receivedChunk = comm.recv(source=0)  # receive chunk from master
        preprocessedChunk = [preprocessSentence(sentence, stopWordsList) for sentence in receivedChunk] # preprocess chunk
        # ASYMMETRIC PAIRING TO AVOID DEADLOCK
        if rank % 2 == 1:
            pairRank = rank + 1
            comm.send(preprocessedChunk, dest=pairRank)  # send to pair
            receivedChunkFromPair = comm.recv(source=pairRank) # receive from pair
        else:
            pairRank = rank - 1
            receivedChunkFromPair = comm.recv(source=pairRank) # receive from pair
            comm.send(preprocessedChunk, dest=pairRank)  # send to pair

        combinedChunk = preprocessedChunk + receivedChunkFromPair

        if rank % 2 == 1:
            termFrequencies = countTermFrequencyW(combinedChunk, vocabList)
            comm.send(( "TF", termFrequencies), dest=0) # send term frequencies
            return None # only master returns result
        else:
            documentFrequencies = countDocumentFrequencyW(combinedChunk, vocabList)
            comm.send(( "DF", documentFrequencies), dest=0) # send document frequencies
            return None # only master returns result


# DRIVER CODE
def main():
    parsedArgs = argparse.ArgumentParser() # parse arguments
    parsedArgs.add_argument('--vocab', type=str, required=True) # vocab file location
    parsedArgs.add_argument('--stopwords', type=str, required=True) # stopwords file location
    parsedArgs.add_argument('--text', type=str, required=True) # text file location
    parsedArgs.add_argument('--pattern', type=int, required=True) # pattern selection
    args = parsedArgs.parse_args() # arguments list

    vocabLocation = args.vocab
    stopWordsLocation = args.stopwords
    textLocation = args.text
    pattern = args.pattern

    sentenceList = tokenizeSentences(textLocation) # tokenize sentences
    vocabList = tokenizeVocab(vocabLocation) # tokenize vocab
    stopWordsList = tokenizeStopWords(stopWordsLocation) # tokenize stop words

    # SELECT CONCURRENCY PATTERN
    if pattern == 1:
        endToEnd(sentenceList, vocabList, stopWordsList)
    elif pattern == 2:
        linearPipeline(sentenceList, vocabList, stopWordsList)
    elif pattern == 3:
        parallelPipelines(sentenceList, vocabList, stopWordsList)
    elif pattern == 4:
        endToEndParallel(sentenceList, vocabList, stopWordsList)
    

if __name__ == "__main__":
    main()