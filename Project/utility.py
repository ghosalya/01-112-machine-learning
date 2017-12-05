#reading file

def read_labeled_file(filename):
    '''
    Read an apropriate file.
    
    Takes the path to file
    returns a list of (word, tag) tuples
    '''
    result = []
    singletweet = []
    with open(filename, "r") as f:
        for line in f:
            if line == "\n":
                result.append(singletweet)
                singletweet = []
            else:
                linelist = line.strip("\n").split(" ")
                singletweet.append(tuple(linelist))
    return result

def supress_infrequent_words(data, k=3):
    '''
    Takes a list of (word, tag) tuple
    returns a new list with infrequent
    words replaced with #UNK#
    
    k = number of occurence that is 
        considered to be known
    '''
    word_count = {}

    #get word count
    for tweet in data:
        for tagged_word in tweet:
            word = tagged_word[0]
            word_count[word] = word_count.get(word, 0) + 1
                
    #generate new list
    result = []
    newtweet = []
    for tweet in data:
        for tagged_word in tweet:
            word = tagged_word[0]
            if word_count[word] >= k:
                newtweet.append(tagged_word)
            else:
                tag = tagged_word[1]
                newtweet.append(("#UNK#",tag))
        result.append(newtweet)
        newtweet = []
        
    return result

def estimate_emission_param(data):
    '''
    Takes a list of (word, tag) tuple.
    returns:
        - iterable of all available words
        - iterable of all available label/tag
        - dictionary of emission parameter 
          with key <word, tag>
    '''
    tag_to_word_count = {}
    word_count = {}
    tag_count = {}
    
    for tweet in data: 
        for tagged_word in tweet:
            # loops through the data and get respective counts
            word = tagged_word[0]
            tag = tagged_word[1]
            
            #incrementing counts
            word_count[word] = word_count.get(word, 0) + 1
            tag_count[tag] = tag_count.get(tag, 0) + 1
            tag_to_word_count[tagged_word] = tag_to_word_count\
                                              .get(tagged_word, 0) + 1
                
    # once count is settled, we can get emission parameter
    emission_parameter = {k: tag_to_word_count[k]/tag_count[k[1]] 
                          for k in tag_to_word_count}
    
    return word_count.keys(), tag_count.keys(), emission_parameter

