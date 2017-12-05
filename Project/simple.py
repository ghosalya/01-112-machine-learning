'''
Part 2
'''
from datetime import datetime
from utility import read_labeled_file, estimate_emission_param, supress_infrequent_words

def single_sentiment_analysis(tags, param, word):
    '''
    Takes:
        - a list of of discovered tags
        - a dictionary for emission parameter
        - the word to be tagged
    return:
        - a tuple of (word, predicted_tag)
    '''
    mle = (word, "O") #assuming tag O for undiscovered word
    mle_value = 0
    for t in tags:
        if (word, t) in param:
            if param[(word, t)] > mle_value:
                mle = (word, t)
                mle_value = param[(word, t)]
    return mle


def write_simple_prediction(country, part, words, tags, param):
    '''
    takes:
        - countri string ("CN","EN" etc)
        - part string (for question part 1, part 2 etc)
        - a list of discovered words
        - a list of discovered tags
        - a dictionary of emission parameter
    '''
    input_filename = country + "/dev.in"
    output_filename = country + "/dev.p"+part+".out"
    with open(input_filename, "r") as inputfile:
        with open(output_filename, "w") as outputfile:
            for line in inputfile:
                if line =="\n":
                    outputfile.write("\n")
                    continue
                if line.strip("\n") in words:
                    pred = single_sentiment_analysis(tags, param, line.strip("\n"))
                    outputfile.write(" ".join(pred)+"\n")
                else:
                    outputfile.write("#UNK# O\n")

if __name__ == '__main__':
    # now we do it for all 4 countries
    # recording timing
    for c in ["CN", "EN", "SG", "FR"]:
        start = datetime.now()
        data = read_labeled_file(c+"/train")
        sdata = supress_infrequent_words(data)
        words, tags, em_param = estimate_emission_param(data)
        write_simple_prediction(c, "2", words, tags, em_param)
        end = datetime.now()
        delt = end - start
        print("{} part 2 done in {}.{}s"\
              .format(c, delt.seconds, delt.microseconds))
