'''
Part 5
'''
from datetime import datetime
from utility import read_labeled_file, estimate_emission_param, supress_infrequent_words
from hmm import write_hmm_prediction, estimate_transition_parameter, predict_tag_sequence
from maxmarginal import predict_tag_sequence_maxmarginal

'''
Using dual-tagged HMM

There is some rule to the tagging that is not fully
covered by the HMM alone, specifically sentiment
vs identity. For example, B-positive cannot be
followed by I-negative. 

Hence, one tag class is sentiment, [positive, negative, neutral, none]
and another  is identity [O, B, I]
before the observation layer
'''

# first of all, we want to generate
# properly-layered file

def split_tag_layer(input_filename, output_filename):
    with open(input_filename, "r") as infile:
        with open(output_filename, "w") as outfile:
            for inline in infile:
                if inline == "\n":
                    outfile.write(inline)
                    continue
                    
                line = inline.strip("\n").split(" ")
                if len(line) < 2:
                    pass
                elif "positive" in line[1]:
                    line[1] = line[1][0]
                    line.append("positive")
                elif "negative" in line[1]:
                    line[1] = line[1][0]
                    line.append("negative")
                elif "neutral" in line[1]:
                    line[1] = line[1][0]
                    line.append("neutral")
                else:
                    line.append("none")
                
                string = " ".join(line) + "\n"
                outfile.write(string)

def read_splitlabel_file(filename):
    sm_emparam = []
    sm_tweet = []
    id_emparam = []
    id_tweet = []
    with open(filename, "r") as f:
        for line in f:
            if line == "\n":
                id_emparam.append(id_tweet)
                id_tweet = []
                sm_emparam.append(sm_tweet)
                sm_tweet = []
            else:
                linetags = line.strip("\n").split(" ")
                id_tweet.append(tuple([linetags[0], linetags[1]]))
                sm_tweet.append(tuple([linetags[0], linetags[2]]))
    return sm_emparam, id_emparam


def predict_dualtag_sequence(word_sequence, words, tags, stags,
                            id_emparam, id_transparam,
                            sm_emparam, sm_transparam):
    id_tagsequence = predict_tag_sequence(word_sequence, words, tags,
                                          id_transparam, id_emparam)
    sm_tagsequence = predict_tag_sequence_maxmarginal(word_sequence, words, stags,
                                                      sm_transparam, sm_emparam)
    # we do some cleanup
    # because in the original, only identities have sentiments
    # but the prediction might not be the case
    sm_count = {}
    for sm in sm_tagsequence:
        sm_count[sm] = sm_count.get(sm, 0) + 1
        
    if "none" in sm_count:
        del sm_count["none"]
    if "START" in sm_count:
        del sm_count["START"]
    if "STOP" in sm_count:
        del sm_count["STOP"]
        
    mostcommon = ("neutral", 0)
    for sm in sm_count:
        if sm_count[sm] > mostcommon[1]:
            mostcommon = (sm, sm_count[sm])

    for i in range(1,len(id_tagsequence)):
        if id_tagsequence[i] == "B" and sm_tagsequence[i] == "none":
            if id_tagsequence[i+1] == "I" and sm_tagsequence[i+1] != "none":
                sm_tagsequence[i] = sm_tagsequence[i+1]
            else:
                sm_tagsequence[i] = mostcommon[0]
        elif id_tagsequence[i] == "I" and sm_tagsequence[i] == "none":
            sm_tagsequence[i] = sm_tagsequence[i-1]
            
    return id_tagsequence, sm_tagsequence


def write_dualhmm_prediction(country, part, prediction_function,
                             words, tags, stags,
                             id_emparam, id_transparam,
                             sm_emparam, sm_data):
    input_filename = country + "/dev.in"
    output_filename = country + "/dev.p"+part+".out"
    indata = []
    with open(input_filename, "r") as infile:
        indata = infile.read().strip('\n').split('\n\n') #read and separate tweets
    
    with open(output_filename, "w") as outfile:
        for tweet in indata:
            word_sequence = tweet.split('\n')
            pred_id_tag, pred_sm_tag = prediction_function(word_sequence, words,
                                                            tags, stags,
                                                            id_emparam, id_transparam,
                                                            sm_emparam, sm_data)
            pred_id_tag.remove("START")
            pred_id_tag.remove("STOP")
            pred_sm_tag.remove("START")
            pred_sm_tag.remove("STOP")
            if len(word_sequence) != len(pred_id_tag):
                print("WARNING!! Different length \n{} / \n{}"\
                      .format(word_sequence, pred_id_tag))
            for i in range(len(word_sequence)):
                line = "{} {} {}\n".format(word_sequence[i], pred_id_tag[i], pred_sm_tag[i])
                outfile.write(line)
            outfile.write("\n")


def merge_tag_layer(input_filename, output_filename):
    with open(input_filename, "r") as infile:
        with open(output_filename, "w") as outfile:
            for inline in infile:
                if inline == "\n":
                    outfile.write("\n")
                    continue
                line = inline.strip("\n").split(" ")
                if line[1] == "O":
                    string = " ".join(line[:-1]) + "\n"
                else:
                    string = " ".join(line[:-1]) + "-"+ line[2] + "\n"
                outfile.write(string)


if __name__ == '__main__':
    for c in ["CN", "EN", "SG", "FR"]:
        start = datetime.now()
        split_tag_layer(c+"/train", c+"/trainl")
        sm_data, id_data = read_splitlabel_file(c+"/trainl")
        sm_data = supress_infrequent_words(sm_data)
        id_data = supress_infrequent_words(id_data)
        
        words, tags, id_emparam = estimate_emission_param(id_data)
        id_transparam = estimate_transition_parameter(id_data)
        swords, stags, sm_emparam = estimate_emission_param(sm_data)
        sm_transparam = estimate_transition_parameter(sm_data)
        
        write_dualhmm_prediction(c,"5l", predict_dualtag_sequence,
                                 words, tags, stags,
                                 id_emparam, id_transparam,
                                 sm_emparam, sm_transparam)
        merge_tag_layer(c+"/dev.p5l.out", c+"/dev.p5.out")
        
        end = datetime.now()
        delt = end - start
        print("{} part 5 done in {}.{}s"\
              .format(c, delt.seconds, delt.microseconds))