'''
Part 5 test-writing
'''
from datetime import datetime
from utility import read_labeled_file, estimate_emission_param, supress_infrequent_words
from hmm import write_hmm_prediction, estimate_transition_parameter, predict_tag_sequence
from maxmarginal import predict_tag_sequence_maxmarginal
from dual_hmm import split_tag_layer, read_splitlabel_file, merge_tag_layer, predict_dualtag_sequence

def write_dualhmm_ontest(country, prediction_function,
                             words, tags, stags,
                             id_emparam, id_transparam,
                             sm_emparam, sm_transparam):
    input_filename = country + "/test.in"
    output_filename = country + "/test.split.out"
    indata = []
    with open(input_filename, "r") as infile:
        indata = infile.read().strip('\n').split('\n\n') #read and separate tweets
    
    with open(output_filename, "w") as outfile:
        for tweet in indata:
            word_sequence = tweet.split('\n')
            pred_id_tag, pred_sm_tag = prediction_function(word_sequence, words,
                                                            tags, stags,
                                                            id_emparam, id_transparam,
                                                            sm_emparam, sm_transparam)
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
    
    merge_tag_layer(c+"/test.split.out", c+"/test.p5.out")


if __name__ == '__main__':
    for c in ["EN", "FR"]:
        start = datetime.now()
        split_tag_layer(c+"/train", c+"/trainl")
        sm_data, id_data = read_splitlabel_file(c+"/trainl")
        sm_data = supress_infrequent_words(sm_data)
        id_data = supress_infrequent_words(id_data)
        
        words, tags, id_emparam = estimate_emission_param(id_data)
        id_transparam = estimate_transition_parameter(id_data)
        swords, stags, sm_emparam = estimate_emission_param(sm_data)
        sm_transparam = estimate_transition_parameter(sm_data)
        
        write_dualhmm_ontest(c, predict_dualtag_sequence,
                             words, tags, stags,
                             id_emparam, id_transparam,
                             sm_emparam, sm_transparam)
        end = datetime.now()
        delt = end - start
        print("{} test done in {}.{}s"\
              .format(c, delt.seconds, delt.microseconds))