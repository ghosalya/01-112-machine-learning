'''
Part 3 - HMM Implementation
'''
from datetime import datetime
from utility import read_labeled_file, estimate_emission_param, supress_infrequent_words

def estimate_transition_parameter(data):
    '''
    takes a list of (word, tag) tuple
    returns a dictionary of estimated transition parameter
    with format <(tag0, tag1):count>
    '''
    
    tag_to_tag_count = {}
    tag_count = {"START": 0, "STOP": 0}
    for tweet in data:
        # due to data structure, we have to add
        # a dummy word with the STOP tag
        aptweet = tweet + [("endword", "STOP")]
        
        for i in range(len(aptweet)):
            # looping each tag, checking previous tag
            tag = aptweet[i][1]
            # keep track of tag count
            tag_count[tag] = tag_count.get(tag, 0) + 1
                
            if i == 0:
                # no previous tag, transitioned from START
                tag_count["START"] += 1
                tag_transition = ("START", tag)
            else:
                prevtag = aptweet[i-1][1]
                tag_transition = (prevtag, tag)
                
            # track the tag_transition count
            prev_count = tag_to_tag_count.get(tag_transition, 0)
            tag_to_tag_count[tag_transition] = prev_count + 1
            
    # once the count is settled, we can generate the probability
    estimated_param = {k: tag_to_tag_count[k]/tag_count[k[0]]
                       for k in tag_to_tag_count}
    return estimated_param


def predict_tag_sequence(word_sequence, words,
                         tags, trans_param, em_param):
    '''
    takes:
        - the list of words (in sequence) to predict
        - a list of known words
        - a list of known tags
        - a dictionary of transition parameter
        - a dictionary of emission parameter
    returns:
        - a list of tags in sequence
    '''
    
    # viterbi's dynamic programming
    pi_dp = {(0, "START"):1}
    tags = list(tags)
    # add START to known tags just in case
    tags.append("START")
    
    # viterbi function to recurse forward
    def viterbi_pi(stage, tag):
        '''
        takes:
            - int stage (position in sequence)
            - string tag
        returns highest value when preceeded with 
                suitable tag
        '''
        if stage == 0:
            # base case, at stage 0 tag should be START
            result = 1.0 if tag == "START" else 0.0
            return result
        elif stage >= len(word_sequence)+1:
            # at stage N there is no emission
            max_weight = 0
            for prev_tag in tags:
                prev_cost = viterbi_pi(stage-1, prev_tag) # recursion
                trans_prob = trans_param.get((prev_tag, tag), 0)
                current_weight = prev_cost*trans_prob
                # compare with highest discovered weight
                if max_weight < current_weight:
                    max_weight = current_weight
            # for dynamic programming
            pi_dp[(stage, tag)] = max_weight
            return max_weight
        else:
            # any other case, considers emission & transition
            if (stage, tag) in pi_dp:
                # if the value is saved by dynamic programming
                return pi_dp[(stage, tag)]
            else:
                max_weight = 0
                for prev_tag in tags:
                    prev_cost = viterbi_pi(stage-1, prev_tag) #recursion
                    trans_prob = trans_param.get((prev_tag, tag), 0)
                    
                    word = word_sequence[stage-1]
                    # handling undiscovered word
                    if word not in words:
                        word = "#UNK#"
                    em_prob = em_param.get((word, tag), 0)
                    # compare with highest discovered weight
                    curr_weight = prev_cost*trans_prob*em_prob
                    if max_weight < curr_weight:
                        max_weight = curr_weight
                # for dynamic programming   
                pi_dp[(stage, tag)] = max_weight
                return max_weight

    # run recursive function on all tags
    for i in range(len(word_sequence)+1):
        for t in tags:
            viterbi_pi(i, t)
    
    tag_seqr = ["STOP"] # reversd tag sequence
    
    # backward checking for tag sequence
    lenw = len(word_sequence)
    for i in range(lenw+1):
        if i == lenw:
            # reaches start of word sequence
            tag_seqr.append("START")
            continue
            
        # --- this part is to support Part 5 ---#
        #         where it has 'none' tag
        if "none" in tags:
            max_tag = "none"
        else:
            max_tag = "O"
        
        # stars working backward to get tag sequence
        max_weight = 0
        for tag in tags:
            prev_prob = viterbi_pi(lenw-i, tag)
            next_tag = tag_seqr[-1]
            trans_prob = trans_param.get((tag, next_tag), 0)
            curr_weight = prev_prob*trans_prob
            if max_weight < curr_weight:
                max_weight = curr_weight
                max_tag = tag # get argmax
        tag_seqr.append(max_tag)
        
    # reverse the tag_seqr to get tag_sequence
    return tag_seqr[::-1]


def write_hmm_prediction(country, part, prediction_function,
                         words, tags, em_param, trans_param):
    '''
    Function to write HMM prediction
    '''
    input_filename = country + "/dev.in"
    output_filename = country + "/dev.p"+part+".out"
    indata = []
    with open(input_filename, "r") as infile:
        indata = infile.read().strip('\n').split('\n\n') #read and separate tweets
    
    with open(output_filename, "w") as outfile:
        for tweet in indata:
            word_sequence = tweet.split('\n')
            predicted_tag_sequence = prediction_function(word_sequence, words,
                                                tags, trans_param, em_param)
            predicted_tag_sequence.remove("START")
            predicted_tag_sequence.remove("STOP")
            if len(word_sequence) != len(predicted_tag_sequence):
                print("WARNING!! Different length {} / {}"\
                      .format(word_sequence, predicted_tag_sequence))
            for i in range(len(word_sequence)):
                line = "{} {}\n".format(word_sequence[i], 
                                        predicted_tag_sequence[i])
                outfile.write(line)
            outfile.write("\n")



if __name__ == '__main__':
    for c in ["CN", "EN", "SG", "FR"]:
        start = datetime.now()
        data = read_labeled_file(c+"/train")
        sdata = supress_infrequent_words(data)
        words, tags, em_param = estimate_emission_param(sdata)
        trans_param = estimate_transition_parameter(sdata)
        write_hmm_prediction(c,"3", predict_tag_sequence,
                            words, tags, em_param, trans_param)
        end = datetime.now()
        delt = end - start
        print("{} part 3 done in {}.{}s"\
              .format(c, delt.seconds, delt.microseconds))