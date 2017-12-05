'''
Part 4
'''
from datetime import datetime
from utility import read_labeled_file, estimate_emission_param, supress_infrequent_words
from hmm import write_hmm_prediction, estimate_transition_parameter

def predict_tag_sequence_maxmarginal(word_sequence, words, tags,
                                     trans_param, em_param):
    '''
    using max-marginal decoding algorithm
    '''

    # ----- Alpha Function -----
    alphas = {}
    def alpha_forward(tag, stage):
        '''
        Forward algorithm
        '''
        if stage <= 1:
            # base case
            score = trans_param.get(("START", tag), 0)
            alphas[(tag, stage)] = score
            return score
        else:
            if (tag, stage) in alphas:
                # return stored value, if any
                return alphas[(tag, stage)]
            score = 0
            for prev_tag in tags:
                prev_score = alpha_forward(prev_tag, stage-1) # recursion
                trans_prob = trans_param.get((prev_tag, tag), 0)
                # handle undiscovered word
                word = word_sequence[stage-2]
                if word not in words:
                    word = "#UNK#"
                em_prob = em_param.get((word, prev_tag), 0)
                curr_score = prev_score*trans_prob*em_prob
                # sum the looped score
                score += curr_score
            alphas[(tag, stage)] = score
            return score

    # ----- Beta Function -----
    betas = {}
    def beta_back(tag, stage):
        '''
        Backward algorithm
        '''
        if stage >= len(word_sequence):
            # if last word in sequence
            trans_prob = trans_param.get((tag, "STOP"), 0)
            word = word_sequence[stage-1]
            em_prob = em_param.get((word, tag), 0)
            score = trans_prob*em_prob
            betas[(tag, stage)] = score
            return score
        else:
            if (tag, stage) in betas:
                return betas[(tag, stage)]
            score = 0
            for t in tags:
                prev_score = beta_back(t, stage+1) # recursion
                trans_prob = trans_param.get((tag, t), 0)
                word = word_sequence[stage-1]
                if word not in words:
                    word = "#UNK#"
                em_prob = em_param.get((word, tag), 0)
                curr_score = prev_score*trans_prob*em_prob
                score += curr_score
            betas[(tag, stage)] = score
            return score

    tag_seq = ["START"]

    lenw = len(word_sequence)
    for i in range(1, lenw+1):
        # --- this part is to support Part 5 ---#
        #         where it has 'none' tag
        if "none" in tags:
            max_tag = "none"
        else:
            max_tag = "O"
        max_weight = 0
        for tag in tags:
            alph = alpha_forward(tag, i)
            beth = beta_back(tag, i)
            curr_weight = alph*beth
            if curr_weight > max_weight:
                max_weight = curr_weight
                max_tag = tag
        tag_seq.append(max_tag)

    tag_seq.append("STOP")
    return tag_seq


if __name__ == '__main__':
    for c in ["EN", "FR"]:
        start = datetime.now()
        data = read_labeled_file(c+"/train")
        sdata = supress_infrequent_words(data)
        words, tags, em_param = estimate_emission_param(sdata)
        trans_param = estimate_transition_parameter(sdata)
        write_hmm_prediction(c, "4",
                             predict_tag_sequence_maxmarginal,
                             words, tags, em_param, trans_param)
        end = datetime.now()
        delt = end - start
        print("{} part 4 done in {}.{}s"\
              .format(c, delt.seconds, delt.microseconds))
