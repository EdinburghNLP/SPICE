# meter class for storing results
class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.wrong = 0
        self.accuracy = 0
        self.exact_match_acc = 0.0
        self.number_of_instance = 0
        self.correct_exact_match = 0.0

    def update(self, gold, result, gold_sparql, pred_sparql):
        self.number_of_instance += 1
        if gold_sparql is not None and pred_sparql is not None and gold_sparql.lower() == pred_sparql.lower():
            self.correct_exact_match += 1
        if gold == result:
            self.correct += 1
        else:
            self.wrong += 1

        self.accuracy = self.correct / (self.correct + self.wrong)
        self.exact_match_acc = self.correct_exact_match / self.number_of_instance

class F1scoreMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.exact_match_acc = 0
        self.correct_exact_match = 0.0
        self.number_of_instance = 0.0
        self.missmatch = 0.0
        ## debug
        self.acc_prec_macro = 0.0
        self.acc_rec_macro = 0.0
        self.acc_f1_macro = 0.0

    def update(self, gold, result, gold_sparql, pred_sparql):
        self.number_of_instance += 1
        if gold_sparql is not None and pred_sparql is not None and gold_sparql.lower() == pred_sparql.lower():
            self.correct_exact_match += 1
            if result != gold:
                self.missmatch += 1
                # debug
                print(gold_sparql)
                print('result', result)
                print('gold', gold)
                print('****** EM but <> results ******')

        self.tp += len(result.intersection(gold))
        self.fp += len(result.difference(gold))
        self.fn += len(gold.difference(result))
        if self.tp > 0 or self.fp > 0:
            self.precision = self.tp / (self.tp + self.fp)
        if self.tp > 0 or self.fn > 0:
            self.recall = self.tp / (self.tp + self.fn)
        if self.precision > 0 or self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)

        prec, rec = 0,0
        if len(result) > 0:
            #print(f'Instance precision: {len(result.intersection(gold)) / len(result)}')
            prec = len(result.intersection(gold)) / len(result)
            self.acc_prec_macro += prec
        if len(gold) > 0:
            rec = len(result.intersection(gold)) / len(gold)
            self.acc_rec_macro += rec
        if prec > 0 or rec > 0:
            self.acc_f1_macro += 2 * prec * rec / (prec + rec)

        ###
        #if (len(result.intersection(gold))!= len(gold) or len(result.intersection(gold))!= len(result)) and \
        #        (gold_sparql is not None and pred_sparql is not None and gold_sparql.lower() == pred_sparql.lower()):
        #    print('gold', gold)
        #    print('result', result)
        #    print('prec/rec', self.precision, self.recall)
        #    print('rec fla', self.tp / (self.tp + self.fn), self.tp, self.fn)
        #    print('tp/inter', result.intersection(gold), len(result.intersection(gold)))
        #    print('fn/diff', gold.difference(result), len(gold.difference(result)))
        #    exit()
        
        self.exact_match_acc = self.correct_exact_match / self.number_of_instance


## Unused???
class ExactMatchMeter(object):
    def __init__(self):
        self.total = 0
        self.correct = 0
    
    def update(self, gold, pred):
        pass
