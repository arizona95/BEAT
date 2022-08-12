import pandas as pd

class userType :
    def __init__(self) :
        self.pandas = dict()
        self.numpy = dict()
        self.dictionary = dict()


    def form_expand(self):
        if len(self.dictionary) != 0 :
            self.pandas = pd.DataFrame.from_dict(self.dictionary)

