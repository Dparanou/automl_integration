from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit

class RandomSearch:
    def __init__(self):
        self.method = RandomizedSearchCV()