from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class TextEncoder(ColTransformerPandasMixin,ColTransformer):
    def transform(self, x):
        return 
    