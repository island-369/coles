import warnings
import pandas as pd

from ptls.preprocessing.base.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class PredefinedCategoryEncoder(ColTransformerPandasMixin, ColCategoryTransformer):
    """使用预定义映射对类别字段进行编码

    此类使用一个预定义的字典来编码类别值。对于字典中不存在的未知值，
    会将其映射到一个指定的代码。

    参数
    ----------
    col_name_original:
        原始列的名称。
    col_name_target:
        目标列的名称。转换后的值将放置在此列中。
        如果 `col_name_target` 为 `None`，则原始列将被转换后的值替换。
    is_drop_original_col:
        当目标列和原始列不同时，控制是否删除原始列。
    predefined_mapping:
        将类别值映射到其编码ID的字典。
    unknown_value_code:
        分配给 `predefined_mapping` 中不存在的未知值的代码。
    """
    def __init__(self,
                 col_name_original: str,
                 predefined_mapping: dict,
                 unknown_value_code: int = None,
                 col_name_target: str = None,
                 is_drop_original_col: bool = True,
                 ):
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=col_name_target,
            is_drop_original_col=is_drop_original_col,
        )
        
        self.predefined_mapping = predefined_mapping
        # 如果未指定 unknown_value_code，则将其设置为预定义映射中最大值加1
        self.unknown_value_code = unknown_value_code if unknown_value_code is not None else max(predefined_mapping.values()) + 1

    def fit(self, x: pd.DataFrame):
        """拟合编码器（在此类中主要调用父类方法）"""
        super().fit(x)
        return self

    @property
    def dictionary_size(self):
        """返回字典的大小，包括预定义映射中的最大值和未知值代码"""
        # 字典大小是预定义映射中的最大值、unknown_value_code 和 1 中的最大值再加 1
        return max(max(self.predefined_mapping.values()), self.unknown_value_code) + 1

    def transform(self, x: pd.DataFrame):
        """转换DataFrame中的指定列

        将原始列中的值根据 `predefined_mapping` 进行映射。对于未知的或缺失的值，
        使用 `unknown_value_code` 填充，并确保数据类型为整数。
        """
        # 将原始列转换为字符串类型，以确保map方法正常工作
        pd_col = x[self.col_name_original].astype(str)
        # 映射值，将未映射的值填充为 unknown_value_code，然后转换为整数类型
        # 并将结果列重命名为目标列名
        transformed_col = pd_col.map(self.predefined_mapping).fillna(self.unknown_value_code).astype(int).rename(self.col_name_target)
        # 将转换后的列附加到DataFrame中
        x = self.attach_column(x, transformed_col)
        # 调用父类的transform方法进行后续处理（例如删除原始列）
        x = super().transform(x)
        return x