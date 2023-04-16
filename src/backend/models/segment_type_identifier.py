import json
from collections import defaultdict

class TypeSegmentor:
    def __init__(self, dataframe_columns:list):
        
        self.columns = [col.lower() for col in dataframe_columns]
        self.types = defaultdict(list)
        
        self.json_path = "src/backend/configs/config.json"
        
    def read_types(self, json_path:str):
        
        file_content = {}
        with open(json_path, "r") as type_file:
            file_content = json.load(type_file)
            
        return file_content
    
    def get_best_type_match(self) -> dict:
        
        self.types = self.read_types(self.json_path)
        self.types = dict(self.types)
        
        best_type_match, full_attribute_columns = "", []
        max_supported_columns = -999
        
        for _type, _full_attr_cols in self.types.items():
            supported_columns = [col for col in _full_attr_cols if col.lower() in self.columns]
            num_supported_columns = len(supported_columns)
            
            if num_supported_columns > max_supported_columns:
                best_type_match, full_attribute_columns = _type, supported_columns

                max_supported_columns = num_supported_columns
        
        return {"segment_type" : best_type_match,
                "segment_columns" : full_attribute_columns}
    
    def get_output(self) -> dict:
        return self.get_best_type_match()


class AutoTypeFeatureSelector(TypeSegmentor):
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.columns = dataframe.columns
        
        super().__init__(dataframe_columns=self.columns)
        
    def find_attribute_columns(self):
        
        attr_dict = self.get_best_type_match()
        
        segment_type = attr_dict["segment_type"]
        segment_columns = attr_dict["segment_columns"]
        
        return segment_type, segment_columns
    
    def get_output(self):
        
        segment_type, segment_columns = self.find_attribute_columns()
        
        self.segment_columns = segment_columns
        self.segment_type = segment_type
        
        return self.segment_type, self.segment_columns