import sys
if "./" not in sys.path:
    sys.path.extend(["./", "../", ".../"])

import json
import pandas as pd

from segmentation.segment_type_identifier import AutoTypeFeatureSelector
from segmentation.cluster_segments import SegmentClusterer

class AutoSegmentor:
    """
    AutoSegmentor for a dataset. Collate and stream platform agnostic text + numerical data.

    run AutoSegmentor.get_output() to get a list of dicts

    Format : [{"segment_type" : <segment type from auto segment_type recognition>,
                "segment_attributes" : <attributes from default attribute fields>,
                "cluster_id" : <identifier for each cluster>
                "cluster_data" : <rows of cluster with the segment_attributes>} ....
            ]
    """

    def __init__(self, dataframe:pd.DataFrame):
        """
        Given a dataset of (known) attributes, suggests a data driven segmentation of data
        Clusters the data with the suggested segmentation.

        To be used as input for segment-persona generation via LLM

        Args:
            dataframe (pd.DataFrame): pd.DataFrame of relevant attributes
        """
        self.dataframe = dataframe

        #what attributes to generate segment-persona by -- increase list after PoC
        default_attributes = None
        with open("/Users/supreeth/HackContlo/segmentation/attribute_fields.json", "r") as file:
            default_attributes = json.load(file)

        self.default_attribute_fields = default_attributes["default_attributes"]

        self.segment_type = None
        self.required_columns = None

        self.output = pd.DataFrame()

    def auto_identify_segment_type(self):

        segment_type, required_columns = AutoTypeFeatureSelector(dataframe=self.dataframe).get_output()
        return segment_type, required_columns

    def auto_cluster_into_segments(self, attribute_columns:list) -> pd.DataFrame:

        segmentor = SegmentClusterer(dataframe=self.dataframe,
                                     dataframe_columns=attribute_columns)
        
        segment_outputs, _, __ = segmentor.get_output() #TODO : remove OPTICS and Birch from output of segmentor

        self.dataframe["cluster_segment"] = segment_outputs

        return self.dataframe

    def generate_llm_inputs_per_segment(self) -> list:
        """
        Get default attributes to use from a segment type map
        Get the output per cluster segment, collate X samples of relevant attributes at random
        Pass this information to an LLM to generate cluster segment persona for a cluster segment

        Returns:
            list: a list of dictionaries where each dictionary has information of a unique cluster segment based on auto-segmentation
        """

        #get the default attributes to use from segment type map
        #get the output per cluster segment, collate random X rows for input
        #form a dict of results
        segment_type, required_columns = self.auto_identify_segment_type()

        print(f"For the given dataset and its attributes, the best segmentation is `{segment_type}`")
        
        #update the segment_type and the attributes required to compute clusters on this segment_type!
        self.segment_type = segment_type
        self.required_columns = required_columns

        #density based cluster into segments -- ensures that only highly similar attribute groups appear
        clustered_segments = self.auto_cluster_into_segments(attribute_columns=self.required_columns)
        self.output = clustered_segments

        #retrieve the attribute types to use to form a segment persona via LLMs
        attribute_types = self.default_attribute_fields[self.segment_type]

        #put the segment data into a list of dicts so that an LLM can generate segment persona
        segment_types = self.output.drop_duplicates(subset=["cluster_segment"])

        #store results per cluster
        llm_inputs = []
        
        for _, row in segment_types.iterrows():

            unique_cluster_id = row["cluster_segment"]

            segment_data = self.output[self.output["cluster_segment"] == unique_cluster_id] #take data of the entire cluster
            segment_data = segment_data[attribute_types] #only slice the relatively important columns from the dataset
            segment_data = segment_data.sample(frac=1)[:10] #randomly shuffle data and only pick the top X data samples

            samples_from_segment = segment_data.to_dict('records') #convert into a json object so its easy to read

            llm_inputs.append({"segment_type" : self.segment_type,
                                "segment_attributes" : attribute_types,
                                "cluster_id" : unique_cluster_id,
                                "cluster_data" : samples_from_segment}
                                )
            
        return llm_inputs

    def get_output(self) -> list:
        """
        Get the basic suggested customer segmentation type, and attributes of the relevant segmentation type.
        Use samples of these as input and generate a cluster segment persona using LLM

        Returns:
            list: list of required input information for an LLM to generate cluster segment persona
        """

        #return the results as a list of dictionaries, they will become inputs to an LLM for segment persona generation
        return self.generate_llm_inputs_per_segment()
    
# if __name__ == "__main__":

#     df = pd.read_csv("data/cleaned_marketing_campaign.csv")
#     segmentor = AutoSegmentor(df)
#     print(segmentor.get_output())