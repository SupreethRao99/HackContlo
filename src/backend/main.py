import time
from pprint import pprint 
import pandas as pd

from models.auto_segment import AutoSegmentor
from models.persona_generation import PersonaGenerator
from models.strategy_generation import StrategyGenerator

if __name__ == "__main__":
    df = pd.read_csv('../../data/cleaned_marketing_campaign.csv')

    segmentor = AutoSegmentor(df)

    clustered_segments = segmentor.get_output()

    persona_gen = PersonaGenerator()
    strategy_gen = StrategyGenerator()

    for cluster in clustered_segments:
        persona = persona_gen.get_user_persona(
            segmentation_type=cluster["cluster_data"], 
            attributes=cluster["segment_attributes"], 
            cluster_data=cluster["cluster_data"]
        )
        pprint(persona)
        
        retention_strategy = strategy_gen.get_retention_strategy(
            user_persona=persona, 
            segmention_type=cluster["segment_type"]
        )
        pprint(retention_strategy)
        
        time.sleep(3)