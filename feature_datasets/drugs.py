from .common import *
import random
import pandas as pd

DRUG_PROMPTS = {
    'random': '',
    'condition': 'What is the usage condition of ',
    'empty_all_caps': '',
    'empty': '',
}