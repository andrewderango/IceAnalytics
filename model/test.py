import time
import pandas as pd
from model_training import *
from model_inference import *
from scraper_functions import *

scrape_teams(True, True)
scrape_games(2025, False, True)