from scraper_functions import *

PROJECTION_YEAR = 2025
# push_to_supabase("team-projections", PROJECTION_YEAR, True)
# push_to_supabase("player-projections", PROJECTION_YEAR, True)
# push_to_supabase("game-projections", PROJECTION_YEAR, True)
push_to_supabase("last-update", PROJECTION_YEAR, True)