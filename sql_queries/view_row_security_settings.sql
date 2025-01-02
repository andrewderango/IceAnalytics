SELECT relname, relrowsecurity, relforcerowsecurity 
FROM pg_class 
WHERE relname = '2025_team_aggregated_projections';