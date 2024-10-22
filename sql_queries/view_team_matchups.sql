SELECT 
  date, 
  time_str AS time, 
  home_abbrev AS home, 
  visitor_abbrev AS visitor, 
  ROUND(home_prob, 3) AS home_prob,
  ROUND(visitor_prob, 3) AS visitor_prob, 
  ROUND(overtime_prob, 3) AS ot_prob
FROM game_projections
WHERE (home_abbrev = 'MTL' and visitor_abbrev = 'TOR') or (home_abbrev = 'TOR' and visitor_abbrev = 'MTL')
ORDER BY game_id ASC