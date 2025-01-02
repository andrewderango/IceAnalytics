SELECT 
  date, 
  time_str AS time, 
  home_abbrev AS home, 
  visitor_abbrev AS visitor, 
  ROUND(home_prob, 3) AS home_prob,
  ROUND(visitor_prob, 3) AS visitor_prob, 
  ROUND(overtime_prob, 3) AS ot_prob,
  ROUND(home_score, 3) AS home_score,
  ROUND(visitor_score, 3) AS visitor_score 
FROM game_projections
WHERE (home_abbrev = 'MTL' and visitor_abbrev = 'TOR') or (home_abbrev = 'TOR' and visitor_abbrev = 'MTL')
ORDER BY game_id ASC