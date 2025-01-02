WITH max_probs AS (
  SELECT 
    date, 
    time_str AS time, 
    home_abbrev AS home, 
    visitor_abbrev AS visitor, 
    ROUND(home_prob, 3) AS home_prob,
    ROUND(visitor_prob, 3) AS visitor_prob, 
    ROUND(overtime_prob, 3) AS ot_prob,
    GREATEST(home_prob, visitor_prob) AS max_prob,
    ROUND(home_score, 3) AS home_score,
    ROUND(visitor_score, 3) AS visitor_score
  FROM game_projections
  WHERE home_prob NOT IN (1.0, 0.0)
)

SELECT 
  date, 
  time, 
  home, 
  visitor, 
  home_prob, 
  visitor_prob, 
  ot_prob,
  ROUND(home_score, 3) AS home_score,
  ROUND(visitor_score, 3) AS visitor_score 
FROM max_probs
ORDER BY max_prob DESC
LIMIT 10;