SELECT
  player,
  team,
  position,
  ROUND(games) AS games,
  ROUND(goals) AS goals,
  ROUND(assists) AS assists,
  ROUND(points) AS points
FROM player_projections
ORDER BY Points DESC
LIMIT 25