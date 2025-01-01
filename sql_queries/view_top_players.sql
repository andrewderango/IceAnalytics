SELECT
  player,
  team,
  position,
  ROUND(games) AS games,
  ROUND(goals) AS goals,
  ROUND(assists) AS assists,
  ROUND(points) AS points,
  ROUND(art_ross, 4) AS art_ross,
  ROUND(rocket, 4) AS rocket
FROM player_projections
ORDER BY Points DESC
LIMIT 25