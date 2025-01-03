import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import supabase from '../supabaseClient';
import '../styles/Player.scss';

function Player() {
  const { playerId } = useParams();
  const [player, setPlayer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPlayer = async () => {
      if (!playerId) {
        setError('Player ID is undefined');
        setLoading(false);
        return;
      }

      const { data, error } = await supabase
        .from('player_projections')
        .select('*')
        .eq('player_id', playerId)
        .single();

      if (error) {
        console.error('Error fetching player data:', error);
        setError(error.message);
      } else {
        setPlayer(data);
      }
      setLoading(false);
    };

    fetchPlayer();
  }, [playerId]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!player) {
    return <div>Player not found</div>;
  }

  // Ensure all numeric fields have default values if they are undefined
  const {
    player: playerName,
    team,
    position,
    games = 0,
    goals = 0,
    assists = 0,
    points = 0,
    logo,
  } = player;

  return (
    <div className="player">
      <div className="header-bar">
        {team && playerId && (
          <img
            src={player.espn_headshot !== "N/A" ? player.espn_headshot : `https://assets.nhle.com/mugs/nhl/20242025/${team}/${playerId}.png`}
            alt={`${playerName} headshot`}
            className="player-headshot"
          />
        )}
        <div className="player-name">{playerName}</div>
        {logo && <img src={logo} alt={`${playerName} logo`} className="team-logo" />}
      </div>
      <div className="content">
        <p><strong>Team:</strong> {team}</p>
        <p><strong>Position:</strong> {position}</p>
        <p><strong>Games Played:</strong> {games}</p>
        <p><strong>Goals:</strong> {goals}</p>
        <p><strong>Assists:</strong> {assists}</p>
        <p><strong>Points:</strong> {points}</p>
      </div>
    </div>
  );
}

export default Player;