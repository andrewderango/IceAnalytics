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
    team = 'Chicago Blackhawks',
    position,
    jersey_number = 98,
    age,
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
        <div className="player-name">
          {playerName}
          <div className="player-details">#{jersey_number} - {team}</div>
        </div>
        <div className="player-info">
          <div className="player-age">
            <span className="label">Age</span> {age}
          </div>
          <div className="player-position">
            <span className="label">Position</span> {position}
          </div>
        </div>
        {logo && <img src={logo} alt={`${playerName} logo`} className="team-logo" />}
      </div>
    </div>
  );
}

export default Player;