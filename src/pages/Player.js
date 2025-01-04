import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import supabase from '../supabaseClient';
import '../styles/Player.scss';

function Player() {
  const { playerId } = useParams();
  const [player, setPlayer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('goals');

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
        console.log('Player details:', data); // temp for debugging
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

  const {
    player: playerName,
    team_name = 'NHL',
    team,
    position,
    jersey_number = 98,
    age,
    logo,
    games,
    goals,
    assists,
    points,
    art_ross,
    rocket,
    goals_90,
    assists_90,
    points_90,
    goals_benchmarks,
    assists_benchmarks,
    points_benchmarks,
  } = player;

  const pointsPerGame = (points / games).toFixed(2);
  const artRossProbability = (art_ross * 100).toFixed(2);
  const rocketRichardProbability = (rocket * 100).toFixed(2);

  const renderProbabilityContent = () => {
    switch (activeTab) {
      case 'goals':
        return (
          <>
            <div className="probability-item">
              {(goals_90 * 100).toFixed(2)}%
              <div className="label">90% Prediction Interval</div>
            </div>
            <div className="probability-item">
              {(goals_benchmarks * 100).toFixed(2)}%
              <div className="label">Probability Benchmarks</div>
            </div>
          </>
        );
      case 'assists':
        return (
          <>
            <div className="probability-item">
              {(assists_90 * 100).toFixed(2)}%
              <div className="label">90% Prediction Interval</div>
            </div>
            <div className="probability-item">
              {(assists_benchmarks * 100).toFixed(2)}%
              <div className="label">Probability Benchmarks</div>
            </div>
          </>
        );
      case 'points':
        return (
          <>
            <div className="probability-item">
              {(points_90 * 100).toFixed(2)}%
              <div className="label">90% Prediction Interval</div>
            </div>
            <div className="probability-item">
              {(points_benchmarks * 100).toFixed(2)}%
              <div className="label">Probability Benchmarks</div>
            </div>
          </>
        );
      default:
        return null;
    }
  };

  return (
    <div className="player">
      <div className="header-bar">
        <div className="player-headshot-name">
          {team_name && playerId && (
            <img
              src={player.espn_headshot !== "N/A" ? player.espn_headshot : `https://assets.nhle.com/mugs/nhl/20242025/${team}/${playerId}.png`}
              alt={`${playerName} headshot`}
              className="player-headshot"
            />
          )}
          <div className="player-name">
            {playerName}
            <div className="player-details">#{jersey_number} - {team_name}</div>
          </div>
        </div>
        <div className="player-info">
          <div className="player-position">
            <span className="label">Position</span> {position}
          </div>
          <div className="player-age">
            <span className="label">Age</span> {age}
          </div>
        </div>
        {logo && <img src={logo} alt={`${playerName} logo`} className="team-logo" />}
      </div>
      <div className="content">
        <div className="left-content">
          <div className="projections">
            <div className="header">Projections</div>
            <div className="projections-row">
              <div className="projection-item">
                {Math.round(games)}
                <div className="label">Games</div>
              </div>
              <div className="projection-item">
                {Math.round(goals)}
                <div className="label">Goals</div>
              </div>
              <div className="projection-item">
                {Math.round(assists)}
                <div className="label">Assists</div>
              </div>
              <div className="projection-item">
                {Math.round(points)}
                <div className="label">Points</div>
              </div>
              <div className="projection-item">
                {pointsPerGame}
                <div className="label">P/GP</div>
              </div>
            </div>
            <div className="awards-row">
              <div className="wide-projection-item">
                {artRossProbability}%
                <div className="label">Art Ross Trophy</div>
              </div>
              <div className="wide-projection-item">
                {rocketRichardProbability}%
                <div className="label">Rocket Richard Trophy</div>
              </div>
            </div>
          </div>
          <div className="probabilities">
            <div className="tabs">
              <div className={`tab ${activeTab === 'goals' ? 'active' : ''}`} onClick={() => setActiveTab('goals')}>
                Goals
              </div>
              <div className={`tab ${activeTab === 'assists' ? 'active' : ''}`} onClick={() => setActiveTab('assists')}>
                Assists
              </div>
              <div className={`tab ${activeTab === 'points' ? 'active' : ''}`} onClick={() => setActiveTab('points')}>
                Points
              </div>
            </div>
            <div className="probability-content">
              {renderProbabilityContent()}
            </div>
          </div>
        </div>
        <div className="right-content">
          <div className="bio-title">Charts</div>
        </div>
      </div>
    </div>
  );
}

export default Player;