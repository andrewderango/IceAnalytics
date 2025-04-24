import React, { useState, useEffect } from 'react';
import { useParams, useHistory } from 'react-router-dom';
import supabase from '../supabaseClient';
import { Scatter } from 'react-chartjs-2';
import 'chart.js/auto';
import '../styles/Player.scss';
import { offseason } from '../config/settings';

function Player() {
  const { playerId } = useParams();
  const history = useHistory();
  const [player, setPlayer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('points');
  const [allPlayers, setAllPlayers] = useState([]);
  const [chartKey, setChartKey] = useState(0);
  const [isScreenSmallerThan1475, setIsScreenSmallerThan1475] = useState(window.innerWidth < 1475);
  const [isScreenSmallerThan1000, setIsScreenSmallerThan1000] = useState(window.innerWidth <= 1000);

  useEffect(() => {
    if (offseason) {
      history.push('/not-found');
    }
  }, [history]);

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
        .eq('player_id', playerId);

      if (error || data.length !== 1) {
        console.error('Error fetching player data:', error || 'Multiple or no rows returned');
        history.push('/not-found');
      } else {
        setPlayer(data[0]);
        console.log('Player details:', data[0]); // temp for debugging
      }
      setLoading(false);
    };

    fetchPlayer();
  }, [playerId, history]);

  useEffect(() => {
    const fetchAllPlayers = async () => {
      const { data, error } = await supabase
        .from('player_projections')
        .select('player_id, goals, assists');

      if (error) {
        console.error('Error fetching all players data:', error);
      } else {
        setAllPlayers(data);
      }
    };

    fetchAllPlayers();
  }, []);

  useEffect(() => {
    const handleResize = () => {
      setChartKey(prevKey => prevKey + 1);
      setIsScreenSmallerThan1475(window.innerWidth < 1475);
      setIsScreenSmallerThan1000(window.innerWidth <= 1000);
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

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
    jersey_number,
    age,
    logo,
    games,
    goals,
    assists,
    points,
    art_ross,
    rocket,
    goals_90pi_low,
    goals_90pi_high,
    assists_90pi_low,
    assists_90pi_high,
    points_90pi_low,
    points_90pi_high,
    p_20g,
    p_30g,
    p_40g,
    p_50g,
    p_60g,
    p_25a,
    p_50a,
    p_75a,
    p_100a,
    p_50p,
    p_75p,
    p_100p,
    p_125p,
    p_150p
  } = player;

  const pointsPerGame = (points / games).toFixed(2);
  const artRossProbability = (art_ross * 100).toFixed(2);
  const rocketRichardProbability = (rocket * 100).toFixed(2);

  const renderPIBox = (low, high) => (
    <div className="pi-row">
      <div className="pi-item">
        {low.toFixed(1)}
        <div className="label">90% PI Min</div>
      </div>
      <div className="pi-item">
        {high.toFixed(1)}
        <div className="label">90% PI Max</div>
      </div>
    </div>
  );

  const renderBenchmarks = (benchmarks) => (
    <div className="benchmarks-row">
      {benchmarks.map((benchmark, index) => (
        <div key={index} className="benchmark-item">
          {isScreenSmallerThan1475 && benchmark.value === 1.0 ? (benchmark.value * 100).toFixed(0) : (benchmark.value * 100).toFixed(1)}%
          <div className="label">
            {isScreenSmallerThan1475 ? benchmark.shortLabel.replace(' ', '\n') : benchmark.label}
          </div>
        </div>
      ))}
    </div>
  );

  const renderProbabilityContent = () => {
    switch (activeTab) {
      case 'goals':
        return (
          <>
            {renderPIBox(goals_90pi_low, goals_90pi_high)}
            {renderBenchmarks([
              { value: p_20g, label: '20+ Goals', shortLabel: '20+ Goals' },
              { value: p_30g, label: '30+ Goals', shortLabel: '30+ Goals' },
              { value: p_40g, label: '40+ Goals', shortLabel: '40+ Goals' },
              { value: p_50g, label: '50+ Goals', shortLabel: '50+ Goals' },
              { value: p_60g, label: '60+ Goals', shortLabel: '60+ Goals' },
            ])}
          </>
        );
      case 'assists':
        return (
          <>
            {renderPIBox(assists_90pi_low, assists_90pi_high)}
            {renderBenchmarks([
              { value: p_25a, label: '25+ Assists', shortLabel: '25+ Assists' },
              { value: p_50a, label: '50+ Assists', shortLabel: '50+ Assists' },
              { value: p_75a, label: '75+ Assists', shortLabel: '75+ Assists' },
              { value: p_100a, label: '100+ Assists', shortLabel: '100+ Assists' },
            ])}
          </>
        );
      case 'points':
        return (
          <>
            {renderPIBox(points_90pi_low, points_90pi_high)}
            {renderBenchmarks([
              { value: p_50p, label: '50+ Points', shortLabel: '50+ Points' },
              { value: p_75p, label: '75+ Points', shortLabel: '75+ Points' },
              { value: p_100p, label: '100+ Points', shortLabel: '100+ Points' },
              { value: p_125p, label: '125+ Points', shortLabel: '125+ Points' },
              { value: p_150p, label: '150+ Points', shortLabel: '150+ Points' },
            ])}
          </>
        );
      default:
        return null;
    }
  };

  const scatterData = {
    datasets: [
      {
        label: 'All Players',
        data: allPlayers
          .filter(p => p.goals !== goals && p.assists !== assists)
          .map(p => ({ x: p.goals, y: p.assists })),
        backgroundColor: 'rgba(75, 75, 75, 0.6)',
        order: 2,
      },
      {
        label: playerName,
        data: [{ x: goals, y: assists }],
        backgroundColor: 'goldenrod',
        pointRadius: 4,
        order: 1,
      },
    ],
  };

  const scatterOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Goals',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Assists',
        },
      },
    },
    plugins: {
      tooltip: {
        enabled: false,
      },
      legend: {
        display: true,
        position: 'bottom',
      },
    },
    hover: {
      mode: null,
    },
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
        {!isScreenSmallerThan1000 && (
          <div className="player-info">
            <div className="player-position">
              <span className="label">Position</span> {position}
            </div>
            <div className="player-age">
              <span className="label">Age</span> {age}
            </div>
          </div>
        )}
        {!isScreenSmallerThan1000 && logo && (
          <img src={logo} alt={`${playerName} logo`} className="team-logo" />
        )}
      </div>
      {isScreenSmallerThan1000 && (
        <div className="info-bar">
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
      )}
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
              <div className="award-item">
                {artRossProbability}%
                <div className="label">Art Ross Trophy</div>
              </div>
              <div className="award-item">
                {rocketRichardProbability}%
                <div className="label">Rocket Richard Trophy</div>
              </div>
            </div>
          </div>
          <div className="probabilities">
            <div className="header">Benchmark Probabilities</div>
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
          <div className="chart-container">
            <div className="header">Relative Production Efficiency</div>
            <Scatter key={chartKey} data={scatterData} options={scatterOptions} />
          </div>
        </div>
      </div>
      <div className="spacer"></div>
    </div>
  );
}

export default Player;