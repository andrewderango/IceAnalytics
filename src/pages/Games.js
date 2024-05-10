import React, { useState, useEffect } from 'react';
import '../styles/Games.scss';

function Games() {
  const [games, setGames] = useState([]);

  // useEffect(() => {
  // Fetch the games data from an API or local data source
  //   fetch('/api/games')
  //     .then(response => response.json())
  //     .then(data => setGames(data));
  // }, []);

  // Test
  useEffect(() => {
    const sampleData = [
      {
        id: 1,
        time: '7:00 PM EST',
        team1: {
          name: 'Pittsburgh Penguins',
          logo: 'https://assets.nhle.com/logos/nhl/svg/PIT_dark.svg',
          record: '5-3-0 (23rd)',
          probability: 0.683,
          projectedGoals: 3
        },
        team2: {
          name: 'Edmonton Oilers',
          logo: 'https://assets.nhle.com/logos/nhl/svg/EDM_dark.svg',
          record: '4-4-1 (27th)',
          probability: 0.317,
          projectedGoals: 3
        }
      },
      {
        id: 2,
        time: '7:30 PM EST',
        team1: {
          name: 'Montreal Canadiens',
          logo: 'https://assets.nhle.com/logos/nhl/svg/MTL_dark.svg',
          record: '2-1-0 (31st)',
          probability: 0.500,
          projectedGoals: 3
        },
        team2: {
          name: 'Toronto Maple Leafs',
          logo: 'https://assets.nhle.com/logos/nhl/svg/TOR_dark.svg',
          record: '7-1-5 (1st)',
          probability: 0.500,
          projectedGoals: 3
        }
      },
      {
        id: 2,
        time: '8:00 PM EST',
        team1: {
          name: 'Boston Bruins',
          logo: 'https://assets.nhle.com/logos/nhl/svg/BOS_dark.svg',
          record: '2-1-0 (31st)',
          probability: 0.598,
          projectedGoals: 3
        },
        team2: {
          name: 'Minnesota Wild',
          logo: 'https://assets.nhle.com/logos/nhl/svg/MIN_dark.svg',
          record: '7-1-5 (1st)',
          probability: 0.402,
          projectedGoals: 3
        }
      },
      {
        id: 2,
        time: '9:00 PM EST',
        team1: {
          name: 'New York Rangers',
          logo: 'https://assets.nhle.com/logos/nhl/svg/NYR_dark.svg',
          record: '2-1-0 (31st)',
          probability: 0.387,
          projectedGoals: 3
        },
        team2: {
          name: 'Tampa Bay Lightning',
          logo: 'https://assets.nhle.com/logos/nhl/svg/TBL_dark.svg',
          record: '7-1-5 (1st)',
          probability: 0.613,
          projectedGoals: 3
        }
      },
      {
        id: 2,
        time: '10:00 PM EST',
        team1: {
          name: 'Vancouver Canucks',
          logo: 'https://assets.nhle.com/logos/nhl/svg/VAN_dark.svg',
          record: '2-1-0 (31st)',
          probability: 0.460,
          projectedGoals: 3
        },
        team2: {
          name: 'Florida Panthers',
          logo: 'https://assets.nhle.com/logos/nhl/svg/FLA_dark.svg',
          record: '7-1-5 (1st)',
          probability: 0.540,
          projectedGoals: 3
        }
      },
    ];

    setGames(sampleData);
  }, []);

  return (
    <div className="games">
      <h1>Games</h1>
      <h2 className="date">Thursday, May 9, 2024</h2>
      {games.map(game => (
          <div className="game" key={game.id}>
            <div className="column1">
              <img src={game.team1.logo} alt={game.team1.name} />
              <p className="team-name">{game.team1.name}</p>
              <p className="record">{game.team1.record}</p>
            </div>
            <div className="column2">
              <p className="probability">{(game.team1.probability*100).toFixed(1)}%</p>
              <p className="projected-goals">Proj. Goals: {game.team1.projectedGoals.toFixed(2)}</p>
            </div>
            <div className="time">
              {game.time}
            </div>
            <div className="column2">
              <p className="probability">{(game.team2.probability*100).toFixed(1)}%</p>
              <p className="projected-goals">Proj. Goals: {game.team2.projectedGoals.toFixed(2)}</p>
            </div>
            <div className="column1">
              <img src={game.team2.logo} alt={game.team2.name} />
              <p className="team-name">{game.team2.name}</p>
              <p className="record">{game.team2.record}</p>
            </div>
          </div>
      ))}
    </div>
  );
}

export default Games;