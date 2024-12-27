import React, { useState, useEffect } from 'react';
import '../styles/Games.scss';
import { GridLoader } from 'react-spinners';
import { createClient } from '@supabase/supabase-js';
import noGamesImage from '../assets/images/404.png';

function Games() {
  const supabaseUrl = process.env.REACT_APP_SUPABASE_PROJ_URL;
  const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
  const supabase = createClient(supabaseUrl, supabaseAnonKey);
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState('');

  // get the current date in EST
  const estDate = new Date().toLocaleString('en-US', { timeZone: 'America/New_York' });
  const [month, day, year] = estDate.split(',')[0].split('/');
  const date = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
  const [currentDate, setCurrentDate] = useState(date);

  const fetchData = async (fetchDate) => {
    const { data: games, error } = await supabase
      .from('game_projections')
      .select('*')
      .eq('date', fetchDate);

    if (error) {
      console.error('Error fetching data:', error);
    } else {
      setGames(games);
      setLoading(false);
    }
  };

  const fetchMetadata = async () => {
    try {
      const { data, error } = await supabase
        .from('last_update')
        .select('datetime')
        .order('datetime', { ascending: false })
        .limit(1);

      if (error) {
        console.error('Error fetching metadata:', error);
        return;
      }

      if (data.length > 0) {
        const timestamp = new Date(data[0].datetime);
        let formattedDate;

        if (window.innerWidth < 600) {
          // MM/DD/YY for mobile
          const options = { year: '2-digit', month: '2-digit', day: '2-digit' };
          formattedDate = timestamp.toLocaleDateString('en-US', options);
        } else {
          // full date otherwise
          const options = { year: 'numeric', month: 'long', day: 'numeric' };
          formattedDate = timestamp.toLocaleDateString('en-US', options);
        }

        setLastUpdated(formattedDate);
      } else {
        console.error('No data found in last_update table.');
      }
    } catch (error) {
      console.error('Error fetching metadata:', error);
    }
  };

  useEffect(() => {
    fetchData(currentDate);
    fetchMetadata();
  }, [currentDate]);

  const handlePrevDay = () => {
    const prevDate = new Date(currentDate);
    prevDate.setDate(prevDate.getDate() - 1);
    setCurrentDate(prevDate.toISOString().split('T')[0]);
  };

  const handleNextDay = () => {
    const nextDate = new Date(currentDate);
    nextDate.setDate(nextDate.getDate() + 1);
    setCurrentDate(nextDate.toISOString().split('T')[0]);
  };

  if (loading) {
    return (
      <div className="loader">
        <GridLoader color="#666666" loading={loading} size={25} />
      </div>
    );
  }

  return (
    <div className="games">
      <h1>Games</h1>
      <h2>Projections last updated {lastUpdated}</h2>
      <div className="day-navigator">
        <button onClick={handlePrevDay}>{'<'}</button>
        <span className="date-display">
          {new Date(new Date(currentDate).setDate(new Date(currentDate).getDate() + 1)).toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          })}
        </span>
        <button onClick={handleNextDay}>{'>'}</button>
      </div>
      {games.length === 0 ? (
        <div className="no-games-message">
          <img src={noGamesImage} alt="No games available" />
          <p>No games available on this date.</p>
        </div>
      ) : (
        <div className="games-container">
          {games.map((game) => (
            <div className="game" key={game.id}>
              <div className="game-head">
                <p className="matchup">{game.home_name} @ {game.visitor_name}</p>
                <p className="time">{game.time_str}</p>
              </div>
              <div className="column-left">
                <img src={game.home_logo} alt={game.home_name} />
                <p className="probability">{(game.home_prob * 100).toFixed(1)}%</p>
                <p className="record">{game.home_record} ({game.home_rank})</p>
              </div>
              <div className="column-right">
                <img src={game.visitor_logo} alt={game.visitor_name} />
                <p className="probability">{(game.visitor_prob * 100).toFixed(1)}%</p>
                <p className="record">{game.visitor_record} ({game.visitor_rank})</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Games;