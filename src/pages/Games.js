import React, { useState, useEffect } from 'react';
import '../styles/Games.scss';
import { GridLoader } from 'react-spinners';
import { createClient } from '@supabase/supabase-js';

function Games() {
  const supabaseUrl = process.env.REACT_APP_SUPABASE_PROJ_URL;
  const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
  const supabase = createClient(supabaseUrl, supabaseAnonKey);
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);

  // get the current date in EST
  const estDate = new Date().toLocaleString('en-US', { timeZone: 'America/New_York' });
  const [month, day, year] = estDate.split(',')[0].split('/');
  const date = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;

  useEffect(() => {
    const fetchData = async () => {
      const { data: games, error } = await supabase
        .from('game_projections')
        .select('*')
        .eq('date', date);
      
      if (error) {
        console.error('Error fetching data:', error);
      } else {
        // console.log('Fetched data:', games);
        setGames(games);
        setLoading(false);
      }
    };

    fetchData();
  }, [date]);

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
        <h2 className="date">
            {new Date(new Date(date).setDate(new Date(date).getDate() + 1)).toLocaleDateString('en-US', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
            })}
        </h2>
        <div className="games-container">
            {games.map(game => (
                <div className="game" key={game.id}>
                    <div className="game-head">
                      <p className="matchup">{game.visitor_name} @ {game.home_name}</p>
                      <p className="time">{game.time_str}</p>
                    </div>
                    <div className="column-left">
                        <img src={game.visitor_logo} alt={game.visitor_name} />
                        <p className="probability">{(game.visitor_prob*100).toFixed(1)}%</p>
                        {/* <p className="projected-goals">{game.team2_projectedGoals.toFixed(2)} Goals</p> */}
                        <p className="record">{game.visitor_record} ({game.visitor_rank})</p>
                    </div>
                    <div className="column-right">
                        <img src={game.home_logo} alt={game.home_name} />
                        <p className="probability">{(game.home_prob*100).toFixed(1)}%</p>
                        {/* <p className="projected-goals">{game.team1_projectedGoals.toFixed(2)} Goals</p> */}
                        <p className="record">{game.home_record} ({game.home_rank})</p>
                    </div>
                </div>
            ))}
        </div>
    </div>
  );
}

export default Games;