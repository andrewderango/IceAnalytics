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

  useEffect(() => {
    const fetchData = async () => {
      const { data: games, error } = await supabase
        .from('game-projections')
        .select('*');
      
      if (error) {
        console.error('Error fetching data:', error);
      } else {
        console.log('Fetched data:', games);
        setGames(games);
        setLoading(false);
      }
    };
  
    fetchData();
  }, []);

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
            {new Date().toLocaleDateString('en-US', { 
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
                      <p className="matchup">{game.team1_name} @ {game.team2_name}</p>
                      <p className="time">{game.time}</p>
                    </div>
                    <div className="column-left">
                        <img src={game.team1_logo} alt={game.team1_name} />
                        <p className="probability">{(game.team1_probability*100).toFixed(1)}%</p>
                        {/* <p className="projected-goals">{game.team1_projectedGoals.toFixed(2)} Goals</p> */}
                        <p className="record">{game.team1_record}</p>
                    </div>
                    <div className="column-right">
                        <img src={game.team2_logo} alt={game.team2_name} />
                        <p className="probability">{(game.team2_probability*100).toFixed(1)}%</p>
                        {/* <p className="projected-goals">{game.team2_projectedGoals.toFixed(2)} Goals</p> */}
                        <p className="record">{game.team2_record}</p>
                    </div>
                </div>
            ))}
        </div>
    </div>
  );
}

export default Games;