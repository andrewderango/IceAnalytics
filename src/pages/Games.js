import React, { useState, useEffect } from 'react';
import '../styles/Games.scss';
import { GridLoader } from 'react-spinners';
import { createClient } from '@supabase/supabase-js';
import noGamesImage from '../assets/images/404.png';
import { offseason } from '../config/settings';

function Games() {
  const supabaseUrl = process.env.REACT_APP_SUPABASE_PROJ_URL;
  const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
  const supabase = createClient(supabaseUrl, supabaseAnonKey);
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState('');
  const [minDate, setMinDate] = useState(null);
  const [maxDate, setMaxDate] = useState(null);
  const [datesLoaded, setDatesLoaded] = useState(false);

  // get the current date in EST
  const estDate = new Date().toLocaleString('en-US', { timeZone: 'America/New_York' });
  const [month, day, year] = estDate.split(',')[0].split('/');
  const today = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
  const [currentDate, setCurrentDate] = useState(today);

  // Show offseason message if no valid dates loaded
  if (offseason || (datesLoaded && (!minDate && !maxDate))) {
    return (
      <div className="games offseason-message">
        <h1>Games</h1>
        <p>It is currently the offseason. Check back in July when the NHL schedule is released to view 2025-26 projections!</p>
        <img src={noGamesImage} alt="Offseason" className="offseason-image" />
      </div>
    );
  }

  // Fetch min/max game dates
  const fetchMinMaxDates = async () => {
    try {
      const { data: minData, error: minError } = await supabase
        .from('game_projections')
        .select('date')
        .order('date', { ascending: true })
        .limit(1);
      const { data: maxData, error: maxError } = await supabase
        .from('game_projections')
        .select('date')
        .order('date', { ascending: false })
        .limit(1);
      if (minError || maxError) {
        console.error('Error fetching min/max dates:', minError || maxError);
        setDatesLoaded(true);
        return;
      }
      if (minData.length > 0 && maxData.length > 0) {
        setMinDate(minData[0].date);
        setMaxDate(maxData[0].date);
        setDatesLoaded(true);
        // Clamp initial date to valid range
        let initial = today;
        if (initial < minData[0].date) initial = minData[0].date;
        if (initial > maxData[0].date) initial = maxData[0].date;
        setCurrentDate(initial);
      } else {
        setDatesLoaded(true);
      }
    } catch (err) {
      console.error('Error fetching min/max dates:', err);
      setDatesLoaded(true);
    }
  };

  // Fetch games for a date
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

  // Fetch min/max dates on mount
  useEffect(() => {
    fetchMinMaxDates();
    // eslint-disable-next-line
  }, []);

  // Fetch games and metadata when currentDate changes and min/max loaded
  useEffect(() => {
    if (!datesLoaded || !minDate || !maxDate) return;
    // Clamp currentDate to valid range
    let clampedDate = currentDate;
    if (currentDate < minDate) clampedDate = minDate;
    if (currentDate > maxDate) clampedDate = maxDate;
    if (clampedDate !== currentDate) setCurrentDate(clampedDate);
    else {
      setLoading(true);
      fetchData(clampedDate);
      fetchMetadata();
    }
    // eslint-disable-next-line
  }, [currentDate, datesLoaded, minDate, maxDate]);

  const handlePrevDay = () => {
    if (!minDate) return;
    const prevDate = new Date(currentDate);
    prevDate.setDate(prevDate.getDate() - 1);
    const prevStr = prevDate.toISOString().split('T')[0];
    if (prevStr >= minDate) setCurrentDate(prevStr);
  };

  const handleNextDay = () => {
    if (!maxDate) return;
    const nextDate = new Date(currentDate);
    nextDate.setDate(nextDate.getDate() + 1);
    const nextStr = nextDate.toISOString().split('T')[0];
    if (nextStr <= maxDate) setCurrentDate(nextStr);
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
        <button onClick={handlePrevDay} disabled={!minDate || currentDate <= minDate}>{'<'}</button>
        <span className="date-display">
          {new Date(currentDate + 'T00:00:00').toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          })}
        </span>
        <button onClick={handleNextDay} disabled={!maxDate || currentDate >= maxDate}>{'>'}</button>
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
                <p className="matchup">{game.visitor_name} @ {game.home_name}</p>
                <p className="time">
                  {game.home_prob === 1 || game.home_prob === 0 ? (game.overtime_prob === 1 ? 'Final - OT' : 'Final - Regulation') : game.time_str}
                 </p>
              </div>
              <div className="column-left">
                <img src={game.visitor_logo} alt={game.visitor_name} />
                <p className="probability">
                  {game.visitor_prob === 1 || game.visitor_prob === 0 ? game.visitor_score : `${(game.visitor_prob * 100).toFixed(1)}%`}
                </p>
                <p className="record">{game.visitor_record} ({game.visitor_rank})</p>
              </div>
              <div className="column-right">
                <img src={game.home_logo} alt={game.home_name} />
                <p className="probability">
                  {game.home_prob === 1 || game.home_prob === 0 ? game.home_score : `${(game.home_prob * 100).toFixed(1)}%`}
                </p>
                <p className="record">{game.home_record} ({game.home_rank})</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Games;