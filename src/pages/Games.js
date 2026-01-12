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
      <div className="games-hero">
        <div className="games-hero-content">
          <div className="hero-badge">NHL GAME PROJECTIONS</div>
          <h1 className="hero-title">Games</h1>
        </div>
      </div>

      <div className="games-content">
        <div className="date-picker-section">
          <div className="day-navigator">
            <button onClick={handlePrevDay} disabled={!minDate || currentDate <= minDate} className="nav-btn">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M12.5 15L7.5 10L12.5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
            <div className="date-picker-wrapper">
              <input
                type="date"
                value={currentDate}
                onChange={(e) => setCurrentDate(e.target.value)}
                min={minDate}
                max={maxDate}
                className="date-picker"
              />
              <span className="date-display">
                {new Date(currentDate + 'T00:00:00').toLocaleDateString('en-US', {
                  weekday: 'long',
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                })}
              </span>
            </div>
            <button onClick={handleNextDay} disabled={!maxDate || currentDate >= maxDate} className="nav-btn">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M7.5 15L12.5 10L7.5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
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
                <div className="game-header">
                  <span className="matchup">
                    <span className="full-names">{game.visitor_name} @ {game.home_name}</span>
                    <span className="abbr-names">{game.visitor_abbr} @ {game.home_abbr}</span>
                  </span>
                  <span className="time">
                    {game.home_prob === 1 || game.home_prob === 0 
                      ? (game.overtime_prob === 1 ? 'Final/OT' : 'Final') 
                      : game.time_str}
                  </span>
                </div>
                <div className="game-content">
                  <div className="team-section visitor">
                    <img src={game.visitor_logo} alt={game.visitor_name} className="team-logo" />
                    <div className="team-info">
                      <p className="probability">
                        {game.visitor_prob === 1 || game.visitor_prob === 0 
                          ? game.visitor_score 
                          : `${(game.visitor_prob * 100).toFixed(1)}%`}
                      </p>
                      <p className="record">{game.visitor_record} ({game.visitor_rank})</p>
                    </div>
                  </div>
                  <div className="vs-divider">
                    <span className="vs-text">VS</span>
                    <div className="divider-line"></div>
                  </div>
                  <div className="team-section home">
                    <img src={game.home_logo} alt={game.home_name} className="team-logo" />
                    <div className="team-info">
                      <p className="probability">
                        {game.home_prob === 1 || game.home_prob === 0 
                          ? game.home_score 
                          : `${(game.home_prob * 100).toFixed(1)}%`}
                      </p>
                      <p className="record">{game.home_record} ({game.home_rank})</p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default Games;