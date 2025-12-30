import React, { useState, useEffect } from 'react';
import supabase from '../supabaseClient';
import '../styles/Footer.scss';
import logo from '../assets/images/logo.svg';

function Footer() {
  const currentYear = new Date().getFullYear();
  const [lastUpdated, setLastUpdated] = useState('');

  useEffect(() => {
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
          const dateOptions = { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' };
          const timeOptions = { hour: '2-digit', minute: '2-digit', timeZone: 'UTC', hour12: false };
          const formattedDate = timestamp.toLocaleDateString('en-US', dateOptions);
          const formattedTime = timestamp.toLocaleTimeString('en-US', timeOptions);
          setLastUpdated(`${formattedDate} at ${formattedTime} UTC`);
        }
      } catch (error) {
        console.error('Error fetching metadata:', error);
      }
    };
    fetchMetadata();
  }, []);
  
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-content">
          <div className="footer-section footer-brand">
            <div className="footer-logo-container">
              <img src={logo} alt="IceAnalytics Logo" className="footer-logo-image" />
              <h3 className="footer-logo">
                <span className="footer-accent">ICE</span>ANALYTICS
              </h3>
            </div>
            <p className="footer-tagline">
              Advanced NHL projections and analytics through simulation-based quantitative modeling.
            </p>
            <div className="footer-stats">
              {lastUpdated && (
                <div className="stat-item">
                  <span className="stat-label">Last Engine Run</span>
                  <span className="stat-value">{lastUpdated}</span>
                </div>
              )}
            </div>
          </div>
          
          <div className="footer-section">
            <h4>Quick Links</h4>
            <ul>
              <li><a href="/home">Home</a></li>
              <li><a href="/games">Games</a></li>
              <li><a href="/players">Players</a></li>
              <li><a href="/teams">Teams</a></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Resources</h4>
            <ul>
              <li><a href="/about">About</a></li>
              <li><a href="https://github.com/andrewderango/IceAnalytics" target="_blank" rel="noopener noreferrer">GitHub</a></li>
              <li><a href="https://github.com/andrewderango/IceAnalytics/releases" target="_blank" rel="noopener noreferrer">Releases</a></li>
              <li><a href="https://github.com/andrewderango/IceAnalytics/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">GPL License</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h4>Connect</h4>
            <ul>
              <li><a href="mailto:hello@iceanalytics.ca">Contact Us</a></li>
              <li><a href="https://www.linkedin.com/in/andrewderango" target="_blank" rel="noopener noreferrer">Developer</a></li>
              <li><a href="https://github.com/andrewderango/IceAnalytics/issues" target="_blank" rel="noopener noreferrer">Report Issues</a></li>
            </ul>
          </div>
        </div>
        
        <div className="footer-bottom">
          <div className="footer-copyright">
            <p>&copy; {currentYear} IceAnalytics. All rights reserved.</p>
          </div>
          <div className="footer-meta">
            <p>IceAnalytics Projection Engine v1.3.1</p>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;