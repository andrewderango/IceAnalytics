import React, { useState, useEffect } from 'react';
import supabase from '../supabaseClient';
import '../styles/Footer.scss';

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
          const options = { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' };
          const formattedDate = timestamp.toLocaleDateString('en-US', options);
          setLastUpdated(formattedDate);
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
            <h3 className="footer-logo">IceAnalytics</h3>
            <p className="footer-tagline">Advanced NHL projections and analytics powered by a predictive modeling engine</p>
            <p className="footer-contact">Contact: <a href="mailto:hello@iceanalytics.ca">hello@iceanalytics.ca</a></p>
          </div>
          
          <div className="footer-section footer-links">
            <h4>Navigation</h4>
            <ul>
              <li><a href="/about">About</a></li>
              <li><a href="https://github.com/andrewderango/IceAnalytics" target="_blank" rel="noopener noreferrer">GitHub Repository</a></li>
              <li><a href="https://github.com/andrewderango/IceAnalytics/releases" target="_blank" rel="noopener noreferrer">Releases</a></li>
            </ul>
          </div>
          
          <div className="footer-section footer-links">
            <h4>Legal</h4>
            <ul>
              <li><a href="https://github.com/andrewderango/IceAnalytics/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">GPL v3 License</a></li>
              <li><a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noopener noreferrer">License Terms</a></li>
              <li><a href="https://github.com/andrewderango/IceAnalytics/issues" target="_blank" rel="noopener noreferrer">Report Issues</a></li>
            </ul>
          </div>
        </div>
        
        <div className="footer-bottom">
          <div className="footer-copyright">
            <p>&copy; {currentYear} IceAnalytics. All rights reserved.</p>
          </div>
          <div className="footer-meta">
            {lastUpdated && <p className="footer-updated">Projections last updated {lastUpdated}</p>}
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;