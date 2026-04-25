import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { SiteConfigProvider } from './context/SiteConfigContext';
import './styles/styles.scss';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <SiteConfigProvider>
    <App />
  </SiteConfigProvider>
);