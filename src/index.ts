import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import './styles/styles.scss';
import { createRoot } from 'react-dom/client';

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}