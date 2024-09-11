import React from 'react';
import '../styles/Footer.scss';

function Footer() {
  return (
    <footer className="footer">
      <p>&copy; {new Date().getFullYear()} IceAnalytics</p>
      <p className="small-text">The source code for this project is released under the GNU General Public License v3.0 (GPL v3), a widely used open-source license that ensures the freedom to use, modify, and distribute the software. This license promotes collaboration and innovation by allowing anyone to access and contribute to the codebase. By releasing the source code under the GPL v3, the aim is to foster transparency, encourage community involvement, and support the principles of free software. For more information on the license terms and conditions, please refer to the full text available <a href="https://www.gnu.org/licenses/gpl-3.0.html">here</a>. To explore and contribute to the source code, visit the GitHub repository <a href="https://github.com/andrewderango/IceAnalytics">here</a>.</p>
      <p><a href="https://github.com/andrewderango/IceAnalytics/blob/main/LICENSE">License</a> | <a href="https://github.com/andrewderango/IceAnalytics">Source Code</a> | <a href="https://github.com/andrewderango/IceAnalytics/releases">Releases</a> |  <a href="/about">About</a></p>
    </footer>
  );
}

export default Footer;