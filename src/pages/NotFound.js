import React from 'react';
import { Link } from 'react-router-dom';
import notFound from '../assets/images/404.png';
import '../styles/NotFound.scss';
import PageStatePanel from '../components/PageStatePanel';
import { getUpcomingProjectionSeasonLabel } from '../utils/seasonLabels';

function NotFound() {
  const projectionSeason = getUpcomingProjectionSeasonLabel();

  return (
    <PageStatePanel
      wrapperClassName="not-found"
      title="404"
      badge="Page Not Found"
      heading="This page does not exist"
      message={`The page you requested could not be found. Use the links below to return to the main sections and follow ${projectionSeason} projections when they go live in July.`}
      imageSrc={notFound}
      imageAlt="404"
      variant="not-found"
      actions={(
        <>
          <Link to="/home" className="state-panel__action-link">Home</Link>
          <Link to="/games" className="state-panel__action-link">Games</Link>
          <Link to="/players" className="state-panel__action-link">Players</Link>
          <Link to="/teams" className="state-panel__action-link">Teams</Link>
        </>
      )}
    />
  );
}

export default NotFound;