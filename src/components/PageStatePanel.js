import React from 'react';
import '../styles/PageStatePanel.scss';

function PageStatePanel({
  wrapperClassName = '',
  title,
  badge,
  heading,
  message,
  imageSrc,
  imageAlt,
  actions,
  variant = 'offseason',
}) {
  const wrapperClasses = [wrapperClassName, 'state-page'].filter(Boolean).join(' ');
  const panelClasses = ['state-panel', `state-panel--${variant}`].filter(Boolean).join(' ');

  return (
    <div className={wrapperClasses}>
      {title && <h1>{title}</h1>}
      <section className={panelClasses} role="status" aria-live="polite">
        <div className="state-panel__content">
          {badge && <p className="state-panel__badge">{badge}</p>}
          {heading && <h2>{heading}</h2>}
          {message && <p className="state-panel__description">{message}</p>}
          {actions && <div className="state-panel__actions">{actions}</div>}
        </div>
        {imageSrc && (
          <div className="state-panel__visual" aria-hidden="true">
            <img src={imageSrc} alt={imageAlt} className="state-panel__image" />
          </div>
        )}
      </section>
    </div>
  );
}

export default PageStatePanel;
