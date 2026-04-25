import React, { createContext, useContext, useEffect, useState } from 'react';
import supabase from '../supabaseClient';

const SiteConfigContext = createContext({
  offseason: false,
  lastUpdatedDatetime: null,
  loading: true,
});

export function SiteConfigProvider({ children }) {
  const [state, setState] = useState({
    offseason: false,
    lastUpdatedDatetime: null,
    loading: true,
  });

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const { data, error } = await supabase
          .from('site_config')
          .select('datetime, offseason')
          .eq('id', 1)
          .single();

        if (error) {
          console.error('Error fetching site_config:', error);
          setState({ offseason: false, lastUpdatedDatetime: null, loading: false });
          return;
        }

        setState({
          offseason: data.offseason,
          lastUpdatedDatetime: data.datetime,
          loading: false,
        });
      } catch (err) {
        console.error('Error fetching site_config:', err);
        setState({ offseason: false, lastUpdatedDatetime: null, loading: false });
      }
    };

    fetchConfig();
  }, []);

  return (
    <SiteConfigContext.Provider value={state}>
      {children}
    </SiteConfigContext.Provider>
  );
}

export function useSiteConfig() {
  return useContext(SiteConfigContext);
}
