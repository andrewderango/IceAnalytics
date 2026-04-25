import React, { useState, useEffect, useMemo } from 'react';
import { useHistory } from 'react-router-dom';
import { useTable, useSortBy } from 'react-table';
import supabase from '../supabaseClient';
import '../styles/Teams.scss';
import { useSiteConfig } from '../context/SiteConfigContext';
import noGamesImage from '../assets/images/404.png';
import PageStatePanel from '../components/PageStatePanel';
import { getUpcomingProjectionSeasonLabel } from '../utils/seasonLabels';

function Teams() {
  const { offseason, loading: configLoading } = useSiteConfig();
  const [data, setData] = useState([]);
  const [sortBy, setSortBy] = useState({ id: null, desc: false });
  const projectionSeason = getUpcomingProjectionSeasonLabel();
  const history = useHistory();

  useEffect(() => {
    if (configLoading || offseason) return;
    const fetchData = async () => {
      const { data: teams, error } = await supabase
        .from('team_projections')
        .select('*');

      if (error) {
        console.error('Error fetching data:', error);
      } else {
        setData(teams);
      }
    };
    fetchData();
  }, [configLoading, offseason]);

  const columns = useMemo(
    () => [
      {
        Header: 'Logo',
        accessor: 'logo',
        Cell: ({ value }) => <img src={value} alt="logo" className="team-logo" />,
        sticky: 'left',
      },
      {
        Header: 'Team',
        accessor: 'team',
        Cell: ({ cell: { value } }) => (
          <div className="team-name-cell" style={{ minWidth: '150px' }}>
            {value}
          </div>
        ),
      },
      {
        Header: 'Wins',
        accessor: 'wins',
        Cell: ({ cell: { value } }) => Math.round(value),
      },
      {
        Header: 'Losses',
        accessor: 'losses',
        Cell: ({ cell: { value } }) => Math.round(value),
      },
      {
        Header: 'OTL',
        accessor: 'otl',
        Cell: ({ cell: { value } }) => Math.round(value),
      },
      {
        Header: 'Points',
        accessor: 'points',
        Cell: ({ cell: { value } }) => Math.round(value),
      },
      {
        Header: 'GF',
        accessor: 'goals_for',
        Cell: ({ cell: { value } }) => Math.round(value),
      },
      {
        Header: 'GA',
        accessor: 'goals_against',
        Cell: ({ cell: { value } }) => Math.round(value),
      },
      {
        Header: 'Playoffs',
        accessor: 'playoff_prob',
        sortType: (rowA, rowB, columnId, desc) => {
          const a = parseFloat(rowA.original[columnId]);
          const b = parseFloat(rowB.original[columnId]);
          return desc ? b - a : a - b;
        },
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === sortBy.id;
          const color = `rgba(138, 125, 91, ${parseFloat(value) * 0.9 + 0.1})`;
          return (
            <div 
              className={isSelected ? 'selected-column' : ''} 
              style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
            >
              {(parseFloat(value) * 100).toFixed(1)}%
            </div>
          );
        },
      },
      {
        Header: "Presidents' Trophy",
        accessor: 'presidents_trophy_prob',
        sortType: (rowA, rowB, columnId, desc) => {
          const a = parseFloat(rowA.original[columnId]);
          const b = parseFloat(rowB.original[columnId]);
          return desc ? b - a : a - b;
        },
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === sortBy.id;
          const color = `rgba(138, 125, 91, ${parseFloat(value) * 0.9 + 0.1})`;
          return (
            <div 
              className={isSelected ? 'selected-column' : ''} 
              style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
            >
              {(parseFloat(value) * 100).toFixed(1)}%
            </div>
          );
        },
      },
      // {
      //   Header: 'Stanley Cup',
      //   accessor: 'stanley_cup_prob',
      //   sortType: (rowA, rowB, columnId, desc) => {
      //     const a = parseFloat(rowA.original[columnId]);
      //     const b = parseFloat(rowB.original[columnId]);
      //     return desc ? b - a : a - b;
      //   },
      //   Cell: ({ cell: { value }, column: { id } }) => {
      //     const isSelected = id === sortBy.id;
      //     const color = `rgba(138, 125, 91, ${parseFloat(value) * 0.9 + 0.1})`;
      //     return (
      //       <div 
      //         className={isSelected ? 'selected-column' : ''} 
      //         style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
      //       >
      //         {(parseFloat(value) * 100).toFixed(1)}%
      //       </div>
      //     );
      //   },
    ],
    [sortBy]
  );

  const handleSort = (columnId) => {
    setSortBy(prev => {
      if (prev.id === columnId) {
        return { id: null, desc: false };
      } else {
        if (columnId === 'logo' || columnId === 'team' || columnId === 'goals_against') {
          return { id: columnId, desc: false };
        } else {
          return { id: columnId, desc: true };
        }
      }
    });
  };

  const sortedData = React.useMemo(() => {
    const { id, desc } = sortBy;
    if (!id) return data;

    const sorted = [...data].sort((a, b) => {
      let aValue = a[id];
      let bValue = b[id];

      if (id === 'playoff_prob' || id === 'presidents_trophy_prob' || id === 'stanley_cup_prob') {
        aValue = parseFloat(aValue);
        bValue = parseFloat(bValue);
      }

      if (aValue < bValue) return desc ? 1 : -1;
      if (aValue > bValue) return desc ? -1 : 1;
      return 0;
    });

    return sorted;
  }, [data, sortBy]);

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({ columns, data: sortedData }, useSortBy);

  if (configLoading) return null;
  if (offseason) {
    return (
      <PageStatePanel
        wrapperClassName="teams"
        title="Teams"
        badge="Offseason"
        heading="Team projections are between seasons"
        message={`It is currently the offseason. Check back in July when the NHL schedule is released to view ${projectionSeason} projections.`}
        seasonLabel={projectionSeason}
        imageSrc={noGamesImage}
        imageAlt="Offseason"
      />
    );
  }

  return (
    <div className="teams">
      <h1>Teams</h1>
      <div className="table-container">
        <table {...getTableProps()} style={{ color: 'white', backgroundColor: '#333' }}>
          <thead>
            {headerGroups.map(headerGroup => (
              <tr {...headerGroup.getHeaderGroupProps()}>
                {headerGroup.headers.map(column => (
                  <th
                    {...column.getHeaderProps({
                      style: {
                        cursor: 'pointer',
                        backgroundColor: sortBy.id === column.id ? 'rgba(218, 165, 32, 0.5)' : undefined,
                        position: column.sticky ? 'sticky' : undefined,
                        left: column.sticky ? 0 : undefined,
                        zIndex: 1,
                      },
                      onClick: () => handleSort(column.id),
                    })}
                  >
                    {column.render('Header')}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody {...getTableBodyProps()}>
            {rows.map(row => {
              prepareRow(row);
              // determine a team identifier to use in URL - strict: only use `abbrev` (e.g., TBL, CAR)
              // do not fall back to slugified names; if abbrev is missing, don't link
              const makeTeamId = (orig) => {
                if (!orig) return null;
                return orig.abbrev ? String(orig.abbrev) : null;
              };
              const teamId = makeTeamId(row.original);
              return (
                <tr
                  {...row.getRowProps({
                    onClick: () => {
                      if (teamId) history.push(`/team/${encodeURIComponent(teamId)}`);
                    },
                    style: { cursor: teamId ? 'pointer' : 'default' }
                  })}
                >
                  {row.cells.map(cell => (
                    <td
                      {...cell.getCellProps({
                        style: {
                          backgroundColor: sortBy.id === cell.column.id ? 'rgba(218, 165, 32, 0.15)' : undefined,
                          position: cell.column.sticky ? 'sticky' : undefined,
                          left: cell.column.sticky ? 0 : undefined,
                          zIndex: 1,
                        },
                      })}
                    >
                      {cell.render('Cell')}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Teams;
