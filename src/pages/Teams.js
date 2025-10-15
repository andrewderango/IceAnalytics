import React, { useState, useEffect, useMemo } from 'react';
import { useHistory } from 'react-router-dom';
import { useTable, useSortBy } from 'react-table';
import supabase from '../supabaseClient';
import '../styles/Teams.scss';
import { offseason } from '../config/settings';
import noGamesImage from '../assets/images/404.png';

function Teams() {
  const [data, setData] = useState([]);
  const [sortBy, setSortBy] = useState({ id: null, desc: false });
  const [lastUpdated, setLastUpdated] = useState('');
  const history = useHistory();

  if (offseason) {
    return (
      <div className="teams offseason-message">
        <h1>Teams</h1>
        <p>It is currently the offseason. Check back in July when the NHL schedule is released to view 2025-26 projections!</p>
        <img src={noGamesImage} alt="Offseason" className="offseason-image" />
      </div>
    );
  }

  useEffect(() => {
    const fetchData = async () => {
      const { data: teams, error } = await supabase
        .from('team_projections')
        .select('*');
      
      if (error) {
        console.error('Error fetching data:', error);
      } else {
        console.log('Fetched data:', teams);
        setData(teams);
      }
    };
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
          let formattedDate;
    
          if (window.innerWidth < 600) {
            // MM/DD/YY for mobile
            const options = { year: '2-digit', month: '2-digit', day: '2-digit' };
            formattedDate = timestamp.toLocaleDateString('en-US', options);
          } else {
            // full date otherwise
            const options = { year: 'numeric', month: 'long', day: 'numeric' };
            formattedDate = timestamp.toLocaleDateString('en-US', options);
          }
    
          setLastUpdated(formattedDate);
        } else {
          console.error('No data found in last_update table.');
        }
      } catch (error) {
        console.error('Error fetching metadata:', error);
      }
    };
    fetchData();
    fetchMetadata();
  }, []);

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
          <div style={{ minWidth: '150px' }}>
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

  return (
    <div className="teams">
      <h1>Teams</h1>
      <h2>Projections last updated {lastUpdated}</h2>
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
