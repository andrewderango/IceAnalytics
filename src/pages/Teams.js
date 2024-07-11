import React, { useState, useEffect } from 'react';
import { useTable } from 'react-table';
import { createClient } from '@supabase/supabase-js';
import '../styles/Teams.scss';

function Teams() {
  const supabaseUrl = process.env.REACT_APP_SUPABASE_PROJ_URL;
  const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
  const supabase = createClient(supabaseUrl, supabaseAnonKey);
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const { data: teams, error } = await supabase
        .from('team-projections')
        .select('*');
      
      if (error) {
        console.error('Error fetching data:', error);
      } else {
        console.log('Fetched data:', teams);
        setData(teams);
      }
    };
  
    fetchData();
  }, []);
  const [selectedColumn, setSelectedColumn] = useState(null);

  const columns = React.useMemo(
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
      // {
      //   Header: 'Playoffs',
      //   accessor: 'playoff_prob',
      //   Cell: ({ cell: { value }, column: { id } }) => {
      //     const isSelected = id === selectedColumn;
      //     const color = `rgba(138, 125, 91, ${value*0.9 + 0.1})`;
      //     return (
      //       <div 
      //         className={isSelected ? 'selected-column' : ''} 
      //         style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
      //       >
      //         {(value*100).toFixed(1)}%
      //       </div>
      //     );
      //   },
      // },
      // {
      //   Header: "Presidents' Trophy",
      //   accessor: 'presidents_trophy_prob',
      //   Cell: ({ cell: { value }, column: { id } }) => {
      //     const isSelected = id === selectedColumn;
      //     const color = `rgba(138, 125, 91, ${value*0.9 + 0.1})`;
      //     return (
      //       <div 
      //         className={isSelected ? 'selected-column' : ''} 
      //         style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
      //       >
      //         {(value*100).toFixed(1)}%
      //       </div>
      //     );
      //   },
      // },
      // {
      //   Header: 'Stanley Cup',
      //   accessor: 'stanley_cup_prob',
      //   Cell: ({ cell: { value }, column: { id } }) => {
      //     const isSelected = id === selectedColumn;
      //     const color = `rgba(138, 125, 91, ${value*0.9 + 0.1})`;
      //     return (
      //       <div 
      //         className={isSelected ? 'selected-column' : ''} 
      //         style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
      //       >
      //         {(value*100).toFixed(1)}%
      //       </div>
      //     );
      //   },
      // },
    ],
    [selectedColumn]
  );

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({ columns, data });

  return (
    <div className="teams">
      <h1>Teams</h1>
      <h2>Last updated May 15, 2024</h2>
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
                      backgroundColor: selectedColumn === column.id ? 'rgba(218, 165, 32, 0.5)' : undefined,
                      position: column.sticky ? 'sticky' : undefined,
                      left: column.sticky ? 0 : undefined,
                      zIndex: 1,
                    },
                    onClick: () => setSelectedColumn(prev => prev === column.id ? null : column.id),
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
            return (
              <tr {...row.getRowProps()}>
              {row.cells.map(cell => (
                <td
                  {...cell.getCellProps({
                    style: {
                      cursor: 'pointer',
                      backgroundColor: selectedColumn === cell.column.id ? 'rgba(218, 165, 32, 0.15)' : undefined,
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