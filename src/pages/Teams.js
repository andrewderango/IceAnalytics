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
        Header: 'Points',
        accessor: 'points',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'GF',
        accessor: 'goals_for',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'GA',
        accessor: 'goals_against',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'Playoffs',
        accessor: 'playoff_prob',
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === selectedColumn;
          const color = `rgba(138, 125, 91, ${value*0.9 + 0.1})`;
          return (
            <div 
              className={isSelected ? 'selected-column' : ''} 
              style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
            >
              {(value*100).toFixed(1)}%
            </div>
          );
        },
      },
      {
        Header: "Presidents' Trophy",
        accessor: 'presidents_trophy_prob',
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === selectedColumn;
          const color = `rgba(138, 125, 91, ${value*0.9 + 0.1})`;
          return (
            <div 
              className={isSelected ? 'selected-column' : ''} 
              style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
            >
              {(value*100).toFixed(1)}%
            </div>
          );
        },
      },
      {
        Header: 'Stanley Cup',
        accessor: 'stanley_cup_prob',
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === selectedColumn;
          const color = `rgba(138, 125, 91, ${value*0.9 + 0.1})`;
          return (
            <div 
              className={isSelected ? 'selected-column' : ''} 
              style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
            >
              {(value*100).toFixed(1)}%
            </div>
          );
        },
      },
    ],
    [selectedColumn]
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