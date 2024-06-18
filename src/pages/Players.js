import React, { useState, useEffect } from 'react';
import { useTable } from 'react-table';
import { createClient } from '@supabase/supabase-js'

function Players() {
  const supabaseUrl = process.env.REACT_APP_SUPABASE_PROJ_URL;
  const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
  const supabase = createClient(supabaseUrl, supabaseAnonKey);
  const [data, setData] = useState([]);

  const columns = React.useMemo(
    () => [
      {
        Header: 'Name',
        accessor: 'name',
      },
      {
        Header: 'Team',
        accessor: 'team',
      },
      {
        Header: 'Position',
        accessor: 'position',
      },
      {
        Header: 'Games',
        accessor: 'games',
      },
      {
        Header: 'Goals',
        accessor: 'goals',
      },
      {
        Header: 'Assists',
        accessor: 'assists',
      },
      {
        Header: 'Points',
        accessor: 'points',
      },
    ],
    []
  );

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({ columns, data });

  useEffect(() => {
    const fetchData = async () => {
      const { data: players, error } = await supabase
        .from('player-projections')
        .select('*');
      
      if (error) {
        console.error('Error fetching data:', error);
      } else {
        console.log('Fetched data:', players);
        setData(players);
      }
    };
  
    fetchData();
  }, []);

  return (
    <div>
      <h1>Players</h1>
      <table {...getTableProps()}>
        <thead>
          {headerGroups.map(headerGroup => (
            <tr {...headerGroup.getHeaderGroupProps()}>
              {headerGroup.headers.map(column => (
                <th {...column.getHeaderProps()}>{column.render('Header')}</th>
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
                  <td {...cell.getCellProps()}>{cell.render('Cell')}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default Players;