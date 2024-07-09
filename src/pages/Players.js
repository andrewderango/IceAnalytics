import React, { useState, useEffect } from 'react';
import { useTable, usePagination } from 'react-table';
import { createClient } from '@supabase/supabase-js';
import '../styles/Players.scss';

function Players() {
  const supabaseUrl = process.env.REACT_APP_SUPABASE_PROJ_URL;
  const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
  const supabase = createClient(supabaseUrl, supabaseAnonKey);
  const [data, setData] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [teamFilter, setTeamFilter] = useState('');
  const [posFilter, setPosFilter] = useState('');
  const [selectedColumn, setSelectedColumn] = useState(null);

  const columns = React.useMemo(
    () => [
      {
        Header: 'Player',
        accessor: 'player',
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
    [selectedColumn]
  );

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

  const filteredData = React.useMemo(() => {
    return data.filter(row => {
      return row.player.toLowerCase().includes(searchTerm.toLowerCase()) &&
        (teamFilter ? row.team === teamFilter : true) &&
        (posFilter ? row.position === posFilter : true);
    });
  }, [data, searchTerm, teamFilter, posFilter]);

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    page,
    prepareRow,
    canPreviousPage,
    canNextPage,
    pageOptions,
    pageCount,
    gotoPage,
    nextPage,
    previousPage,
    setPageSize,
    state: { pageIndex, pageSize },
  } = useTable(
    { columns, data: filteredData, initialState: { pageIndex: 0, pageSize: 10 } },
    usePagination
  );

  const teams = [...new Set(data.map(player => player.team))].sort();
  const positions = [...new Set(data.map(player => player.position))].sort();

  return (
    <div className="players">
      <h1>Players</h1>
      <h2>Last updated May 15, 2024</h2>
      <div className="filter-container">
        <div className="select-container">
          <select value={teamFilter} onChange={e => setTeamFilter(e.target.value)}>
            <option value="">All Teams</option>
            {teams.map(team => (
              <option key={team} value={team}>{team}</option>
            ))}
          </select>
          <select value={posFilter} onChange={e => setPosFilter(e.target.value)}>
            <option value="">All Positions</option>
            {positions.map(pos => (
              <option key={pos} value={pos}>{pos}</option>
            ))}
          </select>
        </div>
        <input
          type="text"
          value={searchTerm}
          onChange={e => setSearchTerm(e.target.value)}
          placeholder="Search Players"
        />
      </div>
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
                        position: column.id === 'player' ? 'sticky' : undefined,
                        left: column.id === 'player' ? 0 : undefined,
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
            {page.map(row => {
              prepareRow(row);
              return (
                <tr {...row.getRowProps()}>
                  {row.cells.map(cell => (
                    <td
                      {...cell.getCellProps({
                        style: {
                          cursor: 'pointer',
                          backgroundColor: selectedColumn === cell.column.id ? 'rgba(218, 165, 32, 0.15)' : undefined,
                          position: cell.column.id === 'player' ? 'sticky' : undefined,
                          left: cell.column.id === 'player' ? 0 : undefined,
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
      <div className="pagination">
        <button onClick={() => gotoPage(0)} disabled={!canPreviousPage}>
          {'<<'}
        </button>
        <button onClick={() => previousPage()} disabled={!canPreviousPage}>
          {'<'}
        </button>
        <span>
          Page{' '}
          <strong>
            {pageIndex + 1} of {pageOptions.length}
          </strong>{' '}
        </span>
        <button onClick={() => nextPage()} disabled={!canNextPage}>
          {'>'}
        </button>
        <button onClick={() => gotoPage(pageCount - 1)} disabled={!canNextPage}>
          {'>>'}
        </button>
        <select
          value={pageSize}
          onChange={e => {
            setPageSize(Number(e.target.value));
          }}
        >
          {[10, 20, 30, 40, 50].map(pageSize => (
            <option key={pageSize} value={pageSize}>
              Show {pageSize}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

export default Players;
