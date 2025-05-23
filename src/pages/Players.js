import React, { useState, useEffect } from 'react';
import { useTable, usePagination, useSortBy } from 'react-table';
import { useHistory, Link } from 'react-router-dom';
import supabase from '../supabaseClient';
import '../styles/Players.scss';
import { offseason } from '../config/settings';
import noGamesImage from '../assets/images/404.png';

function Players() {
  const [data, setData] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [teamFilter, setTeamFilter] = useState('');
  const [posFilter, setPosFilter] = useState('');
  const [selectedColumn, setSelectedColumn] = useState(null);
  const [sortBy, setSortByState] = useState([
    { id: 'points', desc: true },
    { id: 'goals', desc: true }
  ]);
  const [lastUpdated, setLastUpdated] = useState('');
  const history = useHistory();

  if (offseason) {
    return (
      <div className="players offseason-message">
        <h1>Players</h1>
        <p>It is currently the offseason. Check back in July when the NHL schedule is released to view 2025-26 projections!</p>
        <img src={noGamesImage} alt="Offseason" className="offseason-image" />
        <div style={{ height: '55vh' }}></div>
      </div>
    );
  }

  const columns = React.useMemo(
    () => [
      {
        Header: 'Player',
        accessor: 'player',
        Cell: ({ row }) => (
          <Link to={`/player/${row.original.player_id}`}>
            {row.original.player}
          </Link>
        ),
      },
      {
        Header: 'Team',
        accessor: 'logo',
        Cell: ({ value }) => <img src={value} alt="logo" className="team-logo" />,
        sticky: 'left',
      },
      {
        Header: 'Position',
        accessor: 'position',
      },
      {
        Header: 'Games',
        accessor: 'games',
        Cell: ({ value }) => Math.round(value),
      },
      {
        Header: 'Goals',
        accessor: 'goals',
        Cell: ({ value }) => Math.round(value),
      },
      {
        Header: 'Assists',
        accessor: 'assists',
        Cell: ({ value }) => Math.round(value),
      },
      {
        Header: 'Points',
        accessor: 'points',
        Cell: ({ value }) => Math.round(value),
      },
      {
        Header: 'Art Ross',
        accessor: 'art_ross',
        sortType: (rowA, rowB, columnId) => {
          const a = parseFloat(rowA.original[columnId]);
          const b = parseFloat(rowB.original[columnId]);
          return b - a;
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
        Header: 'Rocket',
        accessor: 'rocket',
        sortType: (rowA, rowB, columnId) => {
          const a = parseFloat(rowA.original[columnId]);
          const b = parseFloat(rowB.original[columnId]);
          return b - a;
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
    ],
    [selectedColumn]
  );

  useEffect(() => {
    const fetchData = async () => {
      const { data: players, error } = await supabase
        .from('player_projections')
        .select('*');
      if (error) {
        console.error('Error fetching data:', error);
      } else {
        console.log('Fetched data:', players);
        setData(players);
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
    setSortBy,
  } = useTable(
    {
      columns,
      data: filteredData,
      initialState: {
        pageIndex: 0,
        pageSize: 25,
        sortBy: [
          { id: 'points', desc: true },
          { id: 'goals', desc: true }
        ],
      },
    },
    useSortBy,
    usePagination
  );

  const teams = [...new Set(data.map(player => player.team))].sort();
  const positions = [...new Set(data.map(player => player.position))].sort();

  const handleColumnClick = (column) => {
    const isDescending = ['games', 'goals', 'assists', 'points'].includes(column.id);
    if (selectedColumn === column.id) {
      setSelectedColumn(null);
      const defaultSort = [
        { id: 'points', desc: true },
        { id: 'goals', desc: true }
      ];
      setSortBy(defaultSort);
      setSortByState(defaultSort);
    } else {
      setSelectedColumn(column.id);
      const sortConfig = [
        { id: column.id, desc: isDescending },
        { id: 'points', desc: true },
        { id: 'goals', desc: true },
      ];
      setSortBy(sortConfig);
      setSortByState(sortConfig);
    }
  };

  const handleRowClick = (playerId) => {
    history.push(`/player/${playerId}`);
  };

  const startRow = pageIndex * pageSize + 1;
  const endRow = Math.min(startRow + pageSize - 1, filteredData.length);

  return (
    <div className="players">
      <h1>Players</h1>
      <h2>Projections last updated {lastUpdated}</h2>
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
                    {...column.getHeaderProps()}
                    style={{
                      cursor: 'pointer',
                      backgroundColor: selectedColumn === column.id ? 'rgba(218, 165, 32, 0.5)' : undefined,
                      position: column.id === 'player' ? 'sticky' : undefined,
                      left: column.id === 'player' ? 0 : undefined,
                      zIndex: 1,
                    }}
                    onClick={() => handleColumnClick(column)}
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
                <tr {...row.getRowProps()} onClick={() => handleRowClick(row.original.player_id)}>
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
      <div className="pagination-container">
        <div className="pagination">
          <div className="page-nav">
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
          </div>
          <div className="page-size">
            <select
              value={pageSize}
              onChange={e => {
                setPageSize(Number(e.target.value));
              }}
            >
              {[10, 25, 50, 100, 250].map(size => (
                <option key={size} value={size}>
                  Show {size}
                </option>
              ))}
            </select>
          </div>
          <div className="pagination-info">
            Showing {startRow}-{endRow} of {filteredData.length}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Players;