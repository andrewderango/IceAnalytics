import React, { useState } from 'react';
import { useTable } from 'react-table';
// import { useTable, useSortBy } from 'react-table';
import '../styles/Players.scss';

function Players() {
  const [searchTerm, setSearchTerm] = React.useState('');
  const [teamFilter, setTeamFilter] = React.useState('');
  const [posFilter, setPosFilter] = React.useState('');
  const [selectedColumn, setSelectedColumn] = useState(null);

  const data = React.useMemo(
    () => [
      {
        name: 'Nikita Kucherov',
        team: 'TBL',
        position: 'RW',
        games: 81,
        goals: 44,
        assists: 100,
        points: 144,
      },
      {
        name: 'Nathan MacKinnon',
        team: 'COL',
        position: 'C',
        games: 82,
        goals: 51,
        assists: 89,
        points: 140,
      },
      {
        name: 'Connor McDavid',
        team: 'EDM',
        position: 'C',
        games: 76,
        goals: 32,
        assists: 100,
        points: 132,
      },
      {
        name: 'Artemi Panarin',
        team: 'NYR',
        position: 'LW',
        games: 82,
        goals: 49,
        assists: 71,
        points: 120,
      }
    ],
    []
  );

  const columns = React.useMemo(
    () => [
      {
        Header: 'Name',
        accessor: 'name',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'Team',
        accessor: 'team',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'Position',
        accessor: 'position',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'Games',
        accessor: 'games',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'Goals',
        accessor: 'goals',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'Assists',
        accessor: 'assists',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'Points',
        accessor: 'points',
        Cell: ({ cell: { value } }) => value,
      },
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

  const filteredRows = rows.filter(row => {
    return row.values.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
      (teamFilter ? row.values.team === teamFilter : true) &&
      (posFilter ? row.values.position === posFilter : true);
  });

  // Get unique teams and for the dropdown
  const teams = [...new Set(rows.map(row => row.values.team))].sort();
  const pos = [...new Set(rows.map(row => row.values.position))].sort();

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
            {pos.map(pos => (
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
                        position: column.id === 'name' ? 'sticky' : undefined,
                        left: column.id === 'name' ? 0 : undefined,
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
            {filteredRows.map(row => {
              prepareRow(row);
              return (
                <tr {...row.getRowProps()}>
                  {row.cells.map(cell => (
                    <td
                      {...cell.getCellProps({
                        style: {
                          cusor: 'pointer',
                          backgroundColor: selectedColumn === cell.column.id ? 'rgba(218, 165, 32, 0.15)' : undefined,
                          position: cell.column.id === 'name' ? 'sticky' : undefined,
                          left: cell.column.id === 'name' ? 0 : undefined,
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

export default Players;