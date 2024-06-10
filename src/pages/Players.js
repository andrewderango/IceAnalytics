import React from 'react';
import { useTable } from 'react-table';
// import { useTable, useSortBy } from 'react-table';
import '../styles/Players.scss';

function Players() {
  const [searchTerm, setSearchTerm] = React.useState('');
  const [teamFilter, setTeamFilter] = React.useState('');
  const [posFilter, setPosFilter] = React.useState('');

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
      },
      {
        name: 'David Pastrnak',
        team: 'BOS',
        position: 'RW',
        games: 82,
        goals: 47,
        assists: 63,
        points: 110,
      },
      {
        name: 'Leon Draisaitl',
        team: 'EDM',
        position: 'C',
        games: 80,
        goals: 50,
        assists: 60,
        points: 110,
      },
      {
        name: 'Auston Matthews',
        team: 'TOR',
        position: 'C',
        games: 82,
        goals: 60,
        assists: 50,
        points: 110,
      },
      {
        name: 'Patrick Kane',
        team: 'CHI',
        position: 'RW',
        games: 81,
        goals: 40,
        assists: 65,
        points: 105,
      },
      {
        name: 'Brad Marchand',
        team: 'BOS',
        position: 'LW',
        games: 82,
        goals: 45,
        assists: 55,
        points: 100,
      },
      {
        name: 'Mikko Rantanen',
        team: 'COL',
        position: 'RW',
        games: 80,
        goals: 39,
        assists: 60,
        points: 99,
      },
      {
        name: 'Jack Eichel',
        team: 'VGK',
        position: 'C',
        games: 79,
        goals: 37,
        assists: 61,
        points: 98,
      },
      {
        name: 'Johnny Gaudreau',
        team: 'CBJ',
        position: 'LW',
        games: 82,
        goals: 36,
        assists: 62,
        points: 98,
      },
      {
        name: 'Steven Stamkos',
        team: 'TBL',
        position: 'C',
        games: 80,
        goals: 44,
        assists: 52,
        points: 96,
      },
      {
        name: 'Jonathan Huberdeau',
        team: 'CGY',
        position: 'LW',
        games: 82,
        goals: 31,
        assists: 64,
        points: 95,
      },
      {
        name: 'Alexander Ovechkin',
        team: 'WSH',
        position: 'LW',
        games: 78,
        goals: 50,
        assists: 44,
        points: 94,
      },
      {
        name: 'Sidney Crosby',
        team: 'PIT',
        position: 'C',
        games: 82,
        goals: 34,
        assists: 60,
        points: 94,
      },
      {
        name: 'Mika Zibanejad',
        team: 'NYR',
        position: 'C',
        games: 81,
        goals: 35,
        assists: 58,
        points: 93,
      },
      {
        name: 'Kirill Kaprizov',
        team: 'MIN',
        position: 'LW',
        games: 77,
        goals: 42,
        assists: 49,
        points: 91,
      },
      {
        name: 'Sebastian Aho',
        team: 'CAR',
        position: 'C',
        games: 80,
        goals: 33,
        assists: 57,
        points: 90,
      },
      {
        name: 'Mark Scheifele',
        team: 'WPG',
        position: 'C',
        games: 81,
        goals: 38,
        assists: 52,
        points: 90,
      },
      {
        name: 'Elias Pettersson',
        team: 'VAN',
        position: 'C',
        games: 82,
        goals: 32,
        assists: 57,
        points: 89,
      },
      {
        name: 'Brayden Point',
        team: 'TBL',
        position: 'C',
        games: 80,
        goals: 41,
        assists: 47,
        points: 88,
      },
      {
        name: 'Matthew Tkachuk',
        team: 'FLA',
        position: 'LW',
        games: 80,
        goals: 30,
        assists: 58,
        points: 88,
      },
      {
        name: 'Timo Meier',
        team: 'NJD',
        position: 'RW',
        games: 78,
        goals: 35,
        assists: 50,
        points: 85,
      },
      {
        name: 'Cale Makar',
        team: 'COL',
        position: 'D',
        games: 82,
        goals: 28,
        assists: 57,
        points: 85,
      },
    ],
    []
  );

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
          placeholder="Search players..."
        />
      </div>
      <div className="table-container">
        <table {...getTableProps()} style={{ color: 'white', backgroundColor: '#333' }}>
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
            {filteredRows.map(row => {
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
    </div>
  );
}

export default Players;