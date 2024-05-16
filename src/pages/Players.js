import React from 'react';
import { useTable } from 'react-table';
// import { useTable, useSortBy } from 'react-table';
import '../styles/Players.scss';

function Players() {
  const data = React.useMemo(
    () => [
      {
        name: 'Nikita Kucherov',
        team: 'TBL',
        games: 81,
        goals: 44,
        assists: 100,
        points: 144,
      },
      {
        name: 'Nathan MacKinnon',
        team: 'COL',
        games: 82,
        goals: 51,
        assists: 89,
        points: 140,
      },
      {
        name: 'Connor McDavid',
        team: 'EDM',
        games: 76,
        goals: 32,
        assists: 100,
        points: 132,
      },
      {
        name: 'Artemi Panarin',
        team: 'NYR',
        games: 82,
        goals: 49,
        assists: 71,
        points: 120,
      },
      {
        name: 'David Pastrnak',
        team: 'BOS',
        games: 82,
        goals: 47,
        assists: 63,
        points: 110,
      },
      {
        name: 'Leon Draisaitl',
        team: 'EDM',
        games: 80,
        goals: 50,
        assists: 60,
        points: 110,
      },
      {
        name: 'Auston Matthews',
        team: 'TOR',
        games: 82,
        goals: 60,
        assists: 50,
        points: 110,
      },
      {
        name: 'Patrick Kane',
        team: 'CHI',
        games: 81,
        goals: 40,
        assists: 65,
        points: 105,
      },
      {
        name: 'Brad Marchand',
        team: 'BOS',
        games: 82,
        goals: 45,
        assists: 55,
        points: 100,
      },
      {
        name: 'Mikko Rantanen',
        team: 'COL',
        games: 80,
        goals: 39,
        assists: 60,
        points: 99,
      },
      {
        name: 'Jack Eichel',
        team: 'VGK',
        games: 79,
        goals: 37,
        assists: 61,
        points: 98,
      },
      {
        name: 'Johnny Gaudreau',
        team: 'CBJ',
        games: 82,
        goals: 36,
        assists: 62,
        points: 98,
      },
      {
        name: 'Steven Stamkos',
        team: 'TBL',
        games: 80,
        goals: 44,
        assists: 52,
        points: 96,
      },
      {
        name: 'Jonathan Huberdeau',
        team: 'CGY',
        games: 82,
        goals: 31,
        assists: 64,
        points: 95,
      },
      {
        name: 'Alexander Ovechkin',
        team: 'WSH',
        games: 78,
        goals: 50,
        assists: 44,
        points: 94,
      },
      {
        name: 'Sidney Crosby',
        team: 'PIT',
        games: 82,
        goals: 34,
        assists: 60,
        points: 94,
      },
      {
        name: 'Mika Zibanejad',
        team: 'NYR',
        games: 81,
        goals: 35,
        assists: 58,
        points: 93,
      },
      {
        name: 'Kirill Kaprizov',
        team: 'MIN',
        games: 77,
        goals: 42,
        assists: 49,
        points: 91,
      },
      {
        name: 'Sebastian Aho',
        team: 'CAR',
        games: 80,
        goals: 33,
        assists: 57,
        points: 90,
      },
      {
        name: 'Mark Scheifele',
        team: 'WPG',
        games: 81,
        goals: 38,
        assists: 52,
        points: 90,
      },
      {
        name: 'Elias Pettersson',
        team: 'VAN',
        games: 82,
        goals: 32,
        assists: 57,
        points: 89,
      },
      {
        name: 'Brayden Point',
        team: 'TBL',
        games: 80,
        goals: 41,
        assists: 47,
        points: 88,
      },
      {
        name: 'Matthew Tkachuk',
        team: 'FLA',
        games: 80,
        goals: 30,
        assists: 58,
        points: 88,
      },
      {
        name: 'Timo Meier',
        team: 'NJD',
        games: 78,
        goals: 35,
        assists: 50,
        points: 85,
      },
      {
        name: 'Teuvo Teravainen',
        team: 'CAR',
        games: 82,
        goals: 28,
        assists: 57,
        points: 85,
      },
      // More players here...
    ],
    []
  );

  const columns = React.useMemo(
    () => [
      {
        Header: 'Name',
        accessor: 'name', // accessor is the "key" in the data
      },
      {
        Header: 'Team',
        accessor: 'team',
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

  return (
    <div className="players">
      <h1>Players</h1>
      <h2>Last updated May 15, 2024.</h2>
      <div class="container">
        <div class="tabs">
          <input type="radio" id="radio-1" name="tabs"/>
          <label class="tab" for="radio-1">Remaining</label>
          <input type="radio" id="radio-2" name="tabs" />
          <label class="tab" for="radio-2">Total</label>
          <input type="radio" id="radio-3" name="tabs" />
          <label class="tab" for="radio-3">Current</label>
          <span class="glider"></span>
        </div>
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
    </div>
  );
}

export default Players;