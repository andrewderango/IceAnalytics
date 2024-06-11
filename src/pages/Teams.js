import React from 'react';
import { useTable } from 'react-table';
// import { useTable, useSortBy } from 'react-table';
import '../styles/Teams.scss';

function Teams() {
  const data = React.useMemo(
    () => [
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/TBL_dark.svg',
        team: 'Tampa Bay Lightning',
        points: 128,
        goalsFor: 325,
        goalsAgainst: 222,
        playoffProb: 0.987,
        presidentsTrophyProb: 0.502,
        stanleyCupProb: 0.234,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/COL_dark.svg',
        team: 'Colorado Avalanche',
        points: 112,
        goalsFor: 289,
        goalsAgainst: 219,
        playoffProb: 0.999,
        presidentsTrophyProb: 0.201,
        stanleyCupProb: 0.123,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/EDM_dark.svg',
        team: 'Edmonton Oilers',
        points: 107,
        goalsFor: 281,
        goalsAgainst: 234,
        playoffProb: 0.999,
        presidentsTrophyProb: 0.199,
        stanleyCupProb: 0.121,
      }
    ],
    []
  );

  const columns = React.useMemo(
    () => [
      {
        Header: 'Logo',
        accessor: 'logo',
        Cell: ({ value }) => <img src={value} alt="logo" className="team-logo" />,
      },
      {
        Header: 'Team',
        accessor: 'team',
      },
      {
        Header: 'Points',
        accessor: 'points',
      },
      {
        Header: 'GF',
        accessor: 'goalsFor',
      },
      {
        Header: 'GA',
        accessor: 'goalsAgainst',
      },
      {
        Header: 'Playoffs',
        accessor: 'playoffProb',
        Cell: ({ value }) => {
          const color = `rgba(138, 125, 91, ${value*0.9 + 0.1})`;
          return <div className="probability-box" style={{ backgroundColor: color }}>{(value*100).toFixed(1)}%</div>;
        },
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
    <div className="teams">
      <h1>Teams</h1>
      <h2>Last updated May 15, 2024</h2>
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

export default Teams;