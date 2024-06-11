import React, { useState } from 'react';
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
  const [selectedColumn, setSelectedColumn] = useState(null);

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
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === selectedColumn;
          return <div className={isSelected ? 'selected-column' : ''}>{value}</div>;
        },
      },
      {
        Header: 'Points',
        accessor: 'points',
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === selectedColumn;
          return <div className={isSelected ? 'selected-column' : ''}>{value}</div>;
        },
      },
      {
        Header: 'GF',
        accessor: 'goalsFor',
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === selectedColumn;
          return <div className={isSelected ? 'selected-column' : ''}>{value}</div>;
        },
      },
      {
        Header: 'GA',
        accessor: 'goalsAgainst',
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === selectedColumn;
          return <div className={isSelected ? 'selected-column' : ''}>{value}</div>;
        },
      },
      {
        Header: 'Playoffs',
        accessor: 'playoffProb',
        Cell: ({ cell: { value }, column: { id } }) => {
          const isSelected = id === selectedColumn;
          const color = `rgba(138, 125, 91, ${value*0.9 + 0.1})`;
          return (
            <div 
              className={isSelected ? 'selected-column' : ''} 
              style={{ backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
            >
              {(value*100).toFixed(1)}%
            </div>
          );
        },
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
                    <th {...column.getHeaderProps()} onClick={() => setSelectedColumn(column.id)}>
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
                  {row.cells.map(cell => {
                    const isSelected = cell.column.id === selectedColumn;
                    return (
                      <td {...cell.getCellProps()} className={isSelected ? 'selected-column' : ''}>
                        {cell.render('Cell')}
                      </td>
                    );
                  })}
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