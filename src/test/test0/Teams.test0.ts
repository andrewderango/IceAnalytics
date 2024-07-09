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
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/NYR_dark.svg',
        team: 'New York Rangers',
        points: 106,
        goalsFor: 280,
        goalsAgainst: 234,
        playoffProb: 0.912,
        presidentsTrophyProb: 0.108,
        stanleyCupProb: 0.120,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/BOS_dark.svg',
        team: 'Boston Bruins',
        points: 105,
        goalsFor: 278,
        goalsAgainst: 235,
        playoffProb: 0.710,
        presidentsTrophyProb: 0.016,
        stanleyCupProb: 0.119,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/FLA_dark.svg',
        team: 'Florida Panthers',
        points: 103,
        goalsFor: 276,
        goalsAgainst: 236,
        playoffProb: 0.437,
        presidentsTrophyProb: 0.010,
        stanleyCupProb: 0.117,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/CGY_dark.svg',
        team: 'Calgary Flames',
        points: 102,
        goalsFor: 275,
        goalsAgainst: 237,
        playoffProb: 0.235,
        presidentsTrophyProb: 0.001,
        stanleyCupProb: 0.116,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/WSH_dark.svg',
        team: 'Washington Capitals',
        points: 101,
        goalsFor: 274,
        goalsAgainst: 238,
        playoffProb: 0.001,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.115,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/CHI_dark.svg',
        team: 'Chicago Blackhawks',
        points: 100,
        goalsFor: 273,
        goalsAgainst: 239,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.114,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/PHI_dark.svg',
        team: 'Philadelphia Flyers',
        points: 99,
        goalsFor: 272,
        goalsAgainst: 240,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.113,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/STL_dark.svg',
        team: 'St. Louis Blues',
        points: 98,
        goalsFor: 271,
        goalsAgainst: 241,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.112,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/MTL_dark.svg',
        team: 'Montreal Canadiens',
        points: 97,
        goalsFor: 270,
        goalsAgainst: 242,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.111,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/SEA_dark.svg',
        team: 'Seattle Kraken',
        points: 96,
        goalsFor: 269,
        goalsAgainst: 243,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.110,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/OTT_dark.svg',
        team: 'Ottawa Senators',
        points: 95,
        goalsFor: 268,
        goalsAgainst: 244,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.109,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/DET_dark.svg',
        team: 'Detroit Red Wings',
        points: 94,
        goalsFor: 267,
        goalsAgainst: 245,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.108,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/NYI_dark.svg',
        team: 'New York Islanders',
        points: 93,
        goalsFor: 266,
        goalsAgainst: 246,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.107,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/ARI_dark.svg',
        team: 'Arizona Coyotes',
        points: 92,
        goalsFor: 265,
        goalsAgainst: 247,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.106,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/VAN_dark.svg',
        team: 'Vancouver Canucks',
        points: 91,
        goalsFor: 264,
        goalsAgainst: 248,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.105,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/NJD_dark.svg',
        team: 'New Jersey Devils',
        points: 90,
        goalsFor: 263,
        goalsAgainst: 249,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.104,
      },
      {
        logo: 'https://assets.nhle.com/logos/nhl/svg/SJS_dark.svg',
        team: 'San Jose Sharks',
        points: 89,
        goalsFor: 262,
        goalsAgainst: 250,
        playoffProb: 0.000,
        presidentsTrophyProb: 0.000,
        stanleyCupProb: 0.103,
      },
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
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'Points',
        accessor: 'points',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'GF',
        accessor: 'goalsFor',
        Cell: ({ cell: { value } }) => value,
      },
      {
        Header: 'GA',
        accessor: 'goalsAgainst',
        Cell: ({ cell: { value } }) => value,
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
              style={{ color: 'white', backgroundColor: color, padding: '5px', borderRadius: '5px', width: '75px', margin: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)'}}
            >
              {(value*100).toFixed(1)}%
            </div>
          );
        },
      },
      {
        Header: "Presidents' Trophy",
        accessor: 'presidentsTrophyProb',
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
        accessor: 'stanleyCupProb',
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
                  {headerGroup.headers.map(column => {
                    const isSelected = column.id === selectedColumn;
                    return (
                      <th 
                        {...column.getHeaderProps()} 
                        onClick={() => setSelectedColumn(prev => prev === column.id ? null : column.id)}
                        style={{ backgroundColor: isSelected ? 'rgba(218, 165, 32, 0.5)' : '' }}
                      >
                        {column.render('Header')}
                      </th>
                    );
                  })}
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