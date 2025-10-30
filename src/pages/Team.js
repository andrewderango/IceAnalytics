import React, { useEffect, useState } from 'react';
import { useParams, useHistory } from 'react-router-dom';
import supabase from '../supabaseClient';
import '../styles/Team.scss';
import { offseason, season } from '../config/settings';

function Team() {
  const { teamId } = useParams();
  const history = useHistory();
  const [team, setTeam] = useState(null);
  const [roster, setRoster] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (offseason) {
      history.push('/not-found');
      return;
    }
    // fetchTeam: loads the team by `abbrev` (primary key) or falls back to fuzzy name match
    const fetchTeam = async () => {
      try {
  // interpret the route param as a team identifier (prefer abbrev)
        const decoded = decodeURIComponent(teamId);
        let { data: teamData, error } = await supabase
          .from('team_projections')
          .select('*')
          .eq('abbrev', decoded)
          .limit(1);

        if (error) console.error('Error fetching team:', error);
        let resolvedTeam = null;
        if (teamData && teamData.length > 0) {
          resolvedTeam = teamData[0];
          setTeam(resolvedTeam);
        } else {
          // fallback: convert slug-like ids (carolina-hurricanes) to a name and ilike match
          const maybeName = decoded.replace(/[-_]+/g, ' ').trim();
          const { data: t2, error: e2 } = await supabase
            .from('team_projections')
            .select('*')
            .ilike('team', `%${maybeName}%`)
            .limit(1);
          if (e2) console.error(e2);
          if (t2 && t2.length > 0) {
            resolvedTeam = t2[0];
            setTeam(resolvedTeam);
          }
        }

        // fetch roster players for this team using the player_projections `team` column
        // player_projections.team contains team abbreviations (e.g., TBL, COL). Prefer resolvedTeam.abbrev or the route param (which should be abbrev).
        const rosterKey = (resolvedTeam && resolvedTeam.abbrev) ? resolvedTeam.abbrev : decodeURIComponent(teamId);
        const { data: players, error: pErr } = await supabase
          .from('player_projections')
          .select('*')
          .eq('team', rosterKey);

        if (pErr) console.error('Error fetching roster:', pErr);
        if (players) setRoster(players);

        // fetch all teams projections to compute NHL ranks
        const { data: allTeams, error: atErr } = await supabase
          .from('team_projections')
          .select('team,abbrev,points,goals_for,goals_against,current_points,current_goals_for,current_goals_against,current_pp_pct,current_pk_pct,current_wins,current_losses,current_otl,offense_score,defense_score,overall_score,playoff_prob');

        if (atErr) console.error('Error fetching all teams for ranks:', atErr);
        // compute ranks and attach to team object via local variables
        if (allTeams && allTeams.length > 0) {
          // functions to compute rank; use current_* values when available, otherwise projected
          const numeric = v => parseFloat(v) || 0;

          const projectedPointsSorted = [...allTeams].sort((a,b) => numeric(b.points) - numeric(a.points));
          const projectedGfSorted = [...allTeams].sort((a,b) => numeric(b.goals_for) - numeric(a.goals_for));
          const projectedGaSorted = [...allTeams].sort((a,b) => numeric(a.goals_against) - numeric(b.goals_against));

          const currentPointsSorted = [...allTeams].sort((a,b) => numeric((b.current_points ?? b.points)) - numeric((a.current_points ?? a.points)));
          const currentGfSorted = [...allTeams].sort((a,b) => numeric((b.current_goals_for ?? b.goals_for)) - numeric((a.current_goals_for ?? a.goals_for)));
          // current GA: lower is better
          const currentGaSorted = [...allTeams].sort((a,b) => numeric((a.current_goals_against ?? a.goals_against)) - numeric((b.current_goals_against ?? b.goals_against)));

            // PP/PK ranks (higher PP% is better; higher PK% is better)
            const projectedPpSorted = [...allTeams].sort((a,b) => numeric(b.current_pp_pct ?? 0) - numeric(a.current_pp_pct ?? 0));
            const projectedPkSorted = [...allTeams].sort((a,b) => numeric(b.current_pk_pct ?? 0) - numeric(a.current_pk_pct ?? 0));

            const currentPpSorted = [...allTeams].sort((a,b) => numeric(b.current_pp_pct ?? 0) - numeric(a.current_pp_pct ?? 0));
            const currentPkSorted = [...allTeams].sort((a,b) => numeric(b.current_pk_pct ?? 0) - numeric(a.current_pk_pct ?? 0));

          const findIndex = (arr) => {
            // use the resolvedTeam (available in this scope) when matching ranks; fallback to component state `team` if necessary
            const target = resolvedTeam || team;
            if (!target) return null;
            let idx = arr.findIndex(x => (target.abbrev && x.abbrev === target.abbrev) || x.team === target.team);
            if (idx === -1) {
              // try matching by team name substring
              idx = arr.findIndex(x => x.team && target.team && x.team.toLowerCase().includes(target.team.toLowerCase()));
            }
            return idx === -1 ? null : idx + 1;
          };

            // compute power percents for ranks (use current stats when available, otherwise projected)
            const toPct = v => {
              const n = parseFloat(v);
              if (!isFinite(n)) return 0;
              return Math.max(0, Math.min(100, Math.round(n)));
            };

            const offenseList = allTeams.map(t => ({
              ...t,
              offensePct: t.offense_score != null ? toPct(t.offense_score) : (
                (t.current_goals_for && t.current_goals_against) 
                  ? Math.round((t.current_goals_for / (t.current_goals_for + t.current_goals_against)) * 100) 
                  : (t.goals_for && t.goals_against ? Math.round((t.goals_for / (t.goals_for + t.goals_against)) * 100) : 0)
              )
            }));
            const defenseList = allTeams.map(t => ({
              ...t,
              defensePct: t.defense_score != null ? toPct(t.defense_score) : (
                (t.current_goals_for && t.current_goals_against)
                  ? Math.round((1 - (t.current_goals_against / (t.current_goals_for + t.current_goals_against))) * 100)
                  : (t.goals_for && t.goals_against ? Math.round((1 - (t.goals_against / (t.goals_for + t.goals_against))) * 100) : 0)
              )
            }));
            const overallList = allTeams.map(t => ({
              ...t,
              overallPct: t.overall_score != null ? toPct(t.overall_score) : Math.round((parseFloat(t.playoff_prob || 0) * 100))
            }));

            const offenseSorted = [...offenseList].sort((a,b) => (b.offensePct || 0) - (a.offensePct || 0));
            const defenseSorted = [...defenseList].sort((a,b) => (b.defensePct || 0) - (a.defensePct || 0));
            const overallSorted = [...overallList].sort((a,b) => (b.overallPct || 0) - (a.overallPct || 0));

            const offenseRank = findIndex(offenseSorted);
            const defenseRank = findIndex(defenseSorted);
            const overallRank = findIndex(overallSorted);

          const ptsRank = findIndex(projectedPointsSorted);
          const gfRank = findIndex(projectedGfSorted);
          const gaRank = findIndex(projectedGaSorted);
          const ppRank = findIndex(projectedPpSorted);
          const pkRank = findIndex(projectedPkSorted);

          const currPtsRank = findIndex(currentPointsSorted);
          const currGfRank = findIndex(currentGfSorted);
          const currGaRank = findIndex(currentGaSorted);
          const currPPRank = findIndex(currentPpSorted);
          const currPKRank = findIndex(currentPkSorted);

          // attach ranks to team state object for rendering
          setTeam(prev => ({ ...prev, _ranks: { ptsRank, gfRank, gaRank, ppRank, pkRank, currPtsRank, currGfRank, currGaRank, currPPRank, currPKRank, offenseRank, defenseRank, overallRank, totalTeams: allTeams.length } }));
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchTeam();
  }, [teamId, history]);

  if (loading) return <div className="team-page">Loading...</div>;
  if (!team) return <div className="team-page">Team not found</div>;

  // split roster by `pos` field from player_projections (values like D, C, LW, RW, G)
  const forwards = roster.filter(p => p.position && !['D', 'G'].includes(String(p.position).toUpperCase()));
  const defense = roster.filter(p => String(p.position).toUpperCase() === 'D');

  

  // derive a primary color from available team fields and a readable foreground color
  const teamPrimaryRaw = team.primary_color || team.color || team.team_color || team.color_primary || team.hex || team.hex_color || '#B01829'; // change to null ###!!!
  const parseHex = (s) => {
    if (!s) return null;
    const hex = s.trim();
    // handle rgb(...) strings
    if (hex.startsWith('rgb')) {
      const nums = hex.replace(/[rgba()]/g, '').split(',').map(n => parseInt(n, 10));
      if (nums.length >= 3) return { r: nums[0], g: nums[1], b: nums[2] };
      return null;
    }
    // normalize #rgb to #rrggbb
    if (/^#?[0-9a-f]{3}$/i.test(hex)) {
      const clean = hex.replace('#', '');
      return {
        r: parseInt(clean[0] + clean[0], 16),
        g: parseInt(clean[1] + clean[1], 16),
        b: parseInt(clean[2] + clean[2], 16),
      };
    }
    if (/^#?[0-9a-f]{6}$/i.test(hex)) {
      const clean = hex.replace('#', '');
      return {
        r: parseInt(clean.substring(0,2), 16),
        g: parseInt(clean.substring(2,4), 16),
        b: parseInt(clean.substring(4,6), 16),
      };
    }
    return null;
  };

  const getContrast = (colorStr) => {
    const rgb = parseHex(colorStr);
    if (!rgb) return '#ffffff';
    // luminance approximation
    const lum = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
    return lum > 0.6 ? '#000000' : '#ffffff';
  };

  const teamPrimary = teamPrimaryRaw ? teamPrimaryRaw : null;
  const teamForeground = teamPrimary ? getContrast(teamPrimary) : '#ffffff';

  const teamNameStyle = teamPrimary ? { '--team-primary': teamPrimary, '--team-foreground': teamForeground } : {};

  const ordinal = (n) => {
    if (!n && n !== 0) return '—';
    const s = ["th","st","nd","rd"], v = n % 100;
    return n + (s[(v-20)%10] || s[v] || s[0]);
  };

  // helper: extract jersey number with common fallbacks
  const getJerseyNumber = (p) => p.number || p.jersey || p.jersey_number || p.uniform_number || p.num || null;

  // (removed) birth country helper per request

  // helper: compute age from explicit age or birth_date
  const getAge = (p) => {
    if (p.age) return p.age;
    const bd = p.birth_date || p.birthdate || p.dob || p.birth;
    if (!bd) return null;
    try {
      const birth = new Date(bd);
      if (isNaN(birth)) return null;
      const now = new Date();
      let age = now.getFullYear() - birth.getFullYear();
      const m = now.getMonth() - birth.getMonth();
      if (m < 0 || (m === 0 && now.getDate() < birth.getDate())) age--;
      return age;
    } catch (e) {
      return null;
    }
  };

  // helper: preferred projected stat with fallbacks
  const pickStat = (p, keys) => {
    for (const k of keys) {
      if (p[k] !== undefined && p[k] !== null) return p[k];
    }
    return 0;
  };


  return (
    <div className="team-page" style={teamNameStyle}>
      <div className="team-hero">
        <h1 className="team-name">
          <img src={team.logo} alt={`${team.team} logo`} className="team-name-logo" />
          <span className="team-name-text">{team.team}</span>
        </h1>
        
        <div className="hero-content">
          <div className="hero-left">
            <div className="logo-wrap">
              <div className="stat-card record-card left-record-card">
                <div className="stat-label">Current Record</div>
                <div className="stat-value">{`${Math.round(team.current_wins || team.wins || 0)} - ${Math.round(team.current_losses || team.losses || 0)} - ${Math.round(team.current_otl || team.otl || 0)}`}</div>
                <div className="stat-sub">Rank: {team._ranks && team._ranks.currPtsRank ? `${ordinal(team._ranks.currPtsRank)}` : '—'}</div>
              </div>
            </div>
          </div>

          <div className="hero-right">
            <div className="hero-top">
              {/* Goals For / Against summary tile */}
              <div className="stat-card record-card">
                <div className="stat-label">Goals For / Against</div>
                {
                  (() => {
                    const gf = Math.round(team.current_goals_for || team.goals_for || 0);
                    const ga = Math.round(team.current_goals_against || team.goals_against || 0);
                    const diff = gf - ga;
                    const diffStr = `${diff >= 0 ? '+' : ''}${diff}`;
                    const rank = team._ranks && (team._ranks.currGfRank || team._ranks.currGfRank === 0) ? team._ranks.currGfRank : null;
                    const ordinal = (n) => {
                      if (!n) return '—';
                      const s = ["th","st","nd","rd"], v = n%100;
                      return n + (s[(v-20)%10]||s[v]||s[0]);
                    };
                    return (
                      <>
                        <div className="stat-value">{`${gf}-${ga} `}<span className={diff > 0 ? 'diff-pos' : (diff < 0 ? 'diff-neg' : undefined)}>{`(`}{diffStr}{`)`}</span></div>
                        <div className="stat-sub">Rank: {rank != null ? ordinal(rank) : '—'}</div>
                      </>
                    );
                  })()
                }
              </div>
            </div>

            <div className="hero-bottom">
              <div className="stat-card">
                <div className="stat-label">PP%</div>
                {
                  (() => {
                    const raw = team.current_pp_pct ?? 0;
                    const pct = (parseFloat(raw) || 0) > 1 ? (parseFloat(raw)) : (parseFloat(raw) * 100);
                    return (
                      <>
                        <div className="stat-value">{pct.toFixed(1)}%</div>
                        <div className="stat-sub">Rank: {team._ranks && team._ranks.currPPRank ? `${ordinal(team._ranks.currPPRank)}` : '—'}</div>
                      </>
                    );
                  })()
                }
              </div>

              <div className="stat-card">
                <div className="stat-label">PK%</div>
                {
                  (() => {
                    const raw = team.current_pk_pct ?? 0;
                    const pct = (parseFloat(raw) || 0) > 1 ? (parseFloat(raw)) : (parseFloat(raw) * 100);
                    return (
                      <>
                        <div className="stat-value">{pct.toFixed(1)}%</div>
                        <div className="stat-sub">Rank: {team._ranks && team._ranks.currPKRank ? `${ordinal(team._ranks.currPKRank)}` : '—'}</div>
                      </>
                    );
                  })()
                }
              </div>
            </div>
          </div>
        
          <div className="hero-extra">
            <div className="power-box">
              <div className="power-heading">Power Scores</div>
              {(() => {
              // compute percent values with fallbacks
              const toPct = v => {
                const n = parseFloat(v);
                if (!isFinite(n)) return 0;
                return Math.max(0, Math.min(100, Math.round(n)));
              };

              const offenseRaw = team.offense_score ?? team.offensive_power ?? team.offence ?? null;
              const defenseRaw = team.defense_score ?? team.defensive_power ?? null;
              const overallRaw = team.overall_score ?? team.power_score ?? null;

              // sensible fallbacks (use current stats when available, otherwise projected)
              const offensePct = offenseRaw != null ? toPct(offenseRaw) : (
                (team.current_goals_for && team.current_goals_against) 
                  ? Math.round((team.current_goals_for / (team.current_goals_for + team.current_goals_against)) * 100)
                  : (team.goals_for && team.goals_against ? Math.round((team.goals_for / (team.goals_for + team.goals_against)) * 100) : Math.round((parseFloat(team.playoff_prob || 0) * 100)))
              );
              const defensePct = defenseRaw != null ? toPct(defenseRaw) : (
                (team.current_goals_for && team.current_goals_against)
                  ? Math.round((1 - (team.current_goals_against / (team.current_goals_for + team.current_goals_against))) * 100)
                  : (team.goals_for && team.goals_against ? Math.round((1 - (team.goals_against / (team.goals_for + team.goals_against))) * 100) : Math.round((1 - parseFloat(team.playoff_prob || 0)) * 100))
              );
              const overallPct = overallRaw != null ? toPct(overallRaw) : Math.round((parseFloat(team.playoff_prob || 0) * 100));

              const makeChart = (key, label, pct, _color, rank) => {
                // map pct (0..100) to a color between red (0) and green (100)
                const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
                const lerp = (a, b, t) => Math.round(a + (b - a) * t);

                const t = clamp((parseFloat(pct) || 0) / 100, 0, 1);
                // red and green anchors (RGB)
                const r0 = 234, g0 = 84, b0 = 85;   // red-ish
                const r1 = 76, g1 = 175, b1 = 80;   // green-ish

                const r = lerp(r0, r1, t);
                const g = lerp(g0, g1, t);
                const b = lerp(b0, b1, t);

                const outerColor = `rgba(${r},${g},${b},0.94)`;
                const innerColor = `rgba(${r},${g},${b},0.22)`;

                return (
                  <div className="power-item" key={key}>
                    <div className="power-chart" style={{ ['--ring']: `conic-gradient(${outerColor} ${pct * 3.6}deg, rgba(255,255,255,0.04) 0deg)` }}>
                      <div className="power-inner" style={{ background: innerColor }}>
                        <div className="power-value">{pct}</div>
                        <div className="power-rank">{rank ? `${ordinal(rank)}` : '—'}</div>
                      </div>
                    </div>
                    <div className="power-caption">{label}</div>
                  </div>
                );
              };

              return (
                <div className="power-row">
                  {makeChart('off', 'Offense', offensePct, 'rgba(234,84,85,0.94)', team._ranks && team._ranks.offenseRank)}
                  {makeChart('def', 'Defense', defensePct, 'rgba(76,175,80,0.94)', team._ranks && team._ranks.defenseRank)}
                  {makeChart('ovr', 'Overall', overallPct, 'rgba(138,125,91,0.94)', team._ranks && team._ranks.overallRank)}
                </div>
              );
            })()}
            </div>
          </div>
        </div>
      </div>

        {/* Projections section - dedicated and clearly labeled */}
      <div className="team-projections">
        <h2>Projections</h2>
        <div className="projections-cards">
          <div className="proj-card">
            <div className="label">Projected Record</div>
            <div className="big">{Math.round(team.wins)} - {Math.round(team.losses)} - {Math.round(team.otl)}</div>
          </div>

          <div className="proj-card">
            <div className="label">Projected Points</div>
            <div className="big">{Math.round(team.points)}</div>
            <div className="sub">Rank: {team._ranks && team._ranks.ptsRank ? `${ordinal(team._ranks.ptsRank)}` : '—'}</div>
          </div>

          <div className="proj-card">
            <div className="label">Projected GF</div>
            <div className="big">{Math.round(team.goals_for)}</div>
            <div className="sub">Rank: {team._ranks && team._ranks.gfRank ? `${ordinal(team._ranks.gfRank)}` : '—'}</div>
          </div>

          <div className="proj-card">
            <div className="label">Projected GA</div>
            <div className="big">{Math.round(team.goals_against)}</div>
            <div className="sub">Rank: {team._ranks && team._ranks.gaRank ? `${ordinal(team._ranks.gaRank)}` : '—'}</div>
          </div>

          <div className="proj-card">
            <div className="label">Playoff Odds</div>
            <div className="big">{(parseFloat(team.playoff_prob || 0) * 100).toFixed(1)}%</div>
          </div>

          <div className="proj-card">
            <div className="label">President's Trophy Odds</div>
            <div className="big">{(parseFloat(team.presidents_trophy_prob || 0) * 100).toFixed(2)}%</div>
          </div>
        </div>
      </div>

      <div className="team-roster">
        <h2>Roster</h2>
        <div className="roster-grid">
          <div className="roster-section">
            <h3>Forwards</h3>
            <div className="player-grid">
              {forwards.length === 0 && <div className="note">No forwards found for this team.</div>}
              {[...forwards].sort((a, b) => {
                const pa = pickStat(a, ['points', 'proj_points', 'projected_points']);
                const pb = pickStat(b, ['points', 'proj_points', 'projected_points']);
                return (pb || 0) - (pa || 0);
              }).map((p, idx) => {
                const key = p.player_id || p.id || `${p.team || 't'}-${p.player || 'p'}-${idx}`;
                const num = getJerseyNumber(p);
                const age = getAge(p);
                const projPoints = pickStat(p, ['points']);
                const projGP = pickStat(p, ['games']);
                const projGoals = pickStat(p, ['goals']);
                const projAssists = pickStat(p, ['assists']);
                return (
                <div key={key} className="player-card">
                  <div className="player-top">
                    <div className="player-avatar">
                      <img src={`https://assets.nhle.com/mugs/nhl/${season}/${p.team}/${p.player_id}.png`} alt={p.player} />
                    </div>
                    <div className="player-info">
                      <div className="player-name">{p.player}</div>
                      <div className="player-sub">
                        {(() => {
                          const parts = [];
                          parts.push(p.position || '—');
                          if (num) parts.push(`#${num}`);
                          parts.push(`${age != null ? age : '—'}yrs`);
                          return parts.join(' · ');
                        })()}
                      </div>
                    </div>
                  </div>
                  <div className="player-stats">
                    <div className="stat">
                      <div className="stat-num">{Math.round(projGP || 0)}</div>
                      <div className="stat-label">GP</div>
                    </div>
                    <div className="stat">
                      <div className="stat-num">{Math.round(projGoals || 0)}</div>
                      <div className="stat-label">G</div>
                    </div>
                    <div className="stat">
                      <div className="stat-num">{Math.round(projAssists || 0)}</div>
                      <div className="stat-label">A</div>
                    </div>
                    <div className="stat">
                      <div className="stat-num">{Math.round(projPoints || 0)}</div>
                      <div className="stat-label">P</div>
                    </div>
                  </div>
                </div>
              );})}
            </div>
          </div>

          <div className="roster-section">
            <h3>Defense</h3>
            <div className="player-grid">
              {defense.length === 0 && <div className="note">No defensemen found for this team.</div>}
              {[...defense].sort((a, b) => {
                const pa = pickStat(a, ['points', 'proj_points', 'projected_points', 'points_proj']);
                const pb = pickStat(b, ['points', 'proj_points', 'projected_points', 'points_proj']);
                return (pb || 0) - (pa || 0);
              }).map((p, idx) => {
                const key = p.player_id || p.id || `${p.team || 't'}-${p.player || 'p'}-${idx}`;
                const num = getJerseyNumber(p);
                const age = getAge(p);
                const projGoals = pickStat(p, ['goals', 'proj_goals', 'projected_goals']);
                const projAssists = pickStat(p, ['assists', 'proj_assists', 'projected_assists']);
                const projPoints = pickStat(p, ['points', 'proj_points', 'projected_points']);
                const projGP = pickStat(p, ['gp', 'proj_gp', 'projected_gp', 'games_played']);
                return (
                <div key={key} className="player-card">
                  <div className="player-top">
                    <div className="player-avatar">
                      <img src={`https://assets.nhle.com/mugs/nhl/${season}/${p.team}/${p.player_id}.png`} alt={p.player} />
                    </div>
                    <div className="player-info">
                      <div className="player-name">{p.player}</div>
                      <div className="player-sub">
                        {(() => {
                          const parts = [];
                          parts.push(p.position || '—');
                          if (num) parts.push(`#${num}`);
                          parts.push(`${age != null ? age : '—'}yrs`);
                          return parts.join(' · ');
                        })()}
                      </div>
                    </div>
                  </div>
                  <div className="player-stats">
                    <div className="stat">
                      <div className="stat-num">{Math.round(projGP || 0)}</div>
                      <div className="stat-label">GP</div>
                    </div>
                    <div className="stat">
                      <div className="stat-num">{Math.round(projGoals || 0)}</div>
                      <div className="stat-label">G</div>
                    </div>
                    <div className="stat">
                      <div className="stat-num">{Math.round(projAssists || 0)}</div>
                      <div className="stat-label">A</div>
                    </div>
                    <div className="stat">
                      <div className="stat-num">{Math.round(projPoints || 0)}</div>
                      <div className="stat-label">P</div>
                    </div>
                  </div>
                </div>
              );})}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Team;
