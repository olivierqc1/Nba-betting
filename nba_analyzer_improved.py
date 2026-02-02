"""
NBA Betting Analyzer v8.3
- CURRENT SEASON ONLY (2024-25) - no fallback to old data
- Better NBA.com headers to avoid blocking
- Longer timeout + retry logic
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime
import os
import numpy as np
from scipy import stats as scipy_stats
import time
import random

app = Flask(__name__)
CORS(app)

ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
NBA_STATS_URL = "https://stats.nba.com/stats/leaguedashplayerstats"
NBA_GAME_LOG_URL = "https://stats.nba.com/stats/playergamelog"

# Enhanced headers to avoid NBA.com blocking
NBA_HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}

PLAYER_ID_CACHE = {}
DEBUG_LOG = []


def log_debug(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    DEBUG_LOG.append(f"[{timestamp}] {msg}")
    print(f"[{timestamp}] {msg}")
    if len(DEBUG_LOG) > 50:
        DEBUG_LOG.pop(0)


def normalize_name(name):
    name = name.lower().strip()
    for s in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', '.']:
        name = name.replace(s, '')
    return name.strip()


def make_nba_request(url, params, max_retries=2):
    """Make request to NBA API with retry logic"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = 2 + random.random() * 2
                log_debug(f"Retry {attempt+1}, waiting {wait:.1f}s...")
                time.sleep(wait)
            
            response = requests.get(
                url, 
                headers=NBA_HEADERS, 
                params=params, 
                timeout=25
            )
            
            log_debug(f"NBA API response: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                log_debug("403 Forbidden - NBA blocking request")
            elif response.status_code == 429:
                log_debug("429 Rate limited")
                time.sleep(5)
                
        except requests.exceptions.Timeout:
            log_debug(f"Timeout on attempt {attempt+1}")
        except Exception as e:
            log_debug(f"Error: {str(e)[:50]}")
    
    return None


def get_all_player_averages():
    """Get current season stats ONLY - no fallback to old seasons"""
    global PLAYER_ID_CACHE
    
    season = '2024-25'  # Current season ONLY
    log_debug(f"Fetching {season} stats (current season only)...")
    
    params = {
        'Conference': '', 'DateFrom': '', 'DateTo': '', 'Division': '',
        'GameScope': '', 'GameSegment': '', 'Height': '', 'LastNGames': '0',
        'LeagueID': '00', 'Location': '', 'MeasureType': 'Base', 'Month': '0',
        'OpponentTeamID': '0', 'Outcome': '', 'PORound': '0', 'PaceAdjust': 'N',
        'PerMode': 'PerGame', 'Period': '0', 'PlayerExperience': '',
        'PlayerPosition': '', 'PlusMinus': 'N', 'Rank': 'N', 'Season': season,
        'SeasonSegment': '', 'SeasonType': 'Regular Season', 'ShotClockRange': '',
        'StarterBench': '', 'TeamID': '0', 'TwoWay': '0', 'VsConference': '',
        'VsDivision': '', 'Weight': ''
    }
    
    data = make_nba_request(NBA_STATS_URL, params)
    
    if data and 'resultSets' in data:
        headers = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        
        if not rows:
            log_debug(f"No data for {season}")
            return {}
        
        players = {}
        for row in rows:
            name = row[headers.index('PLAYER_NAME')]
            pid = row[headers.index('PLAYER_ID')]
            PLAYER_ID_CACHE[normalize_name(name)] = pid
            players[normalize_name(name)] = {
                'id': pid,
                'name': name,
                'pts': row[headers.index('PTS')],
                'ast': row[headers.index('AST')],
                'reb': row[headers.index('REB')],
                'gp': row[headers.index('GP')],
                'min': row[headers.index('MIN')]
            }
        
        log_debug(f"SUCCESS: {len(players)} players from {season}")
        return players
    
    log_debug("FAILED: NBA API blocked or unavailable")
    return {}


def get_player_game_log(player_id, stat_type='points'):
    """Get player game log for CURRENT season only"""
    stat_map = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB'}
    stat_col = stat_map.get(stat_type, 'PTS')
    
    season = '2024-25'  # Current season ONLY
    params = {
        'PlayerID': player_id,
        'Season': season,
        'SeasonType': 'Regular Season',
        'LeagueID': '00'
    }
    
    data = make_nba_request(NBA_GAME_LOG_URL, params, max_retries=1)
    
    if data and 'resultSets' in data:
        h = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        
        if not rows:
            return None
        
        games = []
        for row in rows:
            mins = row[h.index('MIN')]
            if isinstance(mins, str) and ':' in mins:
                mins = int(mins.split(':')[0]) + int(mins.split(':')[1])/60
            games.append({
                'stat': row[h.index(stat_col)],
                'minutes': float(mins) if mins else 0
            })
        
        return games
    
    return None


def analyze_game_log(games, line):
    if not games or len(games) < 5:
        return None
    
    values = np.array([g['stat'] for g in games if g['stat'] is not None], dtype=float)
    if len(values) < 5:
        return None
    
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    clean = values[(values >= q1-1.5*iqr) & (values <= q3+1.5*iqr)]
    clean_mean = np.mean(clean) if len(clean) > 0 else mean
    
    over_count = np.sum(values > line)
    total = len(values)
    over_prob = (over_count / total) * 100
    
    try:
        chi2, p_value = scipy_stats.chisquare([over_count, total-over_count], [total/2, total/2])
        chi_ok = p_value > 0.05
    except:
        chi2, p_value, chi_ok = 0, 1, True
    
    kelly = min(25, max(0, ((over_prob/100) - 0.524) / 0.91) * 100) if over_prob > 50 else 0
    
    return {
        'games_analyzed': len(values),
        'mean': round(mean, 1),
        'median': round(median, 1),
        'std': round(std, 2),
        'clean_mean': round(clean_mean, 1),
        'outliers_removed': len(values) - len(clean),
        'avg_last_5': round(np.mean(values[:5]), 1),
        'avg_last_10': round(np.mean(values[:10]), 1) if len(values) >= 10 else round(mean, 1),
        'trend': 'UP' if len(values) >= 10 and np.mean(values[:5]) > np.mean(values[5:10]) * 1.05 else 'STABLE',
        'consistency': round(max(0, 100 - (std / mean * 100)), 1) if mean > 0 else 50,
        'over_probability': round(over_prob, 1),
        'under_probability': round(100 - over_prob, 1),
        'over_count': int(over_count),
        'under_count': int(total - over_count),
        'chi_ok': chi_ok,
        'kelly_criterion': round(kelly, 1)
    }


def get_nba_games():
    try:
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american'
        }
        response = requests.get(f"{ODDS_BASE_URL}/sports/basketball_nba/odds", params=params, timeout=10)
        if response.status_code == 200:
            games = response.json()
            log_debug(f"Found {len(games)} games from Odds API")
            return games
    except Exception as e:
        log_debug(f"Odds API error: {e}")
    return []


def get_player_props(stat_type='points'):
    games = get_nba_games()
    if not games:
        return [], {}
    
    all_props = []
    game_info = {}
    
    market = {
        'points': 'player_points',
        'assists': 'player_assists',
        'rebounds': 'player_rebounds'
    }.get(stat_type, 'player_points')
    
    for game in games[:10]:
        gid = game['id']
        game_info[gid] = {
            'home_team': game.get('home_team', ''),
            'away_team': game.get('away_team', '')
        }
        
        try:
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'us',
                'markets': market,
                'oddsFormat': 'american'
            }
            response = requests.get(
                f"{ODDS_BASE_URL}/sports/basketball_nba/events/{gid}/odds",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                player_best = {}
                
                for bk in data.get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        for oc in mk.get('outcomes', []):
                            p = oc.get('description', '')
                            ln = oc.get('point', 0)
                            pr = oc.get('price', 0)
                            tp = oc.get('name', '')
                            
                            if p and ln:
                                if p not in player_best:
                                    player_best[p] = {
                                        'player': p,
                                        'game_id': gid,
                                        'stat_type': stat_type,
                                        'lines': []
                                    }
                                player_best[p]['lines'].append({
                                    'book': bk['key'],
                                    'line': ln,
                                    'price': pr,
                                    'type': tp
                                })
                
                all_props.extend(player_best.values())
        except:
            continue
    
    log_debug(f"Found {len(all_props)} {stat_type} props")
    return all_props, game_info


def quick_filter(props, player_stats, min_edge=2):
    candidates = []
    stat_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb'}
    
    for prop in props:
        name_norm = normalize_name(prop['player'])
        pd = player_stats.get(name_norm)
        
        if not pd:
            for k, v in player_stats.items():
                if name_norm.split()[-1] == k.split()[-1]:
                    pd = v
                    break
        
        if not pd:
            continue
        
        avg = pd.get(stat_map.get(prop['stat_type'], 'pts'), 0)
        gp = pd.get('gp', 0)
        pid = pd.get('id')
        
        if not avg or gp < 5 or not pid:
            continue
        
        overs = [l for l in prop['lines'] if l['type'] == 'Over']
        unders = [l for l in prop['lines'] if l['type'] == 'Under']
        
        if not overs:
            continue
        
        bo = min(overs, key=lambda x: x['line'])
        bu = max(unders, key=lambda x: x['line']) if unders else None
        
        oe = ((avg - bo['line']) / bo['line']) * 100
        ue = ((bu['line'] - avg) / bu['line']) * 100 if bu else 0
        
        if oe >= min_edge or ue >= min_edge:
            candidates.append({
                'prop': prop,
                'player_id': pid,
                'season_avg': avg,
                'games_played': gp,
                'best_over': bo,
                'best_under': bu,
                'over_edge': oe,
                'under_edge': ue
            })
    
    candidates.sort(key=lambda x: max(x['over_edge'], x['under_edge']), reverse=True)
    return candidates[:15]


def deep_analyze(candidates, game_info, min_edge=5):
    opps = []
    
    for c in candidates:
        games = get_player_game_log(c['player_id'], c['prop']['stat_type'])
        if not games:
            continue
        
        a = analyze_game_log(games, c['best_over']['line'])
        if not a:
            continue
        
        oe = ((a['clean_mean'] - c['best_over']['line']) / c['best_over']['line']) * 100
        ue = ((c['best_under']['line'] - a['clean_mean']) / c['best_under']['line']) * 100 if c['best_under'] else 0
        
        if a['over_probability'] > 52 and oe >= min_edge:
            rec, edge, bl = 'OVER', oe, c['best_over']
        elif a['under_probability'] > 52 and ue >= min_edge:
            rec, edge, bl = 'UNDER', ue, c['best_under']
        else:
            continue
        
        sc = 0
        sc += 2 if edge >= 10 else (1 if edge >= 7 else 0)
        sc += 2 if a['over_probability'] >= 65 or a['under_probability'] >= 65 else (1 if a['over_probability'] >= 55 else 0)
        sc += 1 if a['consistency'] >= 70 else 0
        sc += 1 if a['chi_ok'] else 0
        sc += 1 if c['games_played'] >= 30 else 0
        conf = 'HIGH' if sc >= 5 else ('MEDIUM' if sc >= 3 else 'LOW')
        
        opps.append({
            'player': c['prop']['player'],
            'stat_type': c['prop']['stat_type'],
            'game_info': game_info.get(c['prop']['game_id'], {}),
            'season_avg': round(c['season_avg'], 1),
            'games_played': c['games_played'],
            'line_analysis': {
                'bookmaker_line': bl['line'],
                'bookmaker': bl['book'].upper(),
                'odds': bl['price'],
                'recommendation': rec,
                'edge': round(edge, 1),
                'over_probability': a['over_probability'],
                'under_probability': a['under_probability'],
                'bet_confidence': conf,
                'kelly_criterion': a['kelly_criterion']
            },
            'deep_stats': {
                'mean': a['mean'],
                'clean_mean': a['clean_mean'],
                'median': a['median'],
                'std': a['std'],
                'avg_last_5': a['avg_last_5'],
                'avg_last_10': a['avg_last_10'],
                'trend': a['trend'],
                'consistency': a['consistency'],
                'games_analyzed': a['games_analyzed'],
                'outliers_removed': a['outliers_removed'],
                'over_count': a['over_count'],
                'under_count': a['under_count']
            }
        })
    
    opps.sort(key=lambda x: x['line_analysis']['edge'], reverse=True)
    return opps


@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    try:
        min_edge = float(request.args.get('min_edge', 5))
        min_conf = request.args.get('min_confidence', 'LOW')
        stat_type = request.args.get('stat_type', 'points')
        
        if stat_type not in ['points', 'assists', 'rebounds']:
            stat_type = 'points'
        
        log_debug(f"=== SCAN: {stat_type} ===")
        
        ps = get_all_player_averages()
        if not ps:
            return jsonify({
                'status': 'ERROR',
                'message': 'Could not fetch NBA stats - NBA API may be blocking cloud requests',
                'debug_log': DEBUG_LOG[-15:]
            }), 500
        
        props, gi = get_player_props(stat_type)
        if not props:
            return jsonify({
                'status': 'SUCCESS',
                'message': 'No props available',
                'opportunities': [],
                'total_props': 0,
                'debug_log': DEBUG_LOG[-15:]
            })
        
        cands = quick_filter(props, ps, 2)
        if not cands:
            return jsonify({
                'status': 'SUCCESS',
                'message': 'No candidates matched filters',
                'opportunities': [],
                'total_props': len(props),
                'players_in_db': len(ps),
                'debug_log': DEBUG_LOG[-15:]
            })
        
        opps = deep_analyze(cands, gi, min_edge)
        
        cl = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        filtered = [o for o in opps if cl.get(o['line_analysis']['bet_confidence'], 0) >= cl.get(min_conf, 1)]
        
        return jsonify({
            'status': 'SUCCESS',
            'stat_type': stat_type,
            'total_props': len(props),
            'players_in_db': len(ps),
            'candidates_analyzed': len(cands),
            'opportunities_found': len(filtered),
            'opportunities': filtered,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'debug_log': DEBUG_LOG[-15:]
        })
        
    except Exception as e:
        log_debug(f"ERROR: {e}")
        return jsonify({
            'status': 'ERROR',
            'message': str(e),
            'debug_log': DEBUG_LOG[-15:]
        }), 500


@app.route('/api/debug', methods=['GET'])
def debug_endpoint():
    r = {'timestamp': datetime.now().isoformat(), 'tests': {}}
    
    ps = get_all_player_averages()
    r['tests']['nba_api'] = {
        'success': len(ps) > 0,
        'players_found': len(ps),
        'sample': list(ps.keys())[:5]
    }
    
    games = get_nba_games()
    r['tests']['odds_api_games'] = {
        'success': len(games) > 0,
        'games_found': len(games),
        'games': [{'home': g.get('home_team'), 'away': g.get('away_team')} for g in games[:3]]
    }
    
    if games:
        props, _ = get_player_props('points')
        r['tests']['odds_api_props'] = {
            'success': len(props) > 0,
            'props_found': len(props),
            'sample_players': [p['player'] for p in props[:5]]
        }
    
    r['debug_log'] = DEBUG_LOG[-20:]
    return jsonify(r)


@app.route('/api/odds/usage', methods=['GET'])
def get_usage():
    try:
        r = requests.get(f"{ODDS_BASE_URL}/sports", params={'apiKey': ODDS_API_KEY}, timeout=10)
        return jsonify({
            'used': r.headers.get('x-requests-used', 'N/A'),
            'remaining': r.headers.get('x-requests-remaining', 'N/A')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'version': '8.3', 'season': '2024-25'})


@app.route('/')
def home():
    return jsonify({'app': 'NBA Betting Analyzer', 'version': '8.3', 'season': '2024-25 only'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)