"""
NBA Betting Analyzer v8.5
- Uses balldontlie.io API (no cloud IP blocking!)
- Requires free API key from https://api.balldontlie.io
- Season 2024-25 data (balldontlie format)
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime
import os
import numpy as np
from scipy import stats as scipy_stats
import time

app = Flask(__name__)
CORS(app)

# API Keys
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
BALLDONTLIE_API_KEY = os.environ.get('BALLDONTLIE_API_KEY', '')

ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
BDL_BASE_URL = "https://api.balldontlie.io/v1"

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


def bdl_request(endpoint, params=None):
    """Make request to balldontlie API"""
    headers = {}
    if BALLDONTLIE_API_KEY:
        headers['Authorization'] = BALLDONTLIE_API_KEY
    
    try:
        url = f"{BDL_BASE_URL}/{endpoint}"
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            log_debug("BDL API: Missing or invalid API key")
        elif response.status_code == 429:
            log_debug("BDL API: Rate limited, waiting...")
            time.sleep(2)
        else:
            log_debug(f"BDL API error: {response.status_code}")
            
    except Exception as e:
        log_debug(f"BDL request error: {str(e)[:50]}")
    
    return None


def search_player(name):
    """Search for a player by name in balldontlie"""
    norm_name = normalize_name(name)
    if norm_name in PLAYER_ID_CACHE:
        return PLAYER_ID_CACHE[norm_name]
    
    search_term = name.split()[-1] if ' ' in name else name
    
    data = bdl_request("players", {"search": search_term, "per_page": 25})
    
    if data and 'data' in data:
        for player in data['data']:
            full_name = f"{player['first_name']} {player['last_name']}"
            p_norm = normalize_name(full_name)
            PLAYER_ID_CACHE[p_norm] = {
                'id': player['id'],
                'name': full_name,
                'team': player.get('team', {}).get('abbreviation', 'N/A')
            }
            
            if p_norm == norm_name or norm_name in p_norm or p_norm in norm_name:
                log_debug(f"Found player: {full_name} (ID: {player['id']})")
                return PLAYER_ID_CACHE[p_norm]
    
    return None


def get_player_season_avg(player_id):
    """Get season averages for a player"""
    data = bdl_request("season_averages", {
        "season": 2024,
        "player_ids[]": player_id
    })
    
    if data and 'data' in data and len(data['data']) > 0:
        stats = data['data'][0]
        return {
            'pts': stats.get('pts', 0),
            'ast': stats.get('ast', 0),
            'reb': stats.get('reb', 0),
            'gp': stats.get('games_played', 0),
            'min': stats.get('min', '0')
        }
    
    return None


def get_player_game_log(player_id, stat_type='points'):
    """Get recent games for a player"""
    stat_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb'}
    stat_col = stat_map.get(stat_type, 'pts')
    
    data = bdl_request("stats", {
        "player_ids[]": player_id,
        "seasons[]": 2024,
        "per_page": 30
    })
    
    if not data or 'data' not in data:
        return None
    
    games = []
    for game in data['data']:
        stat_val = game.get(stat_col, 0)
        mins = game.get('min', '0')
        
        if isinstance(mins, str) and ':' in mins:
            parts = mins.split(':')
            mins = int(parts[0]) + int(parts[1])/60
        elif isinstance(mins, str):
            mins = float(mins) if mins else 0
        
        if stat_val is not None:
            games.append({
                'stat': float(stat_val),
                'minutes': float(mins) if mins else 0,
                'date': game.get('game', {}).get('date', '')
            })
    
    games.sort(key=lambda x: x['date'], reverse=True)
    
    log_debug(f"Got {len(games)} games for player {player_id}")
    return games if games else None


def analyze_game_log(games, line):
    """Analyze game log and return stats"""
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
    """Get NBA games from Odds API"""
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
    """Get player props from Odds API"""
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


def analyze_player_prop(player_name, stat_type, line):
    """Analyze a single player prop using balldontlie data"""
    
    player_info = search_player(player_name)
    if not player_info:
        log_debug(f"Player not found: {player_name}")
        return None
    
    player_id = player_info['id']
    
    games = get_player_game_log(player_id, stat_type)
    if not games:
        log_debug(f"No games found for {player_name}")
        return None
    
    season_avg = get_player_season_avg(player_id)
    
    analysis = analyze_game_log(games, line)
    if not analysis:
        return None
    
    stat_key = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb'}.get(stat_type, 'pts')
    
    return {
        'player_id': player_id,
        'player_name': player_info['name'],
        'season_avg': season_avg.get(stat_key, 0) if season_avg else analysis['mean'],
        'games_played': season_avg.get('gp', len(games)) if season_avg else len(games),
        'analysis': analysis
    }


def deep_analyze_props(props, game_info, stat_type, min_edge=5):
    """Deep analyze props using balldontlie data"""
    opps = []
    
    for prop in props[:20]:
        player_name = prop['player']
        
        overs = [l for l in prop['lines'] if l['type'] == 'Over']
        unders = [l for l in prop['lines'] if l['type'] == 'Under']
        
        if not overs:
            continue
        
        best_over = min(overs, key=lambda x: x['line'])
        best_under = max(unders, key=lambda x: x['line']) if unders else None
        line = best_over['line']
        
        result = analyze_player_prop(player_name, stat_type, line)
        if not result:
            continue
        
        a = result['analysis']
        
        oe = ((a['clean_mean'] - line) / line) * 100
        ue = ((line - a['clean_mean']) / line) * 100 if best_under else 0
        
        if a['over_probability'] > 52 and oe >= min_edge:
            rec, edge, bl = 'OVER', oe, best_over
        elif a['under_probability'] > 52 and ue >= min_edge:
            rec, edge, bl = 'UNDER', ue, best_under
        else:
            continue
        
        sc = 0
        sc += 2 if edge >= 10 else (1 if edge >= 7 else 0)
        sc += 2 if a['over_probability'] >= 65 or a['under_probability'] >= 65 else (1 if a['over_probability'] >= 55 else 0)
        sc += 1 if a['consistency'] >= 70 else 0
        sc += 1 if a['chi_ok'] else 0
        sc += 1 if result['games_played'] >= 20 else 0
        conf = 'HIGH' if sc >= 5 else ('MEDIUM' if sc >= 3 else 'LOW')
        
        opps.append({
            'player': result['player_name'],
            'stat_type': stat_type,
            'game_info': game_info.get(prop['game_id'], {}),
            'season_avg': round(result['season_avg'], 1),
            'games_played': result['games_played'],
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
        
        time.sleep(0.4)
    
    opps.sort(key=lambda x: x['line_analysis']['edge'], reverse=True)
    return opps


@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    """Main endpoint - scan for betting opportunities"""
    try:
        min_edge = float(request.args.get('min_edge', 5))
        min_conf = request.args.get('min_confidence', 'LOW')
        stat_type = request.args.get('stat_type', 'points')
        
        if stat_type not in ['points', 'assists', 'rebounds']:
            stat_type = 'points'
        
        log_debug(f"=== SCAN: {stat_type} ===")
        
        if not BALLDONTLIE_API_KEY:
            return jsonify({
                'status': 'ERROR',
                'message': 'BALLDONTLIE_API_KEY not set. Get free key at https://api.balldontlie.io',
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
        
        opps = deep_analyze_props(props, gi, stat_type, min_edge)
        
        cl = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        filtered = [o for o in opps if cl.get(o['line_analysis']['bet_confidence'], 0) >= cl.get(min_conf, 1)]
        
        return jsonify({
            'status': 'SUCCESS',
            'stat_type': stat_type,
            'total_props': len(props),
            'players_in_db': len(PLAYER_ID_CACHE),
            'candidates_analyzed': len(props),
            'opportunities_found': len(filtered),
            'opportunities': filtered,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'api_source': 'balldontlie.io',
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
    """Debug endpoint to test APIs"""
    r = {'timestamp': datetime.now().isoformat(), 'tests': {}}
    
    if BALLDONTLIE_API_KEY:
        player = search_player("LeBron James")
        r['tests']['balldontlie'] = {
            'success': player is not None,
            'api_key_set': True,
            'test_player': player
        }
        
        if player:
            games = get_player_game_log(player['id'], 'points')
            r['tests']['game_log'] = {
                'success': games is not None,
                'games_found': len(games) if games else 0
            }
    else:
        r['tests']['balldontlie'] = {
            'success': False,
            'api_key_set': False,
            'message': 'Set BALLDONTLIE_API_KEY env var. Get free at https://api.balldontlie.io'
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
    """Check Odds API usage"""
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
    return jsonify({
        'status': 'healthy',
        'version': '8.5',
        'data_source': 'balldontlie.io',
        'bdl_key_set': bool(BALLDONTLIE_API_KEY)
    })


@app.route('/')
def home():
    return jsonify({
        'app': 'NBA Betting Analyzer',
        'version': '8.5',
        'data_source': 'balldontlie.io (no cloud blocking!)',
        'setup': 'Set BALLDONTLIE_API_KEY env var'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)