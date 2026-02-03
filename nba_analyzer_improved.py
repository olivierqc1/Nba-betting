"""
NBA Betting Analyzer v8.5.3
- Uses balldontlie.io API (no cloud IP blocking!)
- Optimized rate limiting + star player priority
- Full stats: mean, std, R², chi², consistency
- Season 2025-26 data
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

# 2025-26 Defensive Ratings (points allowed per 100 possessions)
DEFENSIVE_RATINGS = {
    'Oklahoma City Thunder': 105.8, 'OKC': 105.8,
    'Cleveland Cavaliers': 107.2, 'CLE': 107.2,
    'Boston Celtics': 108.1, 'BOS': 108.1,
    'Houston Rockets': 108.5, 'HOU': 108.5,
    'Memphis Grizzlies': 109.2, 'MEM': 109.2,
    'Orlando Magic': 109.4, 'ORL': 109.4,
    'Minnesota Timberwolves': 110.1, 'MIN': 110.1,
    'Denver Nuggets': 110.4, 'DEN': 110.4,
    'New York Knicks': 110.8, 'NYK': 110.8,
    'Los Angeles Lakers': 111.2, 'LAL': 111.2,
    'Miami Heat': 111.5, 'MIA': 111.5,
    'Golden State Warriors': 111.8, 'GSW': 111.8,
    'Milwaukee Bucks': 112.0, 'MIL': 112.0,
    'Sacramento Kings': 112.3, 'SAC': 112.3,
    'Dallas Mavericks': 112.5, 'DAL': 112.5,
    'Phoenix Suns': 112.8, 'PHX': 112.8,
    'Los Angeles Clippers': 113.0, 'LAC': 113.0,
    'Indiana Pacers': 113.2, 'IND': 113.2,
    'Philadelphia 76ers': 113.4, 'PHI': 113.4,
    'New Orleans Pelicans': 113.6, 'NOP': 113.6,
    'Toronto Raptors': 113.8, 'TOR': 113.8,
    'Brooklyn Nets': 114.2, 'BKN': 114.2,
    'Chicago Bulls': 114.5, 'CHI': 114.5,
    'Detroit Pistons': 114.8, 'DET': 114.8,
    'Atlanta Hawks': 115.0, 'ATL': 115.0,
    'San Antonio Spurs': 115.3, 'SAS': 115.3,
    'Portland Trail Blazers': 115.6, 'POR': 115.6,
    'Utah Jazz': 115.9, 'UTA': 115.9,
    'Charlotte Hornets': 116.2, 'CHA': 116.2,
    'Washington Wizards': 117.5, 'WAS': 117.5,
}


def get_defense_rating(team_name):
    if not team_name:
        return 112.0
    if team_name in DEFENSIVE_RATINGS:
        return DEFENSIVE_RATINGS[team_name]
    for key, rating in DEFENSIVE_RATINGS.items():
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return rating
    return 112.0


def get_defense_category(rating):
    if rating < 110:
        return {'category': 'ELITE', 'emoji': '🛡️', 'impact': -2.5}
    elif rating < 112:
        return {'category': 'GOOD', 'emoji': '🔒', 'impact': -1.0}
    elif rating < 114:
        return {'category': 'AVERAGE', 'emoji': '➖', 'impact': 0}
    elif rating < 116:
        return {'category': 'WEAK', 'emoji': '🎯', 'impact': +1.5}
    else:
        return {'category': 'POOR', 'emoji': '🔥', 'impact': +3.0}


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


def bdl_request(endpoint, params=None, retries=3):
    headers = {}
    if BALLDONTLIE_API_KEY:
        headers['Authorization'] = BALLDONTLIE_API_KEY
    
    for attempt in range(retries):
        try:
            url = f"{BDL_BASE_URL}/{endpoint}"
            if attempt == 0:
                log_debug(f"BDL request: {endpoint}")
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                time.sleep(0.8)
                return response.json()
            elif response.status_code == 401:
                log_debug("BDL API: Missing or invalid API key")
                return None
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 3
                log_debug(f"BDL API: Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                log_debug(f"BDL API error: {response.status_code}")
                return None
        except Exception as e:
            log_debug(f"BDL request error: {str(e)[:50]}")
            time.sleep(2)
    return None


def search_player(name):
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
    data = bdl_request("season_averages", {"season": 2025, "player_ids[]": player_id})
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
    stat_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb'}
    stat_col = stat_map.get(stat_type, 'pts')
    data = bdl_request("stats", {"player_ids[]": player_id, "seasons[]": 2025, "per_page": 30})
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
    
    # Calculate R² (trend analysis)
    x = np.arange(len(values))
    try:
        slope, intercept, r_value, p_value_trend, std_err = scipy_stats.linregress(x, values)
        r_squared = round(r_value ** 2, 3)
        trend_slope = round(slope, 3)
    except:
        r_squared = 0
        trend_slope = 0
        p_value_trend = 1
    
    over_count = np.sum(values > line)
    total = len(values)
    over_prob = (over_count / total) * 100
    
    try:
        chi2, p_value = scipy_stats.chisquare([over_count, total-over_count], [total/2, total/2])
        chi_ok = p_value > 0.05
    except:
        chi2, p_value, chi_ok = 0, 1, True
    
    kelly = min(25, max(0, ((over_prob/100) - 0.524) / 0.91) * 100) if over_prob > 50 else 0
    
    # Determine trend direction
    if len(values) >= 10:
        recent_5 = np.mean(values[:5])
        prev_5 = np.mean(values[5:10])
        if recent_5 > prev_5 * 1.05:
            trend_dir = 'UP ↑'
        elif recent_5 < prev_5 * 0.95:
            trend_dir = 'DOWN ↓'
        else:
            trend_dir = 'STABLE →'
    else:
        trend_dir = 'STABLE →'
    
    return {
        'games_analyzed': len(values),
        'mean': round(mean, 1),
        'median': round(median, 1),
        'std': round(std, 2),
        'clean_mean': round(clean_mean, 1),
        'outliers_removed': len(values) - len(clean),
        'avg_last_5': round(np.mean(values[:5]), 1),
        'avg_last_10': round(np.mean(values[:10]), 1) if len(values) >= 10 else round(mean, 1),
        'min_val': round(float(np.min(values)), 1),
        'max_val': round(float(np.max(values)), 1),
        'r_squared': r_squared,
        'trend_slope': trend_slope,
        'trend': trend_dir,
        'consistency': round(max(0, 100 - (std / mean * 100)), 1) if mean > 0 else 50,
        'over_probability': round(over_prob, 1),
        'under_probability': round(100 - over_prob, 1),
        'over_count': int(over_count),
        'under_count': int(total - over_count),
        'chi_ok': chi_ok,
        'chi_p_value': round(p_value, 4),
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
    market = {'points': 'player_points', 'assists': 'player_assists', 'rebounds': 'player_rebounds'}.get(stat_type, 'player_points')
    for game in games[:10]:
        gid = game['id']
        game_info[gid] = {'home_team': game.get('home_team', ''), 'away_team': game.get('away_team', '')}
        try:
            params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': market, 'oddsFormat': 'american'}
            response = requests.get(f"{ODDS_BASE_URL}/sports/basketball_nba/events/{gid}/odds", params=params, timeout=10)
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
                                    player_best[p] = {'player': p, 'game_id': gid, 'stat_type': stat_type, 'lines': []}
                                player_best[p]['lines'].append({'book': bk['key'], 'line': ln, 'price': pr, 'type': tp})
                all_props.extend(player_best.values())
        except:
            continue
    log_debug(f"Found {len(all_props)} {stat_type} props")
    return all_props, game_info


def analyze_player_prop(player_name, stat_type, line):
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
        'team': player_info.get('team', 'N/A'),
        'season_avg': season_avg.get(stat_key, 0) if season_avg else analysis['mean'],
        'games_played': season_avg.get('gp', len(games)) if season_avg else len(games),
        'analysis': analysis
    }


# ============================================================
# PART 1 END - Continue with PART 2
# ============================================================# ============================================================
# PART 2 - Paste this DIRECTLY after Part 1 (no blank lines)
# ============================================================

def deep_analyze_props(props, game_info, stat_type, min_edge=5):
    opps = []
    
    # Star players list - prioritize these
    star_players = [
        'lebron james', 'stephen curry', 'kevin durant', 'giannis antetokounmpo',
        'luka doncic', 'nikola jokic', 'joel embiid', 'jayson tatum', 'damian lillard',
        'anthony davis', 'devin booker', 'donovan mitchell', 'jimmy butler', 'kyrie irving',
        'paul george', 'kawhi leonard', 'trae young', 'ja morant', 'zion williamson',
        'anthony edwards', 'tyrese haliburton', 'shai gilgeous-alexander', 'lamelo ball',
        'de\'aaron fox', 'darius garland', 'cade cunningham', 'paolo banchero',
        'brandon ingram', 'julius randle', 'bam adebayo', 'karl-anthony towns',
        'jaylen brown', 'jalen brunson', 'fred vanvleet', 'chet holmgren'
    ]
    
    def is_star(p):
        return normalize_name(p['player']) in star_players
    
    sorted_props = sorted(props, key=lambda p: (0 if is_star(p) else 1))
    
    analyzed_count = 0
    max_analyze = 12
    
    for prop in sorted_props:
        if analyzed_count >= max_analyze:
            log_debug(f"Reached max {max_analyze} players analyzed")
            break
            
        player_name = prop['player']
        overs = [l for l in prop['lines'] if l['type'] == 'Over']
        unders = [l for l in prop['lines'] if l['type'] == 'Under']
        if not overs:
            continue
        best_over = min(overs, key=lambda x: x['line'])
        best_under = max(unders, key=lambda x: x['line']) if unders else None
        line = best_over['line']
        result = analyze_player_prop(player_name, stat_type, line)
        
        analyzed_count += 1
        
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
        sc += 1 if a['r_squared'] >= 0.1 else 0
        conf = 'HIGH' if sc >= 5 else ('MEDIUM' if sc >= 3 else 'LOW')
        
        gi_data = game_info.get(prop['game_id'], {})
        home_team = gi_data.get('home_team', '')
        away_team = gi_data.get('away_team', '')
        opponent = away_team if home_team else away_team
        def_rating = get_defense_rating(opponent) if opponent else 112.0
        def_info = get_defense_category(def_rating)
        adjusted_edge = edge + def_info['impact']
        
        opps.append({
            'player': result['player_name'],
            'team': result.get('team', 'N/A'),
            'stat_type': stat_type,
            'game_info': gi_data,
            'season_avg': round(result['season_avg'], 1),
            'games_played': result['games_played'],
            'line_analysis': {
                'bookmaker_line': bl['line'],
                'bookmaker': bl['book'].upper(),
                'odds': bl['price'],
                'recommendation': rec,
                'edge': round(edge, 1),
                'adjusted_edge': round(adjusted_edge, 1),
                'over_probability': a['over_probability'],
                'under_probability': a['under_probability'],
                'bet_confidence': conf,
                'kelly_criterion': a['kelly_criterion']
            },
            'defense_analysis': {
                'opponent': opponent or 'Unknown',
                'def_rating': def_rating,
                'category': def_info['category'],
                'emoji': def_info['emoji'],
                'impact_pts': def_info['impact'],
                'note': f"{def_info['emoji']} {def_info['category']} defense ({def_rating})"
            },
            'deep_stats': {
                'mean': a['mean'],
                'clean_mean': a['clean_mean'],
                'median': a['median'],
                'std': a['std'],
                'min': a['min_val'],
                'max': a['max_val'],
                'avg_last_5': a['avg_last_5'],
                'avg_last_10': a['avg_last_10'],
                'r_squared': a['r_squared'],
                'trend_slope': a['trend_slope'],
                'trend': a['trend'],
                'consistency': a['consistency'],
                'games_analyzed': a['games_analyzed'],
                'outliers_removed': a['outliers_removed'],
                'over_count': a['over_count'],
                'under_count': a['under_count'],
                'chi_ok': a['chi_ok'],
                'chi_p_value': a['chi_p_value']
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
        log_debug(f"BDL key set: {bool(BALLDONTLIE_API_KEY)}")
        if not BALLDONTLIE_API_KEY:
            return jsonify({'status': 'ERROR', 'message': 'BALLDONTLIE_API_KEY not set', 'debug_log': DEBUG_LOG[-15:]}), 500
        props, gi = get_player_props(stat_type)
        if not props:
            return jsonify({'status': 'SUCCESS', 'message': 'No props available', 'opportunities': [], 'total_props': 0, 'debug_log': DEBUG_LOG[-15:]})
        opps = deep_analyze_props(props, gi, stat_type, min_edge)
        cl = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        filtered = [o for o in opps if cl.get(o['line_analysis']['bet_confidence'], 0) >= cl.get(min_conf, 1)]
        return jsonify({
            'status': 'SUCCESS',
            'stat_type': stat_type,
            'total_props': len(props),
            'players_in_db': len(PLAYER_ID_CACHE),
            'candidates_analyzed': min(12, len(props)),
            'opportunities_found': len(filtered),
            'opportunities': filtered,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'api_source': 'balldontlie.io',
            'season': '2025-26',
            'debug_log': DEBUG_LOG[-15:]
        })
    except Exception as e:
        log_debug(f"ERROR: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e), 'debug_log': DEBUG_LOG[-15:]}), 500


@app.route('/api/debug', methods=['GET'])
def debug_endpoint():
    r = {'timestamp': datetime.now().isoformat(), 'tests': {}}
    r['env_check'] = {
        'BALLDONTLIE_API_KEY': bool(BALLDONTLIE_API_KEY),
        'ODDS_API_KEY': bool(ODDS_API_KEY),
        'bdl_key_length': len(BALLDONTLIE_API_KEY) if BALLDONTLIE_API_KEY else 0
    }
    if BALLDONTLIE_API_KEY:
        player = search_player("LeBron James")
        r['tests']['balldontlie'] = {'success': player is not None, 'api_key_set': True, 'test_player': player}
        if player:
            games = get_player_game_log(player['id'], 'points')
            r['tests']['game_log'] = {'success': games is not None, 'games_found': len(games) if games else 0}
    else:
        r['tests']['balldontlie'] = {'success': False, 'api_key_set': False, 'message': 'Set BALLDONTLIE_API_KEY env var'}
    games = get_nba_games()
    r['tests']['odds_api_games'] = {'success': len(games) > 0, 'games_found': len(games), 'games': [{'home': g.get('home_team'), 'away': g.get('away_team')} for g in games[:3]]}
    if games:
        props, _ = get_player_props('points')
        r['tests']['odds_api_props'] = {'success': len(props) > 0, 'props_found': len(props), 'sample_players': [p['player'] for p in props[:5]]}
    r['debug_log'] = DEBUG_LOG[-20:]
    return jsonify(r)


@app.route('/api/odds/usage', methods=['GET'])
def get_usage():
    try:
        r = requests.get(f"{ODDS_BASE_URL}/sports", params={'apiKey': ODDS_API_KEY}, timeout=10)
        return jsonify({'used': r.headers.get('x-requests-used', 'N/A'), 'remaining': r.headers.get('x-requests-remaining', 'N/A')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'version': '8.5.3',
        'data_source': 'balldontlie.io',
        'season': '2025-26',
        'bdl_key_set': bool(BALLDONTLIE_API_KEY),
        'bdl_key_length': len(BALLDONTLIE_API_KEY) if BALLDONTLIE_API_KEY else 0
    })


@app.route('/')
def home():
    return jsonify({
        'app': 'NBA Betting Analyzer',
        'version': '8.5.3',
        'season': '2025-26',
        'data_source': 'balldontlie.io',
        'bdl_key_set': bool(BALLDONTLIE_API_KEY),
        'setup': 'Set BALLDONTLIE_API_KEY env var'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting NBA Analyzer v8.5.3")
    print(f"BALLDONTLIE_API_KEY set: {bool(BALLDONTLIE_API_KEY)}")
    print(f"ODDS_API_KEY set: {bool(ODDS_API_KEY)}")
    app.run(host='0.0.0.0', port=port, debug=False)