"""
NBA Betting Analyzer v8.5.5
- Fixed ALL numpy type conversions
- Ultra-defensive JSON serialization
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

ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
BALLDONTLIE_API_KEY = os.environ.get('BALLDONTLIE_API_KEY', '')

ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
BDL_BASE_URL = "https://api.balldontlie.io/v1"

PLAYER_ID_CACHE = {}
DEBUG_LOG = []

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
    'Brooklyn Nets': 113.6, 'BKN': 113.6,
    'San Antonio Spurs': 114.0, 'SAS': 114.0,
    'Toronto Raptors': 114.3, 'TOR': 114.3,
    'Chicago Bulls': 114.5, 'CHI': 114.5,
    'Atlanta Hawks': 114.8, 'ATL': 114.8,
    'New Orleans Pelicans': 115.0, 'NOP': 115.0,
    'Portland Trail Blazers': 115.3, 'POR': 115.3,
    'Detroit Pistons': 115.5, 'DET': 115.5,
    'Charlotte Hornets': 115.8, 'CHA': 115.8,
    'Utah Jazz': 116.0, 'UTA': 116.0,
    'Washington Wizards': 116.5, 'WAS': 116.5
}


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return True if obj else False
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def log_debug(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    DEBUG_LOG.append(f"[{ts}] {msg}")
    if len(DEBUG_LOG) > 100:
        DEBUG_LOG.pop(0)


def normalize_name(name):
    return name.lower().strip().replace('.', '').replace("'", "")


def get_defense_rating(team):
    for k, v in DEFENSIVE_RATINGS.items():
        if k.lower() in team.lower() or team.lower() in k.lower():
            return v
    return 112.0


def get_defense_category(rating):
    if rating < 108:
        return {'category': 'ELITE', 'emoji': '🔒', 'impact': -2}
    elif rating < 111:
        return {'category': 'GOOD', 'emoji': '🛡️', 'impact': -1}
    elif rating < 114:
        return {'category': 'AVERAGE', 'emoji': '⚖️', 'impact': 0}
    elif rating < 116:
        return {'category': 'POOR', 'emoji': '🎯', 'impact': 1}
    return {'category': 'BAD', 'emoji': '🔥', 'impact': 2}


def bdl_request(endpoint, params=None, retries=3):
    headers = {}
    if BALLDONTLIE_API_KEY:
        headers['Authorization'] = BALLDONTLIE_API_KEY
    for attempt in range(retries):
        try:
            url = f"{BDL_BASE_URL}/{endpoint}"
            log_debug(f"BDL request: {endpoint}")
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                time.sleep(0.5)
                return response.json()
            elif response.status_code == 429:
                wait = 3 * (attempt + 1)
                log_debug(f"BDL API: Rate limited, waiting {wait}s...")
                time.sleep(wait)
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
            team_data = player.get('team')
            team_abbr = 'N/A'
            if team_data and isinstance(team_data, dict):
                team_abbr = team_data.get('abbreviation', 'N/A')
            PLAYER_ID_CACHE[p_norm] = {
                'id': int(player['id']),
                'name': str(full_name),
                'team': str(team_abbr)
            }
            if p_norm == norm_name or norm_name in p_norm or p_norm in norm_name:
                log_debug(f"Found player: {full_name} (ID: {player['id']})")
                return PLAYER_ID_CACHE[p_norm]
    return None


def get_player_season_avg(player_id):
    data = bdl_request("season_averages", {"season": 2024, "player_id": int(player_id)})
    if data and 'data' in data and len(data['data']) > 0:
        stats = data['data'][0]
        return {
            'pts': float(stats.get('pts') or 0),
            'ast': float(stats.get('ast') or 0),
            'reb': float(stats.get('reb') or 0),
            'gp': int(stats.get('games_played') or 0),
            'min': str(stats.get('min') or '0')
        }
    return None


def get_player_game_log(player_id, stat_type='points'):
    stat_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb'}
    stat_col = stat_map.get(stat_type, 'pts')
    data = bdl_request("stats", {"player_ids[]": int(player_id), "seasons[]": 2024, "per_page": 30})
    if not data or 'data' not in data:
        return None
    games = []
    for game in data['data']:
        stat_val = game.get(stat_col)
        if stat_val is None:
            continue
        mins = game.get('min', '0') or '0'
        if isinstance(mins, str) and ':' in mins:
            parts = mins.split(':')
            mins = int(parts[0]) + int(parts[1])/60
        elif isinstance(mins, str):
            try:
                mins = float(mins)
            except:
                mins = 0
        game_data = game.get('game', {})
        games.append({
            'stat': float(stat_val),
            'minutes': float(mins),
            'date': str(game_data.get('date', '') or '')
        })
    games.sort(key=lambda x: x['date'], reverse=True)
    log_debug(f"Got {len(games)} games for player {player_id}")
    return games if games else None


def analyze_game_log(games, line):
    if not games or len(games) < 5:
        return None
    values = [float(g['stat']) for g in games if g['stat'] is not None]
    if len(values) < 5:
        return None
    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    median_val = float(np.median(arr))
    q1_val = float(np.percentile(arr, 25))
    q3_val = float(np.percentile(arr, 75))
    iqr = q3_val - q1_val
    clean = arr[(arr >= q1_val-1.5*iqr) & (arr <= q3_val+1.5*iqr)]
    clean_mean = float(np.mean(clean)) if len(clean) > 0 else mean_val
    x = np.arange(len(arr))
    try:
        slope, intercept, r_value, p_value_trend, std_err = scipy_stats.linregress(x, arr)
        r_squared = round(float(r_value ** 2), 3)
        trend_slope = round(float(slope), 3)
    except:
        r_squared = 0.0
        trend_slope = 0.0
    over_count = int(np.sum(arr > line))
    total = len(values)
    over_prob = (over_count / total) * 100
    try:
        chi2, p_val = scipy_stats.chisquare([over_count, total-over_count], [total/2, total/2])
        chi_ok = True if float(p_val) > 0.05 else False
        p_val = float(p_val)
    except:
        chi_ok = True
        p_val = 1.0
    kelly = min(25, max(0, ((over_prob/100) - 0.524) / 0.91) * 100) if over_prob > 50 else 0
    if len(values) >= 10:
        recent_5 = float(np.mean(arr[:5]))
        prev_5 = float(np.mean(arr[5:10]))
        if recent_5 > prev_5 * 1.05:
            trend_dir = 'UP'
        elif recent_5 < prev_5 * 0.95:
            trend_dir = 'DOWN'
        else:
            trend_dir = 'STABLE'
    else:
        trend_dir = 'STABLE'
    return {
        'games_analyzed': int(total),
        'mean': round(mean_val, 1),
        'median': round(median_val, 1),
        'std': round(std_val, 2),
        'clean_mean': round(clean_mean, 1),
        'outliers_removed': int(len(values) - len(clean)),
        'avg_last_5': round(float(np.mean(arr[:5])), 1),
        'avg_last_10': round(float(np.mean(arr[:10])), 1) if len(values) >= 10 else round(mean_val, 1),
        'min_val': round(float(np.min(arr)), 1),
        'max_val': round(float(np.max(arr)), 1),
        'r_squared': r_squared,
        'trend_slope': trend_slope,
        'trend': str(trend_dir),
        'consistency': round(max(0, 100 - (std_val / mean_val * 100)), 1) if mean_val > 0 else 50.0,
        'over_probability': round(over_prob, 1),
        'under_probability': round(100 - over_prob, 1),
        'over_count': int(over_count),
        'under_count': int(total - over_count),
        'chi_ok': chi_ok,
        'chi_p_value': round(p_val, 4),
        'kelly_criterion': round(float(kelly), 1)
    }


def get_nba_games():
    try:
        params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
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
        game_info[gid] = {'home_team': str(game.get('home_team', '')), 'away_team': str(game.get('away_team', ''))}
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
                                    player_best[p] = {'player': str(p), 'game_id': gid, 'stat_type': stat_type, 'lines': []}
                                player_best[p]['lines'].append({'book': str(bk['key']), 'line': float(ln), 'price': int(pr), 'type': str(tp)})
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
    season_val = float(season_avg.get(stat_key, 0)) if season_avg else float(analysis['mean'])
    games_played = int(season_avg.get('gp', len(games))) if season_avg else len(games)
    return {
        'player_id': int(player_id),
        'player_name': str(player_info['name']),
        'team': str(player_info.get('team', 'N/A')),
        'season_avg': season_val,
        'games_played': games_played,
        'analysis': analysis
    }def deep_analyze_props(props, game_info, stat_type, min_edge=5):
    opps = []
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
    max_analyze = 15
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
        opp = {
            'player': str(result['player_name']),
            'team': str(result.get('team', 'N/A')),
            'stat_type': str(stat_type),
            'game_info': {'home_team': str(gi_data.get('home_team', '')), 'away_team': str(gi_data.get('away_team', ''))},
            'season_avg': round(float(result['season_avg']), 1),
            'games_played': int(result['games_played']),
            'line_analysis': {
                'bookmaker_line': float(bl['line']),
                'bookmaker': str(bl['book']).upper(),
                'odds': int(bl['price']),
                'recommendation': str(rec),
                'edge': round(float(edge), 1),
                'adjusted_edge': round(float(adjusted_edge), 1),
                'over_probability': float(a['over_probability']),
                'under_probability': float(a['under_probability']),
                'bet_confidence': str(conf),
                'kelly_criterion': float(a['kelly_criterion'])
            },
            'defense_analysis': {
                'opponent': str(opponent) if opponent else 'Unknown',
                'def_rating': float(def_rating),
                'category': str(def_info['category']),
                'emoji': str(def_info['emoji']),
                'impact_pts': int(def_info['impact']),
                'note': f"{def_info['emoji']} {def_info['category']} defense ({def_rating})"
            },
            'deep_stats': {
                'mean': float(a['mean']),
                'clean_mean': float(a['clean_mean']),
                'median': float(a['median']),
                'std': float(a['std']),
                'min': float(a['min_val']),
                'max': float(a['max_val']),
                'avg_last_5': float(a['avg_last_5']),
                'avg_last_10': float(a['avg_last_10']),
                'r_squared': float(a['r_squared']),
                'trend_slope': float(a['trend_slope']),
                'trend': str(a['trend']),
                'consistency': float(a['consistency']),
                'games_analyzed': int(a['games_analyzed']),
                'outliers_removed': int(a['outliers_removed']),
                'over_count': int(a['over_count']),
                'under_count': int(a['under_count']),
                'chi_ok': True if a['chi_ok'] else False,
                'chi_p_value': float(a['chi_p_value'])
            }
        }
        opps.append(opp)
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
        log_debug(f"=== SCAN v8.5.5: {stat_type} ===")
        if not BALLDONTLIE_API_KEY:
            return jsonify({'status': 'ERROR', 'message': 'BALLDONTLIE_API_KEY not set'}), 500
        props, gi = get_player_props(stat_type)
        if not props:
            return jsonify({'status': 'SUCCESS', 'message': 'No props', 'opportunities': [], 'total_props': 0})
        opps = deep_analyze_props(props, gi, stat_type, min_edge)
        cl = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        filtered = [o for o in opps if cl.get(o['line_analysis']['bet_confidence'], 0) >= cl.get(min_conf, 1)]
        result = {
            'status': 'SUCCESS',
            'stat_type': str(stat_type),
            'total_props': int(len(props)),
            'players_in_db': int(len(PLAYER_ID_CACHE)),
            'candidates_analyzed': int(min(15, len(props))),
            'opportunities_found': int(len(filtered)),
            'opportunities': filtered,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'season': '2024-25',
            'debug_log': DEBUG_LOG[-15:]
        }
        return jsonify(to_python(result))
    except Exception as e:
        import traceback
        log_debug(f"ERROR: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e), 'debug_log': DEBUG_LOG[-20:]}), 500


@app.route('/api/debug', methods=['GET'])
def debug_endpoint():
    r = {'timestamp': datetime.now().isoformat(), 'version': '8.5.5', 'tests': {}}
    r['env_check'] = {
        'BALLDONTLIE_API_KEY': True if BALLDONTLIE_API_KEY else False,
        'ODDS_API_KEY': True if ODDS_API_KEY else False,
        'bdl_key_length': int(len(BALLDONTLIE_API_KEY)) if BALLDONTLIE_API_KEY else 0
    }
    if BALLDONTLIE_API_KEY:
        player = search_player("LeBron James")
        r['tests']['balldontlie'] = {'success': True if player else False, 'test_player': player}
        if player:
            games = get_player_game_log(player['id'], 'points')
            r['tests']['game_log'] = {'success': True if games else False, 'games_found': int(len(games)) if games else 0}
    games = get_nba_games()
    r['tests']['odds_api'] = {'success': True if len(games) > 0 else False, 'games_found': int(len(games))}
    r['debug_log'] = DEBUG_LOG[-20:]
    return jsonify(to_python(r))


@app.route('/api/odds/usage', methods=['GET'])
def get_usage():
    try:
        r = requests.get(f"{ODDS_BASE_URL}/sports", params={'apiKey': ODDS_API_KEY}, timeout=10)
        return jsonify({'used': str(r.headers.get('x-requests-used', 'N/A')), 'remaining': str(r.headers.get('x-requests-remaining', 'N/A'))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'version': '8.5.5',
        'season': '2024-25',
        'bdl_key_set': True if BALLDONTLIE_API_KEY else False
    })


@app.route('/')
def home():
    return jsonify({
        'app': 'NBA Betting Analyzer',
        'version': '8.5.5',
        'season': '2024-25',
        'bdl_key_set': True if BALLDONTLIE_API_KEY else False
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting NBA Analyzer v8.5.5")
    app.run(host='0.0.0.0', port=port, debug=False)