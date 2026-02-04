"""
NBA Betting Analyzer v9.0
- Analyzes ALL players (no 15 limit)
- Added: rest days, pace, fatigue, adjusted probabilities
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime, timedelta
import os
import numpy as np
from scipy import stats as scipy_stats
import time
import math

app = Flask(__name__)
CORS(app)

ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
BALLDONTLIE_API_KEY = os.environ.get('BALLDONTLIE_API_KEY', '')

ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
BDL_BASE_URL = "https://api.balldontlie.io/v1"

PLAYER_ID_CACHE = {}
PLAYER_GAMES_CACHE = {}
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

TEAM_PACE = {
    'Indiana Pacers': 103.2, 'IND': 103.2,
    'Atlanta Hawks': 101.8, 'ATL': 101.8,
    'Milwaukee Bucks': 101.5, 'MIL': 101.5,
    'Sacramento Kings': 101.2, 'SAC': 101.2,
    'New Orleans Pelicans': 100.8, 'NOP': 100.8,
    'Utah Jazz': 100.5, 'UTA': 100.5,
    'Minnesota Timberwolves': 100.2, 'MIN': 100.2,
    'Denver Nuggets': 99.8, 'DEN': 99.8,
    'Golden State Warriors': 99.5, 'GSW': 99.5,
    'Boston Celtics': 99.2, 'BOS': 99.2,
    'Dallas Mavericks': 99.0, 'DAL': 99.0,
    'Phoenix Suns': 98.8, 'PHX': 98.8,
    'Los Angeles Lakers': 98.5, 'LAL': 98.5,
    'Oklahoma City Thunder': 98.2, 'OKC': 98.2,
    'Brooklyn Nets': 98.0, 'BKN': 98.0,
    'Chicago Bulls': 97.8, 'CHI': 97.8,
    'Toronto Raptors': 97.5, 'TOR': 97.5,
    'Houston Rockets': 97.2, 'HOU': 97.2,
    'Portland Trail Blazers': 97.0, 'POR': 97.0,
    'San Antonio Spurs': 96.8, 'SAS': 96.8,
    'Detroit Pistons': 96.5, 'DET': 96.5,
    'Washington Wizards': 96.2, 'WAS': 96.2,
    'Charlotte Hornets': 96.0, 'CHA': 96.0,
    'New York Knicks': 95.8, 'NYK': 95.8,
    'Los Angeles Clippers': 95.5, 'LAC': 95.5,
    'Miami Heat': 95.2, 'MIA': 95.2,
    'Cleveland Cavaliers': 95.0, 'CLE': 95.0,
    'Philadelphia 76ers': 94.8, 'PHI': 94.8,
    'Memphis Grizzlies': 94.5, 'MEM': 94.5,
    'Orlando Magic': 94.2, 'ORL': 94.2
}

LEAGUE_AVG_PACE = 98.5
LEAGUE_AVG_DEF_RATING = 112.0


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        return 0.0 if math.isnan(val) or math.isinf(val) else val
    elif isinstance(obj, float):
        return 0.0 if math.isnan(obj) or math.isinf(obj) else obj
    elif isinstance(obj, (np.bool_, bool)):
        return True if obj else False
    elif isinstance(obj, np.ndarray):
        return [to_python(v) for v in obj.tolist()]
    return obj


def log_debug(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    DEBUG_LOG.append(f"[{ts}] {msg}")
    if len(DEBUG_LOG) > 200:
        DEBUG_LOG.pop(0)


def normalize_name(name):
    return name.lower().strip().replace('.', '').replace("'", "")


def get_defense_rating(team):
    for k, v in DEFENSIVE_RATINGS.items():
        if k.lower() in team.lower() or team.lower() in k.lower():
            return v
    return LEAGUE_AVG_DEF_RATING


def get_team_pace(team):
    for k, v in TEAM_PACE.items():
        if k.lower() in team.lower() or team.lower() in k.lower():
            return v
    return LEAGUE_AVG_PACE


def get_defense_category(rating):
    if rating < 108:
        return {'category': 'ELITE', 'emoji': '🔒', 'impact': -2.5}
    elif rating < 111:
        return {'category': 'GOOD', 'emoji': '🛡️', 'impact': -1.0}
    elif rating < 114:
        return {'category': 'AVERAGE', 'emoji': '⚖️', 'impact': 0}
    elif rating < 116:
        return {'category': 'POOR', 'emoji': '🎯', 'impact': 1.5}
    else:
        return {'category': 'BAD', 'emoji': '🔥', 'impact': 2.5}


def get_pace_impact(opp_pace):
    diff = opp_pace - LEAGUE_AVG_PACE
    return round(diff / 2, 1)


def bdl_request(endpoint, params=None, retries=3):
    headers = {'Authorization': BALLDONTLIE_API_KEY} if BALLDONTLIE_API_KEY else {}
    for attempt in range(retries):
        try:
            response = requests.get(f"{BDL_BASE_URL}/{endpoint}", headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                time.sleep(0.3)
                return response.json()
            elif response.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                return None
        except:
            time.sleep(1)
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
            team_abbr = team_data.get('abbreviation', 'N/A') if team_data and isinstance(team_data, dict) else 'N/A'
            PLAYER_ID_CACHE[p_norm] = {
                'id': int(player['id']),
                'name': str(full_name),
                'team': str(team_abbr),
                'position': str(player.get('position', 'N/A'))
            }
            if p_norm == norm_name or norm_name in p_norm or p_norm in norm_name:
                return PLAYER_ID_CACHE[p_norm]
    return None


def get_player_game_log(player_id, stat_type='points'):
    cache_key = f"{player_id}_{stat_type}"
    if cache_key in PLAYER_GAMES_CACHE:
        return PLAYER_GAMES_CACHE[cache_key]
    stat_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb'}
    stat_col = stat_map.get(stat_type, 'pts')
    data = bdl_request("stats", {"player_ids[]": int(player_id), "seasons[]": 2025, "per_page": 30})
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
    PLAYER_GAMES_CACHE[cache_key] = games
    return games if games else None


def calculate_rest_days(games):
    if not games or len(games) < 1:
        return 2
    try:
        last_game = datetime.strptime(games[0]['date'][:10], '%Y-%m-%d')
        return min((datetime.now() - last_game).days, 7)
    except:
        return 2


def is_back_to_back(games):
    if not games:
        return False
    try:
        last_game = datetime.strptime(games[0]['date'][:10], '%Y-%m-%d')
        return (datetime.now() - last_game).days <= 1
    except:
        return False


def calculate_fatigue_factor(games):
    if not games or len(games) < 5:
        return 0
    avg_mins = np.mean([g['minutes'] for g in games[:5]])
    if avg_mins > 38:
        return -1.5
    elif avg_mins > 36:
        return -0.5
    elif avg_mins < 28:
        return 0.5
    return 0

def analyze_game_log(games, line, opponent=None):
    if not games or len(games) < 5:
        return None
    values = [float(g['stat']) for g in games if g['stat'] is not None]
    if len(values) < 5:
        return None
    arr = np.array(values, dtype=float)
    mean_val, std_val, median_val = float(np.mean(arr)), float(np.std(arr)), float(np.median(arr))
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q3 - q1
    clean = arr[(arr >= q1-1.5*iqr) & (arr <= q3+1.5*iqr)]
    clean_mean = float(np.mean(clean)) if len(clean) > 0 else mean_val
    clean_std = float(np.std(clean)) if len(clean) > 0 else std_val
    try:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(np.arange(len(arr)), arr)
        r_squared, trend_slope = round(float(r_value ** 2), 3), round(float(slope), 3)
    except:
        r_squared, trend_slope = 0.0, 0.0
    over_count, total = int(np.sum(arr > line)), len(values)
    over_prob = (over_count / total) * 100
    try:
        chi2, p_val = scipy_stats.chisquare([over_count, total-over_count], [total/2, total/2])
        chi_ok, p_val = float(p_val) > 0.05, float(p_val)
    except:
        chi_ok, p_val = True, 1.0
    rest_days, b2b, fatigue = calculate_rest_days(games), is_back_to_back(games), calculate_fatigue_factor(games)
    rest_impact = 1.0 if rest_days >= 3 else (0.5 if rest_days == 2 else (-1.5 if b2b else 0))
    opp_def_impact, opp_pace_impact = 0, 0
    if opponent:
        def_info = get_defense_category(get_defense_rating(opponent))
        opp_def_impact, opp_pace_impact = def_info['impact'], get_pace_impact(get_team_pace(opponent))
    total_adj = rest_impact + fatigue + opp_def_impact + opp_pace_impact
    adjusted_mean = clean_mean + total_adj
    adjusted_over_prob = (1 - scipy_stats.norm.cdf((line - adjusted_mean) / clean_std)) * 100 if clean_std > 0 else over_prob
    kelly = min(25, max(0, ((adjusted_over_prob/100) - 0.524) / 0.91 * 100)) if adjusted_over_prob > 52.4 else 0
    trend_dir = 'STABLE'
    if len(values) >= 10:
        recent_5, prev_5 = float(np.mean(arr[:5])), float(np.mean(arr[5:10]))
        trend_dir = 'UP' if recent_5 > prev_5 * 1.05 else ('DOWN' if recent_5 < prev_5 * 0.95 else 'STABLE')
    return {
        'games_analyzed': int(total), 'mean': round(mean_val, 1), 'median': round(median_val, 1),
        'std': round(std_val, 2), 'clean_mean': round(clean_mean, 1), 'clean_std': round(clean_std, 2),
        'adjusted_mean': round(adjusted_mean, 1), 'outliers_removed': int(len(values) - len(clean)),
        'avg_last_5': round(float(np.mean(arr[:5])), 1),
        'avg_last_10': round(float(np.mean(arr[:10])), 1) if len(values) >= 10 else round(mean_val, 1),
        'min_val': round(float(np.min(arr)), 1), 'max_val': round(float(np.max(arr)), 1),
        'r_squared': r_squared, 'trend_slope': trend_slope, 'trend': str(trend_dir),
        'consistency': round(max(0, 100 - (clean_std / clean_mean * 100)), 1) if clean_mean > 0 else 50.0,
        'over_probability': round(over_prob, 1), 'adjusted_over_probability': round(adjusted_over_prob, 1),
        'under_probability': round(100 - over_prob, 1), 'adjusted_under_probability': round(100 - adjusted_over_prob, 1),
        'over_count': int(over_count), 'under_count': int(total - over_count),
        'chi_ok': chi_ok, 'chi_p_value': round(p_val, 4), 'kelly_criterion': round(float(kelly), 1),
        'rest_days': rest_days, 'is_b2b': b2b, 'rest_impact': round(rest_impact, 1),
        'fatigue_impact': round(fatigue, 1), 'defense_impact': round(opp_def_impact, 1),
        'pace_impact': round(opp_pace_impact, 1), 'total_adjustment': round(total_adj, 1)
    }


def get_nba_games():
    try:
        params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h,spreads,totals', 'oddsFormat': 'american'}
        response = requests.get(f"{ODDS_BASE_URL}/sports/basketball_nba/odds", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


def get_player_props(stat_type='points'):
    games = get_nba_games()
    if not games:
        return [], {}
    all_props, game_info = [], {}
    market = {'points': 'player_points', 'assists': 'player_assists', 'rebounds': 'player_rebounds'}.get(stat_type, 'player_points')
    for game in games[:12]:
        gid = game['id']
        spread, total = None, None
        for bk in game.get('bookmakers', []):
            for mk in bk.get('markets', []):
                if mk['key'] == 'spreads':
                    for oc in mk.get('outcomes', []):
                        if oc['name'] == game.get('home_team'):
                            spread = oc.get('point', 0)
                elif mk['key'] == 'totals':
                    for oc in mk.get('outcomes', []):
                        if oc['name'] == 'Over':
                            total = oc.get('point', 0)
        game_info[gid] = {'home_team': str(game.get('home_team', '')), 'away_team': str(game.get('away_team', '')), 'spread': spread, 'total': total}
        try:
            params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': market, 'oddsFormat': 'american'}
            response = requests.get(f"{ODDS_BASE_URL}/sports/basketball_nba/events/{gid}/odds", params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                player_best = {}
                for bk in data.get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        for oc in mk.get('outcomes', []):
                            p, ln, pr, tp = oc.get('description', ''), oc.get('point', 0), oc.get('price', 0), oc.get('name', '')
                            if p and ln:
                                if p not in player_best:
                                    player_best[p] = {'player': str(p), 'game_id': gid, 'stat_type': stat_type, 'lines': []}
                                player_best[p]['lines'].append({'book': str(bk['key']), 'line': float(ln), 'price': int(pr), 'type': str(tp)})
                all_props.extend(player_best.values())
        except:
            continue
    log_debug(f"Found {len(all_props)} {stat_type} props")
    return all_props, game_info


def analyze_player_prop(player_name, stat_type, line, opponent=None):
    player_info = search_player(player_name)
    if not player_info:
        return None
    games = get_player_game_log(player_info['id'], stat_type)
    if not games:
        return None
    analysis = analyze_game_log(games, line, opponent)
    if not analysis:
        return None
    return {'player_id': int(player_info['id']), 'player_name': str(player_info['name']),
            'team': str(player_info.get('team', 'N/A')), 'position': str(player_info.get('position', 'N/A')), 'analysis': analysis}


def deep_analyze_props(props, game_info, stat_type, min_edge=5):
    opps, analyzed = [], 0
    log_debug(f"Analyzing ALL {len(props)} props...")
    for prop in props:
        gi_data = game_info.get(prop['game_id'], {})
        opponent = gi_data.get('away_team', '') or gi_data.get('home_team', '')
        overs = [l for l in prop['lines'] if l['type'] == 'Over']
        unders = [l for l in prop['lines'] if l['type'] == 'Under']
        if not overs:
            continue
        best_over, best_under = min(overs, key=lambda x: x['line']), max(unders, key=lambda x: x['line']) if unders else None
        line = best_over['line']
        result = analyze_player_prop(prop['player'], stat_type, line, opponent)
        analyzed += 1
        if not result:
            continue
        a = result['analysis']
        adj_over, adj_under = a['adjusted_over_probability'], a['adjusted_under_probability']
        oe, ue = ((a['adjusted_mean'] - line) / line) * 100, ((line - a['adjusted_mean']) / line) * 100
        if adj_over > 52 and oe >= min_edge:
            rec, edge, bl, prob = 'OVER', oe, best_over, adj_over
        elif adj_under > 52 and ue >= min_edge:
            rec, edge, bl, prob = 'UNDER', ue, best_under, adj_under
        else:
            continue
        sc = (2 if edge >= 10 else (1 if edge >= 7 else 0)) + (2 if prob >= 65 else (1 if prob >= 55 else 0))
        sc += (1 if a['consistency'] >= 70 else 0) + (1 if a['chi_ok'] else 0) + (1 if a['games_analyzed'] >= 15 else 0)
        sc += (1 if a['r_squared'] >= 0.1 else 0) + (1 if not a['is_b2b'] else 0)
        conf = 'HIGH' if sc >= 6 else ('MEDIUM' if sc >= 3 else 'LOW')
        def_rating, opp_pace = get_defense_rating(opponent) if opponent else LEAGUE_AVG_DEF_RATING, get_team_pace(opponent) if opponent else LEAGUE_AVG_PACE
        def_info = get_defense_category(def_rating)
        opps.append({
            'player': str(result['player_name']), 'team': str(result.get('team', 'N/A')),
            'position': str(result.get('position', 'N/A')), 'stat_type': str(stat_type),
            'game_info': {'home_team': str(gi_data.get('home_team', '')), 'away_team': str(gi_data.get('away_team', '')), 'spread': gi_data.get('spread'), 'total': gi_data.get('total')},
            'line_analysis': {'bookmaker_line': float(bl['line']), 'bookmaker': str(bl['book']).upper(), 'odds': int(bl['price']),
                'recommendation': str(rec), 'edge': round(float(edge), 1), 'over_probability': float(a['over_probability']),
                'under_probability': float(a['under_probability']), 'adjusted_over_prob': float(a['adjusted_over_probability']),
                'adjusted_under_prob': float(a['adjusted_under_probability']), 'bet_confidence': str(conf), 'kelly_criterion': float(a['kelly_criterion'])},
            'context_factors': {'rest_days': int(a['rest_days']), 'is_b2b': a['is_b2b'], 'rest_impact': float(a['rest_impact']),
                'fatigue_impact': float(a['fatigue_impact']), 'defense_impact': float(a['defense_impact']),
                'pace_impact': float(a['pace_impact']), 'total_adjustment': float(a['total_adjustment'])},
            'defense_analysis': {'opponent': str(opponent) if opponent else 'Unknown', 'def_rating': float(def_rating),
                'opp_pace': float(opp_pace), 'category': str(def_info['category']), 'emoji': str(def_info['emoji'])},
            'deep_stats': {'mean': float(a['mean']), 'clean_mean': float(a['clean_mean']), 'adjusted_mean': float(a['adjusted_mean']),
                'median': float(a['median']), 'std': float(a['std']), 'clean_std': float(a['clean_std']),
                'min': float(a['min_val']), 'max': float(a['max_val']), 'avg_last_5': float(a['avg_last_5']),
                'avg_last_10': float(a['avg_last_10']), 'r_squared': float(a['r_squared']), 'trend_slope': float(a['trend_slope']),
                'trend': str(a['trend']), 'consistency': float(a['consistency']), 'games_analyzed': int(a['games_analyzed']),
                'outliers_removed': int(a['outliers_removed']), 'over_count': int(a['over_count']), 'under_count': int(a['under_count']),
                'chi_ok': a['chi_ok'], 'chi_p_value': float(a['chi_p_value'])}
        })
    log_debug(f"Analyzed {analyzed}, found {len(opps)} opportunities")
    opps.sort(key=lambda x: x['line_analysis']['edge'], reverse=True)
    return opps


@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    try:
        min_edge, min_conf = float(request.args.get('min_edge', 5)), request.args.get('min_confidence', 'LOW')
        stat_type = request.args.get('stat_type', 'points')
        if stat_type not in ['points', 'assists', 'rebounds']:
            stat_type = 'points'
        log_debug(f"=== SCAN v9.0: {stat_type} (ALL PLAYERS) ===")
        if not BALLDONTLIE_API_KEY:
            return jsonify({'status': 'ERROR', 'message': 'BALLDONTLIE_API_KEY not set'}), 500
        props, gi = get_player_props(stat_type)
        if not props:
            return jsonify({'status': 'SUCCESS', 'message': 'No props', 'opportunities': [], 'total_props': 0})
        opps = deep_analyze_props(props, gi, stat_type, min_edge)
        cl = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        filtered = [o for o in opps if cl.get(o['line_analysis']['bet_confidence'], 0) >= cl.get(min_conf, 1)]
        return jsonify(to_python({'status': 'SUCCESS', 'version': '9.0', 'stat_type': str(stat_type),
            'total_props': int(len(props)), 'players_analyzed': int(len(props)), 'opportunities_found': int(len(filtered)),
            'opportunities': filtered, 'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'season': '2025-26'}))
    except Exception as e:
        import traceback
        return jsonify({'status': 'ERROR', 'message': str(e), 'traceback': traceback.format_exc()[:500]}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'version': '9.0', 'features': ['all_players', 'rest_days', 'pace', 'fatigue']})


@app.route('/api/odds/usage', methods=['GET'])
def get_usage():
    try:
        r = requests.get(f"{ODDS_BASE_URL}/sports", params={'apiKey': ODDS_API_KEY}, timeout=10)
        return jsonify({'used': str(r.headers.get('x-requests-used', 'N/A')), 'remaining': str(r.headers.get('x-requests-remaining', 'N/A'))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return jsonify({'app': 'NBA Betting Analyzer', 'version': '9.0'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🏀 NBA Analyzer v9.0 - ALL PLAYERS")
    app.run(host='0.0.0.0', port=port, debug=False)