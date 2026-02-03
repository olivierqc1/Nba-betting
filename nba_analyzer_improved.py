"""
NBA Betting Analyzer v8.6.0 - MANUAL LINE MODE
- NEW: /api/players-stats - Get player stats without bookmaker lines
- NEW: /api/analyze-manual - Analyze with YOUR line from BetOnline
- Stats from balldontlie API (reliable)
- No dependency on The Odds API for analysis
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

BALLDONTLIE_API_KEY = os.environ.get('BALLDONTLIE_API_KEY', '')
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

TOP_PLAYERS = [
    'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
    'Luka Doncic', 'Nikola Jokic', 'Joel Embiid', 'Jayson Tatum', 'Damian Lillard',
    'Anthony Davis', 'Devin Booker', 'Donovan Mitchell', 'Jimmy Butler', 'Kyrie Irving',
    'Trae Young', 'Ja Morant', 'Anthony Edwards', 'Tyrese Haliburton',
    'Shai Gilgeous-Alexander', 'LaMelo Ball', "De'Aaron Fox", 'Paolo Banchero',
    'Brandon Ingram', 'Julius Randle', 'Bam Adebayo', 'Karl-Anthony Towns',
    'Jaylen Brown', 'Jalen Brunson', 'Chet Holmgren', 'Victor Wembanyama',
    'Scottie Barnes', 'Domantas Sabonis', 'Lauri Markkanen', 'Alperen Sengun'
]


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
            'stl': float(stats.get('stl') or 0),
            'blk': float(stats.get('blk') or 0),
            'fg3m': float(stats.get('fg3m') or 0),
            'gp': int(stats.get('games_played') or 0),
            'min': str(stats.get('min') or '0')
        }
    return None


def get_player_game_log(player_id, stat_type='points'):
    stat_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb', 
                'steals': 'stl', 'blocks': 'blk', 'threes': 'fg3m'}
    stat_col = stat_map.get(stat_type, 'pts')
    data = bdl_request("stats", {"player_ids[]": int(player_id), "seasons[]": 2024, "per_page": 30})
    if not data or 'data' not in data:
        log_debug(f"No stats data returned for player {player_id}")
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
    log_debug(f"Got {len(games)} games for player {player_id}, stat_col={stat_col}")
    return games if games else None


def analyze_game_log(games, line, season_avg_fallback=None):
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
    
    if season_avg_fallback is not None and season_avg_fallback > 1:
        if clean_mean < 1 or abs(clean_mean - season_avg_fallback) > season_avg_fallback * 0.5:
            log_debug(f"clean_mean={clean_mean} invalid, using season_avg={season_avg_fallback}")
            clean_mean = season_avg_fallback
            mean_val = season_avg_fallback if mean_val < 1 else mean_val
            median_val = season_avg_fallback if median_val < 1 else median_val
    
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
        if recent_5 < 1 and season_avg_fallback:
            recent_5 = season_avg_fallback
        if prev_5 < 1 and season_avg_fallback:
            prev_5 = season_avg_fallback
        if recent_5 > prev_5 * 1.05:
            trend_dir = 'UP'
        elif recent_5 < prev_5 * 0.95:
            trend_dir = 'DOWN'
        else:
            trend_dir = 'STABLE'
        avg_last_5 = round(recent_5, 1)
        avg_last_10 = round(float(np.mean(arr[:10])), 1)
        if avg_last_10 < 1 and season_avg_fallback:
            avg_last_10 = round(season_avg_fallback, 1)
    else:
        trend_dir = 'STABLE'
        avg_last_5 = round(float(np.mean(arr[:5])), 1) if len(arr) >= 5 else round(mean_val, 1)
        avg_last_10 = round(mean_val, 1)
        if avg_last_5 < 1 and season_avg_fallback:
            avg_last_5 = round(season_avg_fallback, 1)
    
    return {
        'games_analyzed': int(total),
        'mean': round(mean_val, 1),
        'median': round(median_val, 1),
        'std': round(std_val, 2),
        'clean_mean': round(clean_mean, 1),
        'outliers_removed': int(len(values) - len(clean)),
        'avg_last_5': avg_last_5,
        'avg_last_10': avg_last_10,
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


def get_player_full_stats(player_name, stat_type='points'):
    player_info = search_player(player_name)
    if not player_info:
        log_debug(f"Player not found: {player_name}")
        return None
    player_id = player_info['id']
    
    season_avg = get_player_season_avg(player_id)
    stat_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb',
                'steals': 'stl', 'blocks': 'blk', 'threes': 'fg3m'}
    stat_key = stat_map.get(stat_type, 'pts')
    season_val = float(season_avg.get(stat_key, 0)) if season_avg else 0
    games_played = int(season_avg.get('gp', 0)) if season_avg else 0
    
    games = get_player_game_log(player_id, stat_type)
    if not games:
        return {
            'player_id': int(player_id),
            'player_name': str(player_info['name']),
            'team': str(player_info.get('team', 'N/A')),
            'stat_type': stat_type,
            'season_avg': round(season_val, 1),
            'games_played': games_played,
            'has_game_log': False
        }
    
    values = [float(g['stat']) for g in games if g['stat'] is not None]
    arr = np.array(values, dtype=float)
    
    q1_val = float(np.percentile(arr, 25))
    q3_val = float(np.percentile(arr, 75))
    iqr = q3_val - q1_val
    clean = arr[(arr >= q1_val-1.5*iqr) & (arr <= q3_val+1.5*iqr)]
    clean_mean = float(np.mean(clean)) if len(clean) > 0 else float(np.mean(arr))
    
    if clean_mean < 1 and season_val > 1:
        clean_mean = season_val
    
    avg_last_5 = float(np.mean(arr[:5])) if len(arr) >= 5 else float(np.mean(arr))
    avg_last_10 = float(np.mean(arr[:10])) if len(arr) >= 10 else float(np.mean(arr))
    
    return {
        'player_id': int(player_id),
        'player_name': str(player_info['name']),
        'team': str(player_info.get('team', 'N/A')),
        'stat_type': stat_type,
        'season_avg': round(season_val, 1),
        'clean_mean': round(clean_mean, 1),
        'median': round(float(np.median(arr)), 1),
        'avg_last_5': round(avg_last_5, 1),
        'avg_last_10': round(avg_last_10, 1),
        'std': round(float(np.std(arr)), 2),
        'min': round(float(np.min(arr)), 1),
        'max': round(float(np.max(arr)), 1),
        'games_played': games_played if games_played > 0 else len(games),
        'games_analyzed': len(values),
        'has_game_log': True,
        'recent_games': [{'stat': round(g['stat'], 1), 'date': g['date'][:10]} for g in games[:5]]
    }
# ============ PART 2 - PASTE BELOW PART 1 ============


@app.route('/api/analyze-manual', methods=['GET'])
def analyze_manual():
    """
    MAIN ENDPOINT: Analyze a player with YOUR BetOnline line
    
    Usage: /api/analyze-manual?player=LeBron James&line=25.5&stat=points
    
    Parameters:
    - player: Player name (required)
    - line: Your BetOnline line (required)
    - stat: points, assists, rebounds, steals, blocks, threes (default: points)
    - opponent: Optional opponent team for defense analysis
    """
    try:
        player_name = request.args.get('player', '')
        line = request.args.get('line', '')
        stat_type = request.args.get('stat', 'points')
        opponent = request.args.get('opponent', '')
        
        if not player_name:
            return jsonify({'status': 'ERROR', 'message': 'Missing player parameter'}), 400
        if not line:
            return jsonify({'status': 'ERROR', 'message': 'Missing line parameter'}), 400
        
        try:
            line = float(line)
        except:
            return jsonify({'status': 'ERROR', 'message': 'Invalid line value'}), 400
        
        if stat_type not in ['points', 'assists', 'rebounds', 'steals', 'blocks', 'threes']:
            stat_type = 'points'
        
        log_debug(f"=== MANUAL ANALYSIS v8.6.0 ===")
        log_debug(f"Player: {player_name}, Line: {line}, Stat: {stat_type}")
        
        # Search player
        player_info = search_player(player_name)
        if not player_info:
            return jsonify({'status': 'ERROR', 'message': f'Player not found: {player_name}'}), 404
        
        player_id = player_info['id']
        
        # Get season average
        season_avg = get_player_season_avg(player_id)
        stat_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb',
                    'steals': 'stl', 'blocks': 'blk', 'threes': 'fg3m'}
        stat_key = stat_map.get(stat_type, 'pts')
        season_val = float(season_avg.get(stat_key, 0)) if season_avg else 0
        games_played = int(season_avg.get('gp', 0)) if season_avg else 0
        
        # Get game log
        games = get_player_game_log(player_id, stat_type)
        if not games or len(games) < 5:
            return jsonify({
                'status': 'ERROR',
                'message': f'Not enough game data for {player_name} ({len(games) if games else 0} games)'
            }), 400
        
        # Full analysis with the user's line
        analysis = analyze_game_log(games, line, season_avg_fallback=season_val)
        if not analysis:
            return jsonify({'status': 'ERROR', 'message': 'Analysis failed'}), 500
        
        # Calculate edge
        clean_mean = analysis['clean_mean']
        over_edge = ((clean_mean - line) / line) * 100
        under_edge = ((line - clean_mean) / line) * 100
        
        # Cap at reasonable values
        over_edge = min(50, max(-50, over_edge))
        under_edge = min(50, max(-50, under_edge))
        
        # Recommendation
        if analysis['over_probability'] > 55 and over_edge > 3:
            recommendation = 'OVER'
            edge = over_edge
        elif analysis['under_probability'] > 55 and under_edge > 3:
            recommendation = 'UNDER'
            edge = under_edge
        elif analysis['over_probability'] > 50:
            recommendation = 'LEAN OVER'
            edge = over_edge
        else:
            recommendation = 'LEAN UNDER'
            edge = under_edge
        
        # Confidence score
        sc = 0
        sc += 2 if abs(edge) >= 10 else (1 if abs(edge) >= 5 else 0)
        sc += 2 if analysis['over_probability'] >= 65 or analysis['under_probability'] >= 65 else (1 if max(analysis['over_probability'], analysis['under_probability']) >= 55 else 0)
        sc += 1 if analysis['consistency'] >= 70 else 0
        sc += 1 if analysis['chi_ok'] else 0
        sc += 1 if games_played >= 20 else 0
        sc += 1 if analysis['r_squared'] >= 0.1 else 0
        confidence = 'HIGH' if sc >= 5 else ('MEDIUM' if sc >= 3 else 'LOW')
        
        # Defense analysis
        def_info = None
        if opponent:
            def_rating = get_defense_rating(opponent)
            def_info = get_defense_category(def_rating)
            def_info['opponent'] = opponent
            def_info['rating'] = def_rating
        
        result = {
            'status': 'SUCCESS',
            'player': {
                'name': str(player_info['name']),
                'team': str(player_info.get('team', 'N/A')),
                'id': int(player_id)
            },
            'input': {
                'line': float(line),
                'stat_type': str(stat_type),
                'source': 'BETONLINE (manual)'
            },
            'stats': {
                'season_avg': round(season_val, 1),
                'clean_mean': round(clean_mean, 1),
                'median': float(analysis['median']),
                'avg_last_5': float(analysis['avg_last_5']),
                'avg_last_10': float(analysis['avg_last_10']),
                'std': float(analysis['std']),
                'min': float(analysis['min_val']),
                'max': float(analysis['max_val']),
                'games_played': int(games_played),
                'games_analyzed': int(analysis['games_analyzed'])
            },
            'analysis': {
                'recommendation': str(recommendation),
                'edge': round(float(edge), 1),
                'over_edge': round(float(over_edge), 1),
                'under_edge': round(float(under_edge), 1),
                'over_probability': float(analysis['over_probability']),
                'under_probability': float(analysis['under_probability']),
                'over_count': int(analysis['over_count']),
                'under_count': int(analysis['under_count']),
                'confidence': str(confidence),
                'kelly_criterion': float(analysis['kelly_criterion'])
            },
            'trend': {
                'direction': str(analysis['trend']),
                'slope': float(analysis['trend_slope']),
                'r_squared': float(analysis['r_squared']),
                'consistency': float(analysis['consistency'])
            },
            'recent_games': [{'stat': round(g['stat'], 1), 'date': g['date'][:10]} for g in games[:5]]
        }
        
        if def_info:
            result['defense'] = def_info
        
        return jsonify(to_python(result))
        
    except Exception as e:
        import traceback
        log_debug(f"ERROR: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e), 'debug_log': DEBUG_LOG[-10:]}), 500


@app.route('/api/player-stats', methods=['GET'])
def player_stats():
    """
    Get player stats WITHOUT requiring a line
    
    Usage: /api/player-stats?player=LeBron James&stat=points
    """
    try:
        player_name = request.args.get('player', '')
        stat_type = request.args.get('stat', 'points')
        
        if not player_name:
            return jsonify({'status': 'ERROR', 'message': 'Missing player parameter'}), 400
        
        if stat_type not in ['points', 'assists', 'rebounds', 'steals', 'blocks', 'threes']:
            stat_type = 'points'
        
        log_debug(f"Player stats request: {player_name}, {stat_type}")
        
        stats = get_player_full_stats(player_name, stat_type)
        if not stats:
            return jsonify({'status': 'ERROR', 'message': f'Player not found: {player_name}'}), 404
        
        return jsonify(to_python({'status': 'SUCCESS', 'data': stats}))
        
    except Exception as e:
        log_debug(f"ERROR: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/top-players', methods=['GET'])
def top_players():
    """
    Get stats for top NBA players (quick scan)
    
    Usage: /api/top-players?stat=points&limit=10
    """
    try:
        stat_type = request.args.get('stat', 'points')
        limit = min(int(request.args.get('limit', 10)), 20)
        
        if stat_type not in ['points', 'assists', 'rebounds', 'steals', 'blocks', 'threes']:
            stat_type = 'points'
        
        log_debug(f"=== TOP PLAYERS SCAN v8.6.0 ===")
        
        results = []
        for player_name in TOP_PLAYERS[:limit]:
            try:
                stats = get_player_full_stats(player_name, stat_type)
                if stats and stats.get('has_game_log'):
                    results.append(stats)
                    log_debug(f"Got stats for {player_name}")
            except Exception as e:
                log_debug(f"Error for {player_name}: {e}")
                continue
        
        # Sort by season average descending
        results.sort(key=lambda x: x.get('season_avg', 0), reverse=True)
        
        return jsonify(to_python({
            'status': 'SUCCESS',
            'stat_type': stat_type,
            'players_found': len(results),
            'players': results,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }))
        
    except Exception as e:
        log_debug(f"ERROR: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/quick-compare', methods=['GET'])
def quick_compare():
    """
    Compare multiple players against lines from BetOnline
    
    Usage: /api/quick-compare?data=LeBron:25.5,Curry:28.5,Jokic:30.5&stat=points
    
    Format: player1:line1,player2:line2,...
    """
    try:
        data = request.args.get('data', '')
        stat_type = request.args.get('stat', 'points')
        
        if not data:
            return jsonify({'status': 'ERROR', 'message': 'Missing data parameter. Format: player1:line1,player2:line2'}), 400
        
        if stat_type not in ['points', 'assists', 'rebounds', 'steals', 'blocks', 'threes']:
            stat_type = 'points'
        
        log_debug(f"=== QUICK COMPARE v8.6.0 ===")
        
        results = []
        pairs = data.split(',')
        
        for pair in pairs[:10]:  # Max 10 players
            try:
                parts = pair.strip().split(':')
                if len(parts) != 2:
                    continue
                player_name = parts[0].strip()
                line = float(parts[1].strip())
                
                player_info = search_player(player_name)
                if not player_info:
                    results.append({'player': player_name, 'error': 'Not found'})
                    continue
                
                player_id = player_info['id']
                season_avg = get_player_season_avg(player_id)
                stat_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb',
                            'steals': 'stl', 'blocks': 'blk', 'threes': 'fg3m'}
                stat_key = stat_map.get(stat_type, 'pts')
                season_val = float(season_avg.get(stat_key, 0)) if season_avg else 0
                
                games = get_player_game_log(player_id, stat_type)
                if not games or len(games) < 5:
                    results.append({'player': player_name, 'error': 'Not enough games'})
                    continue
                
                analysis = analyze_game_log(games, line, season_avg_fallback=season_val)
                if not analysis:
                    results.append({'player': player_name, 'error': 'Analysis failed'})
                    continue
                
                edge = ((analysis['clean_mean'] - line) / line) * 100
                edge = min(50, max(-50, edge))
                
                rec = 'OVER' if analysis['over_probability'] > 55 and edge > 3 else (
                    'UNDER' if analysis['under_probability'] > 55 and edge < -3 else 'PASS')
                
                results.append({
                    'player': str(player_info['name']),
                    'team': str(player_info.get('team', 'N/A')),
                    'line': float(line),
                    'season_avg': round(season_val, 1),
                    'clean_mean': round(float(analysis['clean_mean']), 1),
                    'edge': round(float(edge), 1),
                    'over_prob': float(analysis['over_probability']),
                    'recommendation': str(rec)
                })
                
            except Exception as e:
                log_debug(f"Error processing {pair}: {e}")
                continue
        
        # Sort by absolute edge
        results.sort(key=lambda x: abs(x.get('edge', 0)), reverse=True)
        
        return jsonify(to_python({
            'status': 'SUCCESS',
            'stat_type': stat_type,
            'comparisons': len(results),
            'results': results
        }))
        
    except Exception as e:
        log_debug(f"ERROR: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/debug', methods=['GET'])
def debug_endpoint():
    r = {'timestamp': datetime.now().isoformat(), 'version': '8.6.0', 'mode': 'MANUAL LINE ENTRY'}
    r['env_check'] = {
        'BALLDONTLIE_API_KEY': True if BALLDONTLIE_API_KEY else False,
        'bdl_key_length': int(len(BALLDONTLIE_API_KEY)) if BALLDONTLIE_API_KEY else 0
    }
    if BALLDONTLIE_API_KEY:
        player = search_player("LeBron James")
        r['tests'] = {'player_search': {'success': True if player else False, 'player': player}}
        if player:
            season_avg = get_player_season_avg(player['id'])
            r['tests']['season_avg'] = {'success': True if season_avg else False, 'data': season_avg}
    r['debug_log'] = DEBUG_LOG[-20:]
    r['endpoints'] = [
        'GET /api/analyze-manual?player=NAME&line=VALUE&stat=points',
        'GET /api/player-stats?player=NAME&stat=points',
        'GET /api/top-players?stat=points&limit=10',
        'GET /api/quick-compare?data=Player1:line1,Player2:line2&stat=points'
    ]
    return jsonify(to_python(r))


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'version': '8.6.0',
        'mode': 'MANUAL_LINE_ENTRY',
        'season': '2024-25',
        'bdl_key_set': True if BALLDONTLIE_API_KEY else False,
        'description': 'Enter YOUR BetOnline lines for analysis'
    })


@app.route('/')
def home():
    return jsonify({
        'app': 'NBA Betting Analyzer',
        'version': '8.6.0',
        'mode': 'MANUAL LINE ENTRY FOR BETONLINE',
        'season': '2024-25',
        'usage': {
            'analyze': '/api/analyze-manual?player=LeBron James&line=25.5&stat=points',
            'stats': '/api/player-stats?player=LeBron James&stat=points',
            'top_players': '/api/top-players?stat=points&limit=10',
            'compare': '/api/quick-compare?data=LeBron:25.5,Curry:28.5&stat=points'
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting NBA Analyzer v8.6.0 - MANUAL LINE MODE")
    print(f"Enter YOUR BetOnline lines for accurate analysis!")
    app.run(host='0.0.0.0', port=port, debug=False) 