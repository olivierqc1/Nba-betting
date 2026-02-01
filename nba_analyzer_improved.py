"""
NBA Betting Analyzer v8.0 - HYBRID APPROACH
1. Bulk fetch all player averages (fast filter)
2. Deep analysis only on promising candidates (~10-15 players)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime
import os
import numpy as np
from scipy import stats as scipy_stats

app = Flask(__name__)
CORS(app)

ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

NBA_STATS_URL = "https://stats.nba.com/stats/leaguedashplayerstats"
NBA_GAME_LOG_URL = "https://stats.nba.com/stats/playergamelog"

NBA_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com'
}

PLAYER_ID_CACHE = {}

def get_all_player_averages():
    """Get ALL NBA players' season averages in ONE API call"""
    global PLAYER_ID_CACHE
    
    params = {
        'Conference': '', 'DateFrom': '', 'DateTo': '', 'Division': '',
        'GameScope': '', 'GameSegment': '', 'Height': '', 'LastNGames': '0',
        'LeagueID': '00', 'Location': '', 'MeasureType': 'Base', 'Month': '0',
        'OpponentTeamID': '0', 'Outcome': '', 'PORound': '0', 'PaceAdjust': 'N',
        'PerMode': 'PerGame', 'Period': '0', 'PlayerExperience': '',
        'PlayerPosition': '', 'PlusMinus': 'N', 'Rank': 'N', 'Season': '2024-25',
        'SeasonSegment': '', 'SeasonType': 'Regular Season', 'ShotClockRange': '',
        'StarterBench': '', 'TeamID': '0', 'TwoWay': '0', 'VsConference': '',
        'VsDivision': '', 'Weight': ''
    }
    
    try:
        response = requests.get(NBA_STATS_URL, headers=NBA_HEADERS, params=params, timeout=15)
        if response.status_code != 200:
            return {}
        
        data = response.json()
        headers = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        
        id_idx = headers.index('PLAYER_ID')
        name_idx = headers.index('PLAYER_NAME')
        pts_idx = headers.index('PTS')
        ast_idx = headers.index('AST')
        reb_idx = headers.index('REB')
        gp_idx = headers.index('GP')
        min_idx = headers.index('MIN')
        
        players = {}
        for row in rows:
            name = row[name_idx]
            player_id = row[id_idx]
            name_norm = normalize_name(name)
            
            PLAYER_ID_CACHE[name_norm] = player_id
            
            players[name_norm] = {
                'id': player_id,
                'name': name,
                'pts': row[pts_idx],
                'ast': row[ast_idx],
                'reb': row[reb_idx],
                'gp': row[gp_idx],
                'min': row[min_idx]
            }
        
        print(f"Loaded {len(players)} players")
        return players
        
    except Exception as e:
        print(f"Error: {e}")
        return {}

def get_player_game_log(player_id, stat_type='points'):
    """Get detailed game log for deep analysis"""
    stat_map = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB'}
    stat_col = stat_map.get(stat_type, 'PTS')
    
    params = {
        'PlayerID': player_id,
        'Season': '2024-25',
        'SeasonType': 'Regular Season',
        'LeagueID': '00'
    }
    
    try:
        response = requests.get(NBA_GAME_LOG_URL, headers=NBA_HEADERS, params=params, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        headers = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        
        if not rows:
            return None
        
        stat_idx = headers.index(stat_col)
        min_idx = headers.index('MIN')
        matchup_idx = headers.index('MATCHUP')
        date_idx = headers.index('GAME_DATE')
        
        games = []
        for row in rows:
            mins = row[min_idx]
            if isinstance(mins, str) and ':' in mins:
                parts = mins.split(':')
                mins = int(parts[0]) + int(parts[1])/60
            
            games.append({
                'date': row[date_idx],
                'matchup': row[matchup_idx],
                'stat': row[stat_idx],
                'minutes': float(mins) if mins else 0
            })
        
        return games
        
    except Exception as e:
        print(f"Error game log {player_id}: {e}")
        return None

def analyze_game_log(games, line, stat_type):
    """Deep statistical analysis"""
    if not games or len(games) < 5:
        return None
    
    values = [g['stat'] for g in games if g['stat'] is not None]
    if len(values) < 5:
        return None
    
    values = np.array(values, dtype=float)
    
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    
    last_5 = values[:5] if len(values) >= 5 else values
    last_10 = values[:10] if len(values) >= 10 else values
    avg_last_5 = np.mean(last_5)
    avg_last_10 = np.mean(last_10)
    
    if len(values) >= 10:
        recent = np.mean(values[:5])
        older = np.mean(values[5:10])
        trend = "📈 UP" if recent > older * 1.05 else ("📉 DOWN" if recent < older * 0.95 else "➡️ STABLE")
    else:
        trend = "N/A"
    
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    clean_values = values[(values >= lower_bound) & (values <= upper_bound)]
    clean_mean = np.mean(clean_values) if len(clean_values) > 0 else mean
    outliers_removed = len(values) - len(clean_values)
    
    consistency = max(0, 100 - (std / mean * 100)) if mean > 0 else 50
    
    over_count = np.sum(values > line)
    under_count = np.sum(values <= line)
    total = len(values)
    over_prob = (over_count / total) * 100
    under_prob = (under_count / total) * 100
    
    try:
        expected = [total / 2, total / 2]
        observed = [over_count, under_count]
        chi2, p_value = scipy_stats.chisquare(observed, expected)
        chi_ok = p_value > 0.05
    except:
        chi2, p_value = 0, 1
        chi_ok = True
    
    if over_prob > 50:
        implied_prob = 0.524
        edge = (over_prob / 100) - implied_prob
        kelly = max(0, edge / (1/implied_prob - 1)) * 100
        kelly = min(kelly, 25)
    else:
        kelly = 0
    
    return {
        'games_analyzed': len(values),
        'mean': round(mean, 1),
        'median': round(median, 1),
        'std': round(std, 2),
        'clean_mean': round(clean_mean, 1),
        'outliers_removed': outliers_removed,
        'avg_last_5': round(avg_last_5, 1),
        'avg_last_10': round(avg_last_10, 1),
        'trend': trend,
        'consistency': round(consistency, 1),
        'over_probability': round(over_prob, 1),
        'under_probability': round(under_prob, 1),
        'over_count': int(over_count),
        'under_count': int(under_count),
        'chi_square': round(chi2, 2),
        'chi_p_value': round(p_value, 3),
        'chi_ok': chi_ok,
        'kelly_criterion': round(kelly, 1)
    }

def get_nba_games():
    url = f"{ODDS_BASE_URL}/sports/basketball_nba/odds"
    params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

def get_player_props(stat_type='points'):
    games = get_nba_games()
    if not games:
        return [], {}
    
    all_props = []
    game_info = {}
    market_map = {'points': 'player_points', 'assists': 'player_assists', 'rebounds': 'player_rebounds'}
    market = market_map.get(stat_type, 'player_points')
    
    for game in games[:10]:
        game_id = game['id']
        game_info[game_id] = {
            'home_team': game.get('home_team', ''),
            'away_team': game.get('away_team', ''),
            'commence_time': game.get('commence_time', '')
        }
        
        url = f"{ODDS_BASE_URL}/sports/basketball_nba/events/{game_id}/odds"
        params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': market, 'oddsFormat': 'american'}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                player_best = {}
                
                for bookmaker in data.get('bookmakers', []):
                    book_name = bookmaker['key']
                    for market_data in bookmaker.get('markets', []):
                        for outcome in market_data.get('outcomes', []):
                            player = outcome.get('description', '')
                            line = outcome.get('point', 0)
                            price = outcome.get('price', 0)
                            bet_type = outcome.get('name', '')
                            
                            if not player or not line:
                                continue
                            
                            if player not in player_best:
                                player_best[player] = {'player': player, 'game_id': game_id, 'stat_type': stat_type, 'lines': []}
                            
                            player_best[player]['lines'].append({'book': book_name, 'line': line, 'price': price, 'type': bet_type})
                
                all_props.extend(player_best.values())
        except:
            continue
    
    return all_props, game_info

def normalize_name(name):
    name = name.lower().strip()
    for suffix in [' jr.', ' sr.', ' iii', ' ii', ' iv', '.']:
        name = name.replace(suffix, '')
    return name.strip()

def quick_filter(props, player_stats, min_edge=3):
    candidates = []
    stat_key_map = {'points': 'pts', 'assists': 'ast', 'rebounds': 'reb'}
    
    for prop in props:
        player_name = prop['player']
        stat_type = prop['stat_type']
        lines = prop['lines']
        
        name_norm = normalize_name(player_name)
        player_data = player_stats.get(name_norm)
        
        if not player_data:
            for key, data in player_stats.items():
                if name_norm in key or key in name_norm:
                    player_data = data
                    break
        
        if not player_data:
            continue
        
        stat_key = stat_key_map.get(stat_type, 'pts')
        season_avg = player_data.get(stat_key, 0)
        games_played = player_data.get('gp', 0)
        player_id = player_data.get('id')
        
        if not season_avg or games_played < 10 or not player_id:
            continue
        
        over_lines = [l for l in lines if l['type'] == 'Over']
        under_lines = [l for l in lines if l['type'] == 'Under']
        
        if not over_lines:
            continue
        
        best_over = min(over_lines, key=lambda x: x['line'])
        best_under = max(under_lines, key=lambda x: x['line']) if under_lines else None
        
        over_edge = ((season_avg - best_over['line']) / best_over['line']) * 100
        under_edge = ((best_under['line'] - season_avg) / best_under['line']) * 100 if best_under else 0
        
        if over_edge >= min_edge or under_edge >= min_edge:
            candidates.append({
                'prop': prop,
                'player_data': player_data,
                'player_id': player_id,
                'season_avg': season_avg,
                'games_played': games_played,
                'best_over': best_over,
                'best_under': best_under,
                'over_edge': over_edge,
                'under_edge': under_edge,
                'over_lines': over_lines
            })
    
    candidates.sort(key=lambda x: max(x['over_edge'], x['under_edge']), reverse=True)
    return candidates[:15]

def deep_analyze(candidates, game_info, min_edge=5):
    opportunities = []
    
    for c in candidates:
        prop = c['prop']
        player_id = c['player_id']
        stat_type = prop['stat_type']
        best_over = c['best_over']
        best_under = c['best_under']
        
        games = get_player_game_log(player_id, stat_type)
        if not games:
            continue
        
        analysis = analyze_game_log(games, best_over['line'], stat_type)
        if not analysis:
            continue
        
        over_prob = analysis['over_probability']
        under_prob = analysis['under_probability']
        clean_mean = analysis['clean_mean']
        
        over_edge = ((clean_mean - best_over['line']) / best_over['line']) * 100
        under_edge = ((best_under['line'] - clean_mean) / best_under['line']) * 100 if best_under else 0
        
        if over_prob > 55 and over_edge >= min_edge:
            recommendation = 'OVER'
            edge = over_edge
            probability = over_prob
            best_line = best_over
        elif under_prob > 55 and under_edge >= min_edge:
            recommendation = 'UNDER'
            edge = under_edge
            probability = under_prob
            best_line = best_under
        else:
            continue
        
        confidence_score = 0
        if edge >= 10: confidence_score += 2
        elif edge >= 7: confidence_score += 1
        if probability >= 65: confidence_score += 2
        elif probability >= 55: confidence_score += 1
        if analysis['consistency'] >= 70: confidence_score += 1
        if analysis['chi_ok']: confidence_score += 1
        if c['games_played'] >= 30: confidence_score += 1
        
        if confidence_score >= 5:
            confidence = 'HIGH'
        elif confidence_score >= 3:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        opp = {
            'player': prop['player'],
            'stat_type': stat_type,
            'game_info': game_info.get(prop['game_id'], {}),
            'season_avg': round(c['season_avg'], 1),
            'games_played': c['games_played'],
            'line_analysis': {
                'bookmaker_line': best_line['line'],
                'bookmaker': best_line['book'].upper(),
                'odds': best_line['price'],
                'recommendation': recommendation,
                'edge': round(edge, 1),
                'over_probability': round(over_prob, 1),
                'under_probability': round(under_prob, 1),
                'bet_confidence': confidence,
                'kelly_criterion': analysis['kelly_criterion']
            },
            'deep_stats': {
                'mean': analysis['mean'],
                'clean_mean': analysis['clean_mean'],
                'median': analysis['median'],
                'std': analysis['std'],
                'avg_last_5': analysis['avg_last_5'],
                'avg_last_10': analysis['avg_last_10'],
                'trend': analysis['trend'],
                'consistency': analysis['consistency'],
                'games_analyzed': analysis['games_analyzed'],
                'outliers_removed': analysis['outliers_removed'],
                'over_count': analysis['over_count'],
                'under_count': analysis['under_count']
            }
        }
        opportunities.append(opp)
    
    opportunities.sort(key=lambda x: x['line_analysis']['edge'], reverse=True)
    return opportunities

@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    try:
        min_edge = float(request.args.get('min_edge', 5))
        min_confidence = request.args.get('min_confidence', 'LOW')
        stat_type = request.args.get('stat_type', 'points')
        
        if stat_type not in ['points', 'assists', 'rebounds']:
            stat_type = 'points'
        
        print("Step 1: Bulk fetch averages...")
        player_stats = get_all_player_averages()
        
        if not player_stats:
            return jsonify({'status': 'ERROR', 'message': 'Could not fetch NBA stats'}), 500
        
        print(f"Step 2: Fetching {stat_type} props...")
        props, game_info = get_player_props(stat_type)
        
        if not props:
            return jsonify({'status': 'SUCCESS', 'message': 'No props available', 'opportunities': [], 'total_props': 0})
        
        print("Step 3: Quick filter...")
        candidates = quick_filter(props, player_stats, min_edge=3)
        
        if not candidates:
            return jsonify({'status': 'SUCCESS', 'opportunities': [], 'total_props': len(props), 'candidates_found': 0})
        
        print(f"Step 4: Deep analysis on {len(candidates)} candidates...")
        opportunities = deep_analyze(candidates, game_info, min_edge)
        
        confidence_levels = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        min_conf_level = confidence_levels.get(min_confidence, 1)
        
        filtered = [o for o in opportunities if confidence_levels.get(o['line_analysis']['bet_confidence'], 0) >= min_conf_level]
        
        return jsonify({
            'status': 'SUCCESS',
            'stat_type': stat_type,
            'total_props': len(props),
            'candidates_analyzed': len(candidates),
            'opportunities_found': len(filtered),
            'opportunities': filtered,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500

@app.route('/api/odds/usage', methods=['GET'])
def get_usage():
    url = f"{ODDS_BASE_URL}/sports"
    params = {'apiKey': ODDS_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        return jsonify({
            'used': response.headers.get('x-requests-used', 'N/A'),
            'remaining': response.headers.get('x-requests-remaining', 'N/A'),
            'status': 'OK'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'version': '8.0-hybrid'})

@app.route('/')
def home():
    return jsonify({'app': 'NBA Betting Analyzer', 'version': '8.0', 'method': 'Hybrid: bulk filter + deep analysis'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)