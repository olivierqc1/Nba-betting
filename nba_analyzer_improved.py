def deep_analyze_props(props, game_info, stat_type, min_edge=5):
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
            'game_info': {
                'home_team': str(gi_data.get('home_team', '')),
                'away_team': str(gi_data.get('away_team', ''))
            },
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
        log_debug(f"BDL key set: {bool(BALLDONTLIE_API_KEY)}")
        if not BALLDONTLIE_API_KEY:
            return jsonify({'status': 'ERROR', 'message': 'BALLDONTLIE_API_KEY not set', 'debug_log': DEBUG_LOG[-15:]}), 500
        props, gi = get_player_props(stat_type)
        if not props:
            return jsonify({'status': 'SUCCESS', 'message': 'No props available', 'opportunities': [], 'total_props': 0, 'debug_log': DEBUG_LOG[-15:]})
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
            'api_source': 'balldontlie.io',
            'season': '2024-25',
            'debug_log': DEBUG_LOG[-15:]
        }
        return jsonify(to_python(result))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_debug(f"ERROR: {e}")
        log_debug(tb[:300])
        return jsonify({'status': 'ERROR', 'message': str(e), 'traceback': tb[:500], 'debug_log': DEBUG_LOG[-20:]}), 500


@app.route('/api/debug', methods=['GET'])
def debug_endpoint():
    r = {'timestamp': datetime.now().isoformat(), 'tests': {}, 'version': '8.5.5'}
    r['env_check'] = {
        'BALLDONTLIE_API_KEY': True if BALLDONTLIE_API_KEY else False,
        'ODDS_API_KEY': True if ODDS_API_KEY else False,
        'bdl_key_length': int(len(BALLDONTLIE_API_KEY)) if BALLDONTLIE_API_KEY else 0
    }
    if BALLDONTLIE_API_KEY:
        player = search_player("LeBron James")
        r['tests']['balldontlie'] = {'success': True if player else False, 'api_key_set': True, 'test_player': player}
        if player:
            games = get_player_game_log(player['id'], 'points')
            r['tests']['game_log'] = {'success': True if games else False, 'games_found': int(len(games)) if games else 0}
    else:
        r['tests']['balldontlie'] = {'success': False, 'api_key_set': False, 'message': 'Set BALLDONTLIE_API_KEY env var'}
    games = get_nba_games()
    r['tests']['odds_api_games'] = {'success': True if len(games) > 0 else False, 'games_found': int(len(games)), 'games': [{'home': str(g.get('home_team', '')), 'away': str(g.get('away_team', ''))} for g in games[:3]]}
    if games:
        props, _ = get_player_props('points')
        r['tests']['odds_api_props'] = {'success': True if len(props) > 0 else False, 'props_found': int(len(props)), 'sample_players': [str(p['player']) for p in props[:5]]}
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
        'data_source': 'balldontlie.io',
        'season': '2024-25',
        'bdl_key_set': True if BALLDONTLIE_API_KEY else False,
        'bdl_key_length': int(len(BALLDONTLIE_API_KEY)) if BALLDONTLIE_API_KEY else 0
    })


@app.route('/')
def home():
    return jsonify({
        'app': 'NBA Betting Analyzer',
        'version': '8.5.5',
        'season': '2024-25',
        'data_source': 'balldontlie.io',
        'bdl_key_set': True if BALLDONTLIE_API_KEY else False
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting NBA Analyzer v8.5.5")
    print(f"BALLDONTLIE_API_KEY set: {bool(BALLDONTLIE_API_KEY)}")
    print(f"ODDS_API_KEY set: {bool(ODDS_API_KEY)}")
    app.run(host='0.0.0.0', port=port, debug=False)def deep_analyze_props(props, game_info, stat_type, min_edge=5):
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
            'game_info': {
                'home_team': str(gi_data.get('home_team', '')),
                'away_team': str(gi_data.get('away_team', ''))
            },
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
        log_debug(f"BDL key set: {bool(BALLDONTLIE_API_KEY)}")
        if not BALLDONTLIE_API_KEY:
            return jsonify({'status': 'ERROR', 'message': 'BALLDONTLIE_API_KEY not set', 'debug_log': DEBUG_LOG[-15:]}), 500
        props, gi = get_player_props(stat_type)
        if not props:
            return jsonify({'status': 'SUCCESS', 'message': 'No props available', 'opportunities': [], 'total_props': 0, 'debug_log': DEBUG_LOG[-15:]})
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
            'api_source': 'balldontlie.io',
            'season': '2024-25',
            'debug_log': DEBUG_LOG[-15:]
        }
        return jsonify(to_python(result))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_debug(f"ERROR: {e}")
        log_debug(tb[:300])
        return jsonify({'status': 'ERROR', 'message': str(e), 'traceback': tb[:500], 'debug_log': DEBUG_LOG[-20:]}), 500


@app.route('/api/debug', methods=['GET'])
def debug_endpoint():
    r = {'timestamp': datetime.now().isoformat(), 'tests': {}, 'version': '8.5.5'}
    r['env_check'] = {
        'BALLDONTLIE_API_KEY': True if BALLDONTLIE_API_KEY else False,
        'ODDS_API_KEY': True if ODDS_API_KEY else False,
        'bdl_key_length': int(len(BALLDONTLIE_API_KEY)) if BALLDONTLIE_API_KEY else 0
    }
    if BALLDONTLIE_API_KEY:
        player = search_player("LeBron James")
        r['tests']['balldontlie'] = {'success': True if player else False, 'api_key_set': True, 'test_player': player}
        if player:
            games = get_player_game_log(player['id'], 'points')
            r['tests']['game_log'] = {'success': True if games else False, 'games_found': int(len(games)) if games else 0}
    else:
        r['tests']['balldontlie'] = {'success': False, 'api_key_set': False, 'message': 'Set BALLDONTLIE_API_KEY env var'}
    games = get_nba_games()
    r['tests']['odds_api_games'] = {'success': True if len(games) > 0 else False, 'games_found': int(len(games)), 'games': [{'home': str(g.get('home_team', '')), 'away': str(g.get('away_team', ''))} for g in games[:3]]}
    if games:
        props, _ = get_player_props('points')
        r['tests']['odds_api_props'] = {'success': True if len(props) > 0 else False, 'props_found': int(len(props)), 'sample_players': [str(p['player']) for p in props[:5]]}
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
        'data_source': 'balldontlie.io',
        'season': '2024-25',
        'bdl_key_set': True if BALLDONTLIE_API_KEY else False,
        'bdl_key_length': int(len(BALLDONTLIE_API_KEY)) if BALLDONTLIE_API_KEY else 0
    })


@app.route('/')
def home():
    return jsonify({
        'app': 'NBA Betting Analyzer',
        'version': '8.5.5',
        'season': '2024-25',
        'data_source': 'balldontlie.io',
        'bdl_key_set': True if BALLDONTLIE_API_KEY else False
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting NBA Analyzer v8.5.5")
    print(f"BALLDONTLIE_API_KEY set: {bool(BALLDONTLIE_API_KEY)}")
    print(f"ODDS_API_KEY set: {bool(ODDS_API_KEY)}")
    app.run(host='0.0.0.0', port=port, debug=False)