"""
NBA Betting Analyzer v3.0 - COMPLET
Inclut: R², p-values, tests de confiance, alertes minutes, volatilité
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)
CORS(app)

# Import nba_api
try:
    from nba_api.stats.endpoints import playergamelog, commonteamroster, teamgamelog
    from nba_api.stats.static import players, teams
    NBA_API_AVAILABLE = True
    print("✅ nba_api imported successfully")
except ImportError as e:
    NBA_API_AVAILABLE = False
    print(f"❌ nba_api not available: {e}")

class NBAAnalyzerV3:
    """
    Analyste NBA v3 avec:
    - R² et tests de significativité
    - Alertes temps de jeu
    - Indicateurs de confiance multiples
    - Volatilité des joueurs
    """
    
    def __init__(self):
        self.current_season = '2025-26'
        
        # Ratings défensifs
        self.defensive_ratings = {
            'ATL': 115.2, 'BOS': 110.5, 'BKN': 114.8, 'CHA': 116.1, 'CHI': 113.9,
            'CLE': 108.2, 'DAL': 112.7, 'DEN': 111.3, 'DET': 115.8, 'GSW': 112.1,
            'HOU': 110.9, 'IND': 116.5, 'LAC': 111.8, 'LAL': 113.4, 'MEM': 112.6,
            'MIA': 111.2, 'MIL': 112.4, 'MIN': 108.9, 'NOP': 114.3, 'NYK': 110.7,
            'OKC': 109.1, 'ORL': 108.6, 'PHI': 113.1, 'PHX': 114.2, 'POR': 115.7,
            'SAC': 114.9, 'SAS': 116.0, 'TOR': 115.4, 'UTA': 115.3, 'WAS': 116.8
        }
    
    def get_player_id(self, player_name):
        """Trouve l'ID d'un joueur"""
        if not NBA_API_AVAILABLE:
            return None
        
        try:
            all_players = players.get_players()
            player = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
            
            if player:
                return player[0]['id']
            return None
        except Exception as e:
            print(f"Error finding player: {e}")
            return None
    
    def get_season_games(self, player_id):
        """Récupère TOUS les matchs de la saison"""
        if not NBA_API_AVAILABLE or not player_id:
            return None
        
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=self.current_season
            )
            df = gamelog.get_data_frames()[0]
            
            if df.empty:
                return None
            
            # Convertit colonnes
            df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')
            df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # Home/Away
            df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')
            
            # Adversaire
            df['OPPONENT'] = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s*([A-Z]{3})')[0]
            
            # Trie par date (plus récent en premier)
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching games: {e}")
            return None
    
    def calculate_weighted_average(self, games_df):
        """
        Moyenne pondérée:
        - 10 derniers: 50%
        - 11-30: 30%
        - Reste: 20%
        """
        if games_df is None or len(games_df) == 0:
            return 0, 0
        
        points = games_df['PTS'].values
        n = len(points)
        
        weights = np.zeros(n)
        
        # 10 derniers
        if n >= 10:
            weights[:10] = 0.5 / 10
        else:
            weights[:n] = 0.5 / n
        
        # 11-30
        if n > 10:
            end_idx = min(30, n)
            count = end_idx - 10
            weights[10:end_idx] = 0.3 / count
        
        # Reste
        if n > 30:
            count = n - 30
            weights[30:] = 0.2 / count
        
        weights = weights / weights.sum()
        
        weighted_avg = np.average(points, weights=weights)
        weighted_std = np.sqrt(np.average((points - weighted_avg)**2, weights=weights))
        
        return float(weighted_avg), float(weighted_std)
    
    def calculate_splits(self, games_df, opponent=None):
        """Splits home/away/vs opponent"""
        if games_df is None or len(games_df) == 0:
            return {}
        
        splits = {}
        
        # Home
        home_games = games_df[games_df['IS_HOME'] == True]
        if len(home_games) > 0:
            home_avg = home_games['PTS'].mean()
            home_std = home_games['PTS'].std()
            splits['home'] = {
                'avg': round(float(home_avg), 1) if not pd.isna(home_avg) else 0.0,
                'std': round(float(home_std), 1) if not pd.isna(home_std) else 0.0,
                'games': int(len(home_games))
            }
        
        # Away
        away_games = games_df[games_df['IS_HOME'] == False]
        if len(away_games) > 0:
            away_avg = away_games['PTS'].mean()
            away_std = away_games['PTS'].std()
            splits['away'] = {
                'avg': round(float(away_avg), 1) if not pd.isna(away_avg) else 0.0,
                'std': round(float(away_std), 1) if not pd.isna(away_std) else 0.0,
                'games': int(len(away_games))
            }
        
        # vs opponent
        if opponent:
            vs_opp = games_df[games_df['OPPONENT'] == opponent]
            if len(vs_opp) > 0:
                opp_avg = vs_opp['PTS'].mean()
                opp_std = vs_opp['PTS'].std()
                splits['vs_opponent'] = {
                    'avg': round(float(opp_avg), 1) if not pd.isna(opp_avg) else 0.0,
                    'std': round(float(opp_std), 1) if not pd.isna(opp_std) else 0.0,
                    'games': int(len(vs_opp)),
                    'last_3': [float(x) for x in (vs_opp.head(3)['PTS'].tolist() if len(vs_opp) >= 3 else vs_opp['PTS'].tolist())]
                }
        
        return splits
    
    def calculate_trend_with_r2(self, games_df, num_games=10):
        """
        Calcule tendance + R² + p-value
        R² mesure la fiabilité de la tendance
        """
        if games_df is None or len(games_df) < 5:
            return {
                'slope': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0,
                'interpretation': 'Données insuffisantes',
                'reliable': False
            }
        
        try:
            recent = games_df.head(num_games)
            n = len(recent)
            
            X = np.arange(n).reshape(-1, 1)
            y = recent['PTS'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            
            # R²
            r2 = r2_score(y, y_pred)
            
            # P-value pour la pente
            residuals = y - y_pred
            s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
            
            # Erreur standard de la pente
            x_mean = X.mean()
            x_var = np.sum((X - x_mean)**2)
            se_slope = s_err / np.sqrt(x_var)
            
            # T-stat et p-value
            t_stat = model.coef_[0] / se_slope if se_slope > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            # Interprétation
            slope = float(model.coef_[0])
            
            if r2 < 0.3:
                interpretation = "Tendance non fiable (R² faible)"
                reliable = False
            elif r2 < 0.6:
                interpretation = "Tendance modérée"
                reliable = p_value < 0.05
            else:
                interpretation = "Tendance forte et fiable"
                reliable = True
            
            return {
                'slope': round(slope, 2),
                'r_squared': round(float(r2), 3),
                'p_value': round(float(p_value), 4),
                'interpretation': interpretation,
                'reliable': reliable,
                'sample_size': n
            }
            
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return {
                'slope': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0,
                'interpretation': 'Erreur calcul',
                'reliable': False
            }
    
    def calculate_minutes_stats(self, games_df):
        """
        Calcule stats minutes + alerte benching
        """
        if games_df is None or len(games_df) == 0:
            return None
        
        recent_10 = games_df.head(10)
        avg_minutes = float(recent_10['MIN'].mean())
        min_minutes = float(recent_10['MIN'].min())
        
        # Alerte
        alert = None
        alert_level = 'OK'
        
        if avg_minutes < 20:
            alert = "⚠️ DANGER: Temps de jeu très faible (<20 min). Risque benching élevé!"
            alert_level = 'CRITICAL'
        elif avg_minutes < 25:
            alert = "⚠️ PRUDENCE: Temps de jeu modéré (20-25 min). Surveiller le coach."
            alert_level = 'WARNING'
        elif avg_minutes < 30:
            alert = "ℹ️ Temps de jeu correct mais pas optimal (25-30 min)."
            alert_level = 'INFO'
        
        return {
            'avg_last_10': round(avg_minutes, 1),
            'min_last_10': round(min_minutes, 1),
            'alert': alert,
            'alert_level': alert_level
        }def calculate_confidence_score(self, games_df, splits, trend_stats, minutes_stats):
        """
        Score de confiance global basé sur plusieurs facteurs:
        - Sample size (nombre de matchs)
        - Écart-type (consistance)
        - R² de la tendance
        - Temps de jeu
        - Cohérence des splits
        
        Retourne score 0-100 et niveau (LOW/MEDIUM/HIGH)
        """
        score = 100
        factors = {}
        
        # 1. Sample size (max 25 pts)
        n_games = len(games_df)
        if n_games < 15:
            sample_penalty = (15 - n_games) * 2
            score -= sample_penalty
            factors['sample_size'] = f"-{sample_penalty} pts (seulement {n_games} matchs)"
        else:
            factors['sample_size'] = "+0 pts (sample OK)"
        
        # 2. Écart-type / Consistance (max 30 pts)
        std_dev = float(games_df['PTS'].std())
        if std_dev > 7.0:
            std_penalty = min(30, (std_dev - 7) * 4)
            score -= std_penalty
            factors['consistency'] = f"-{std_penalty:.0f} pts (écart-type {std_dev:.1f} trop élevé)"
        elif std_dev < 3.0:
            factors['consistency'] = "+5 pts (très consistant)"
            score += 5
        else:
            factors['consistency'] = "+0 pts (consistance OK)"
        
        # 3. R² tendance (max 20 pts)
        r2 = trend_stats['r_squared']
        if r2 < 0.3:
            score -= 15
            factors['trend_quality'] = f"-15 pts (R²={r2:.3f} - tendance peu fiable)"
        elif r2 > 0.7:
            factors['trend_quality'] = f"+5 pts (R²={r2:.3f} - tendance forte)"
            score += 5
        else:
            factors['trend_quality'] = f"+0 pts (R²={r2:.3f} - tendance modérée)"
        
        # 4. Minutes de jeu (max 20 pts)
        if minutes_stats:
            avg_min = minutes_stats['avg_last_10']
            if avg_min < 20:
                score -= 20
                factors['playing_time'] = f"-20 pts ({avg_min:.1f} min - benching risk)"
            elif avg_min < 25:
                score -= 10
                factors['playing_time'] = f"-10 pts ({avg_min:.1f} min - rotation limitée)"
            else:
                factors['playing_time'] = f"+0 pts ({avg_min:.1f} min OK)"
        
        # 5. Cohérence splits home/away (max 15 pts)
        if 'home' in splits and 'away' in splits:
            home_avg = splits['home']['avg']
            away_avg = splits['away']['avg']
            overall_avg = float(games_df['PTS'].mean())
            
            # Si écart home/away > 20% de la moyenne, c'est suspect
            diff = abs(home_avg - away_avg)
            if diff > overall_avg * 0.25:
                score -= 15
                factors['split_consistency'] = f"-15 pts (écart H/A de {diff:.1f} pts trop large)"
            else:
                factors['split_consistency'] = "+0 pts (splits cohérents)"
        
        # Score final 0-100
        final_score = max(0, min(100, score))
        
        # Niveau
        if final_score >= 75:
            level = 'HIGH'
        elif final_score >= 60:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {
            'score': round(final_score, 1),
            'level': level,
            'factors': factors,
            'recommendation': self._get_confidence_recommendation(final_score)
        }
    
    def _get_confidence_recommendation(self, score):
        """Recommandation basée sur le score"""
        if score >= 80:
            return "✅ Excellente fiabilité - Bet avec confiance"
        elif score >= 70:
            return "✅ Bonne fiabilité - Bet recommandé"
        elif score >= 60:
            return "⚠️ Fiabilité moyenne - Bet avec prudence"
        elif score >= 50:
            return "⚠️ Fiabilité faible - Réduire la mise"
        else:
            return "❌ Fiabilité insuffisante - SKIP ce bet"
    
    def adjust_for_matchup(self, base_prediction, opponent, is_home, splits):
        """Ajuste prédiction selon matchup"""
        adjusted = base_prediction
        adjustments = {}
        
        # 1. Défense adverse
        avg_rating = 113.0
        opp_rating = self.defensive_ratings.get(opponent, avg_rating)
        defense_factor = opp_rating / avg_rating
        adjusted *= defense_factor
        adjustments['defense'] = f"{((defense_factor - 1) * 100):.1f}%"
        
        # 2. Home/Away
        if is_home and 'home' in splits:
            home_diff = splits['home']['avg'] - base_prediction
            adjusted += home_diff * 0.5
            adjustments['home_away'] = f"+{home_diff * 0.5:.1f} pts (home)"
        elif not is_home and 'away' in splits:
            away_diff = splits['away']['avg'] - base_prediction
            adjusted += away_diff * 0.5
            adjustments['home_away'] = f"{away_diff * 0.5:+.1f} pts (away)"
        
        # 3. vs Opponent
        if 'vs_opponent' in splits and splits['vs_opponent']['games'] >= 3:
            opp_avg = splits['vs_opponent']['avg']
            opp_diff = opp_avg - base_prediction
            weight = min(splits['vs_opponent']['games'] / 10, 0.3)
            adjusted += opp_diff * weight
            adjustments['vs_opponent'] = f"{opp_diff * weight:+.1f} pts (history)"
        
        return float(adjusted), adjustments
    
    def predict_points(self, player_name, opponent, is_home=True, line=None):
        """Prédiction COMPLÈTE avec tous les indicateurs"""
        if not NBA_API_AVAILABLE:
            return {
                'error': 'nba_api not available',
                'player': player_name,
                'status': 'API_UNAVAILABLE'
            }
        
        # 1. Joueur
        player_id = self.get_player_id(player_name)
        if not player_id:
            return {
                'error': f'Player not found: {player_name}',
                'status': 'PLAYER_NOT_FOUND'
            }
        
        # 2. Games
        season_games = self.get_season_games(player_id)
        if season_games is None or len(season_games) < 10:
            return {
                'error': 'Not enough games this season',
                'player': player_name,
                'status': 'INSUFFICIENT_DATA'
            }
        
        # 3. Moyenne pondérée
        weighted_avg, weighted_std = self.calculate_weighted_average(season_games)
        
        # 4. Splits
        splits = self.calculate_splits(season_games, opponent)
        
        # 5. Tendance + R²
        trend_stats = self.calculate_trend_with_r2(season_games, num_games=10)
        
        # 6. Minutes
        minutes_stats = self.calculate_minutes_stats(season_games)
        
        # 7. Score de confiance global
        confidence_analysis = self.calculate_confidence_score(
            season_games, splits, trend_stats, minutes_stats
        )
        
        # 8. Prédiction base
        base_prediction = weighted_avg
        if trend_stats['reliable']:
            base_prediction += (trend_stats['slope'] * 1.5)
        
        # 9. Ajustements matchup
        final_prediction, adjustment_details = self.adjust_for_matchup(
            base_prediction, opponent, is_home, splits
        )
        
        # 10. Intervalle confiance 95%
        n = len(season_games)
        se = weighted_std / np.sqrt(n)
        confidence_interval = stats.t.interval(
            0.95,
            n - 1,
            loc=final_prediction,
            scale=se
        )
        
        # 11. Analyse ligne bookmaker
        recommendation = None
        over_probability = None
        edge = None
        
        if line is not None:
            z_score = (line - final_prediction) / weighted_std
            over_probability = 1 - stats.norm.cdf(z_score)
            edge = over_probability - 0.5
            
            # Recommandation
            if over_probability >= 0.58 and edge >= 0.08:
                recommendation = 'OVER'
            elif over_probability <= 0.42 and edge <= -0.08:
                recommendation = 'UNDER'
            else:
                recommendation = 'SKIP'
        
        # 12. RÉSULTAT COMPLET
        result = {
            'player': player_name,
            'opponent': opponent,
            'is_home': is_home,
            'prediction': round(final_prediction, 1),
            
            'confidence_interval': {
                'lower': round(float(confidence_interval[0]), 1),
                'upper': round(float(confidence_interval[1]), 1),
                'confidence_level': '95%',
                'width': round(float(confidence_interval[1] - confidence_interval[0]), 1)
            },
            
            'season_stats': {
                'weighted_avg': round(weighted_avg, 1),
                'std_dev': round(weighted_std, 1),
                'games_played': len(season_games),
                'consistency_level': 'Excellent' if weighted_std < 3 else 'Bon' if weighted_std < 5 else 'Moyen' if weighted_std < 7 else 'Faible'
            },
            
            'trend_analysis': trend_stats,
            
            'minutes_stats': minutes_stats,
            
            'confidence_score': confidence_analysis,
            
            'splits': splits,
            
            'adjustments': adjustment_details,
            
            'timestamp': datetime.now().isoformat()
        }
        
        if line is not None:
            result['line_analysis'] = {
                'bookmaker_line': float(line),
                'over_probability': round(float(over_probability), 3),
                'under_probability': round(float(1 - over_probability), 3),
                'edge': round(float(edge), 3),
                'recommendation': recommendation,
                'bet_confidence': 'HIGH' if abs(edge) >= 0.12 else 'MEDIUM' if abs(edge) >= 0.08 else 'LOW',
                'kelly_criterion': round(float(edge * 2), 3) if abs(edge) >= 0.08 else 0.0
            }
        
        result['status'] = 'SUCCESS'
        return result


# Initialize analyzer
analyzer = NBAAnalyzerV3()

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'NBA Betting Analyzer API v3.0',
        'status': 'online',
        'season': '2025-26',
        'new_features': [
            'R² et tests statistiques',
            'Score de confiance multi-facteurs',
            'Alertes temps de jeu',
            'Analyse cohérence splits',
            'P-values pour tendances'
        ]
    })

@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        'name': 'NBA Betting Analyzer API',
        'version': '3.0',
        'season': '2025-26',
        'data_source': 'nba_api (REAL DATA)',
        'nba_api_status': 'AVAILABLE' if NBA_API_AVAILABLE else 'UNAVAILABLE',
        'features': [
            'R² regression analysis',
            'P-values for trends',
            'Multi-factor confidence score',
            'Playing time alerts',
            'Split consistency checks',
            'Kelly Criterion sizing'
        ]
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'nba_api': NBA_API_AVAILABLE,
        'season': '2025-26',
        'version': '3.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_player():
    """Analyse complète v3"""
    try:
        data = request.json
        
        player = data.get('player')
        opponent = data.get('opponent')
        is_home = data.get('is_home', True)
        line = data.get('line')
        
        if not player or not opponent:
            return jsonify({
                'error': 'Missing required fields: player, opponent'
            }), 400
        
        result = analyzer.predict_points(player, opponent, is_home, line)
        
        if result.get('status') == 'SUCCESS':
            return jsonify(result)
        else:
            return jsonify(result), 404
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'ERROR'
        }), 500

@app.route('/api/player-stats/<player_name>', methods=['GET'])
def get_player_stats(player_name):
    """Stats complètes joueur"""
    try:
        player_id = analyzer.get_player_id(player_name)
        if not player_id:
            return jsonify({
                'error': f'Player not found: {player_name}',
                'status': 'PLAYER_NOT_FOUND'
            }), 404
        
        season_games = analyzer.get_season_games(player_id)
        if season_games is None:
            return jsonify({
                'error': 'No games found',
                'status': 'NO_DATA'
            }), 404
        
        # Tous les matchs
        all_games_list = []
        for idx, row in season_games.iterrows():
            all_games_list.append({
                'game_number': len(season_games) - idx,
                'date': str(row['GAME_DATE'].date()),
                'matchup': str(row['MATCHUP']),
                'opponent': str(row['OPPONENT']),
                'points': float(row['PTS']),
                'minutes': float(row.get('MIN', 0)) if not pd.isna(row.get('MIN', 0)) else 0,
                'is_home': bool(row['IS_HOME']),
                'location': 'Domicile' if row['IS_HOME'] else 'Extérieur'
            })
        
        # Stats
        recent_10 = season_games.head(10)['PTS'].mean()
        recent_5 = season_games.head(5)['PTS'].mean()
        home_games = season_games[season_games['IS_HOME'] == True]
        away_games = season_games[season_games['IS_HOME'] == False]
        
        # Volatilité
        volatility = float(season_games['PTS'].std())
        
        return jsonify({
            'player': player_name,
            'season': '2025-26',
            'total_games': int(len(season_games)),
            'date_range': {
                'first_game': str(season_games.iloc[-1]['GAME_DATE'].date()),
                'last_game': str(season_games.iloc[0]['GAME_DATE'].date())
            },
            'averages': {
                'season': round(float(season_games['PTS'].mean()), 1),
                'last_10': round(float(recent_10), 1),
                'last_5': round(float(recent_5), 1),
                'home': round(float(home_games['PTS'].mean()), 1) if len(home_games) > 0 else 0,
                'away': round(float(away_games['PTS'].mean()), 1) if len(away_games) > 0 else 0
            },
            'volatility': round(volatility, 2),
            'all_games': all_games_list,
            'status': 'SUCCESS'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'ERROR'
        }), 500

@app.route('/api/team-roster/<team_code>', methods=['GET'])
def get_team_roster(team_code):
    """Roster équipe avec volatilité"""
    if not NBA_API_AVAILABLE:
        return jsonify({
            'error': 'NBA API not available',
            'status': 'API_UNAVAILABLE'
        }), 503
    
    try:
        team_dict = {
            'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751, 'CHA': 1610612766,
            'CHI': 1610612741, 'CLE': 1610612739, 'DAL': 1610612742, 'DEN': 1610612743,
            'DET': 1610612765, 'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
            'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763, 'MIA': 1610612748,
            'MIL': 1610612749, 'MIN': 1610612750, 'NOP': 1610612740, 'NYK': 1610612752,
            'OKC': 1610612760, 'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
            'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'TOR': 1610612761,
            'UTA': 1610612762, 'WAS': 1610612764
        }
        
        team_id = team_dict.get(team_code.upper())
        if not team_id:
            return jsonify({
                'error': f'Invalid team code: {team_code}',
                'status': 'INVALID_TEAM'
            }), 400
        
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season='2025-26')
        roster_df = roster.get_data_frames()[0]
        
        roster_list = []
        for _, player in roster_df.iterrows():
            player_name = str(player['PLAYER'])
            player_id_val = analyzer.get_player_id(player_name)
            volatility = None
            
            if player_id_val:
                games = analyzer.get_season_games(player_id_val)
                if games is not None and len(games) >= 5:
                    volatility = float(games['PTS'].std())
            
            roster_list.append({
                'name': player_name,
                'position': str(player.get('POSITION', 'N/A')),
                'number': str(player.get('NUM', '')),
                'volatility': round(volatility, 2) if volatility else None
            })
        
        # Trie par volatilité (plus consistants en premier)
        roster_list.sort(key=lambda x: x['volatility'] if x['volatility'] else 999)
        
        gamelog = teamgamelog.TeamGameLog(team_id=team_id, season='2025-26')
        games_df = gamelog.get_data_frames()[0]
        
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        games_df = games_df.sort_values('GAME_DATE', ascending=False)
        
        last_game = games_df.iloc[0]
        matchup = str(last_game['MATCHUP'])
        is_home_val = 'vs.' in matchup
        opponent_val = matchup.split('vs.' if is_home_val else '@')[1].strip()
        
        next_game_info = {
            'last_game_date': str(last_game['GAME_DATE'].date()),
            'opponent': opponent_val,
            'is_home': is_home_val,
            'location': 'Domicile' if is_home_val else 'Extérieur'
        }
        
        return jsonify({
            'team': team_code.upper(),
            'roster': roster_list,
            'next_game': next_game_info,
            'status': 'SUCCESS'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'ERROR'
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)